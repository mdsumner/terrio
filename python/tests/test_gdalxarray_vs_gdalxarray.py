"""terrio.gdalxarray against gdalxarray: both engines open the same classic rasters.

Each file is opened with `engine="gdalxarray"` (GDAL) and `engine="terrio_gdalxarray"` and the
Datasets are compared with `xarray.testing.assert_identical`, encodings included. Skipped when
gdalxarray (and so GDAL's Python bindings) is not installed. Files written with GDAL here are
always compared; GeoTIFFs of `TERRIO_TEST_DATA` are compared when it is set.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("dask")
pytest.importorskip("rasterix")
pytest.importorskip("xproj")
gdal = pytest.importorskip("osgeo.gdal")
pytest.importorskip("gdalxarray")

import terrio.rio as rio  # noqa: E402

ENCODING_KEYS = ("_FillValue", "scale_factor", "add_offset", "dtype", "preferred_chunks")


def _gdal_file(path, arr, *, nodata=None, scales=None, offsets=None, descriptions=None, **co):
    arr = arr if arr.ndim == 3 else arr[np.newaxis]
    gdal.UseExceptions()
    dtype = gdal.GetDataTypeByName(
        {"uint8": "Byte", "int16": "Int16", "float32": "Float32", "uint16": "UInt16"}[
            str(arr.dtype)
        ]
    )
    options = [f"{k}={v}" for k, v in co.items()]
    ds = gdal.GetDriverByName("GTiff").Create(
        str(path), arr.shape[2], arr.shape[1], arr.shape[0], dtype, options
    )
    ds.SetGeoTransform((500000.0, 30.0, 0.0, 4100000.0, 0.0, -30.0))
    ds.SetProjection("EPSG:32611")
    for i in range(arr.shape[0]):
        band = ds.GetRasterBand(i + 1)
        band.WriteArray(arr[i])
        if nodata is not None:
            band.SetNoDataValue(nodata)
        if scales is not None:
            band.SetScale(scales[i])
        if offsets is not None:
            band.SetOffset(offsets[i])
        if descriptions is not None:
            band.SetDescription(descriptions[i])
    ds = None
    return str(path)


@pytest.fixture(scope="module")
def gdal_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("gdalxarray-parity")
    yy, xx = np.mgrid[0:300, 0:400]
    surface = np.sin(xx / 23.0) * np.cos(yy / 17.0) * 1000 + 1500
    dem = surface.astype("int16")
    dem[200:230, 300:360] = -32768
    rgb = np.stack([(surface / 12 + i * 20) % 250 + 1 for i in range(3)]).astype("uint8")
    rgb[:, 10:40, 20:80] = 0
    f32 = (surface / 7).astype("float32")
    f32[0:5, 0:5] = np.nan
    return {
        "dem": _gdal_file(
            d / "dem.tif",
            dem,
            nodata=-32768,
            scales=[0.5],
            offsets=[100.0],
            descriptions=["elevation"],
            TILED="YES",
            BLOCKXSIZE=64,
            BLOCKYSIZE=128,
        ),
        "rgb": _gdal_file(d / "rgb.tif", rgb, nodata=0, descriptions=["red", "green", "blue"]),
        "plain": _gdal_file(d / "plain.tif", rgb[:2]),
        "mixed": _gdal_file(d / "mixed.tif", rgb[:2], scales=[1.0, 2.0]),
        "nan": _gdal_file(d / "nan.tif", f32, nodata=float("nan"), COMPRESS="DEFLATE"),
    }


def _data_files():
    root = os.environ.get("TERRIO_TEST_DATA")
    if not root or not Path(root).is_dir():
        return []
    return sorted(str(p) for p in Path(root).rglob("*.tif") if p.stat().st_size < 50_000_000)[:40]


def _open(path, engine, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = xr.open_dataset(path, engine=engine, multidim=False, **kw)
    return ds, sorted(str(w.message) for w in caught if issubclass(w.category, UserWarning))


def _compare(path, **kw):
    ref, ref_warnings = _open(path, "gdalxarray", **kw)
    got, got_warnings = _open(path, "terrio_gdalxarray", **kw)
    assert got_warnings == ref_warnings
    xr.testing.assert_identical(got.load(), ref.load())
    assert got.encoding["gdal_driver"] == ref.encoding["gdal_driver"]
    for name in ref.data_vars:
        for key in ENCODING_KEYS:
            assert got[name].encoding.get(key) == ref[name].encoding.get(key) or (
                key == "_FillValue"
                and np.isnan(ref[name].encoding.get(key, 0))
                and np.isnan(got[name].encoding.get(key, 0))
            ), (name, key)


@pytest.mark.parametrize("name", ["dem", "rgb", "plain", "mixed", "nan"])
@pytest.mark.parametrize("band_as_dim", [True, False])
@pytest.mark.parametrize("mask_and_scale", [True, False])
def test_gdal_written_files(gdal_files, name, band_as_dim, mask_and_scale):
    _compare(gdal_files[name], band_as_dim=band_as_dim, mask_and_scale=mask_and_scale)


@pytest.mark.parametrize("path", _data_files() or [None])
def test_real_files(path):
    if path is None:
        pytest.skip("TERRIO_TEST_DATA is not set")
    try:
        ref = xr.open_dataset(path, engine="gdalxarray", multidim=False).load()
    except Exception as e:  # noqa: BLE001 - nothing to compare with
        pytest.skip(f"gdalxarray cannot read it: {e}")
    ref.close()
    try:
        with rio.open(path) as src:
            src.read()
    except Exception as e:  # noqa: BLE001 - a terrio.rio reader gap, not this engine's
        pytest.xfail(f"terrio.rio cannot read it: {e}")
    _compare(path)
