"""terrio.gdalxarray on synthetic rasters: the classic-raster layout of gdalxarray's engine.

Comparisons with gdalxarray itself are in `test_gdalxarray_vs_gdalxarray.py`; these tests need
xarray, dask, rasterix and xproj only.
"""

import pickle
import subprocess
import sys
import textwrap

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("dask")
pytest.importorskip("rasterix")
pytest.importorskip("xproj")

from affine import Affine  # noqa: E402
from rasterix import RasterIndex  # noqa: E402

import terrio.rio as rio  # noqa: E402
from terrio.gdalxarray import GDALBackendEntrypoint, TerrioGdalxarrayEntrypoint  # noqa: E402

TRANSFORM = Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4100000.0)
ENGINE = "terrio_gdalxarray"


def _surface(h=300, w=400):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.sin(xx / 23.0) * np.cos(yy / 17.0) * 1000 + 1500


def _write(path, arr, **kw):
    arr = arr if arr.ndim == 3 else arr[np.newaxis]
    meta = {k: kw.pop(k) for k in ("scales", "offsets", "descriptions") if k in kw}
    profile = dict(
        driver="GTiff",
        width=arr.shape[2],
        height=arr.shape[1],
        count=arr.shape[0],
        dtype=arr.dtype,
        crs="EPSG:32611",
        transform=TRANSFORM,
        tiled=True,
        blockxsize=64,
        blockysize=128,
    )
    profile.update(kw)
    with rio.open(path, "w", **profile) as dst:
        dst.write(arr)
        for k, v in meta.items():
            setattr(dst, k, v)
    return str(path)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("gdalxarray")
    dem = _surface().astype("int16")
    dem[200:230, 300:360] = -32768
    rgb = np.stack([(_surface() / 12 + i * 20) % 250 + 1 for i in range(3)]).astype("uint8")
    rgb[:, 10:40, 20:80] = 0
    return {
        "dem": _write(
            d / "dem.tif",
            dem,
            nodata=-32768,
            scales=[0.5],
            offsets=[100.0],
            descriptions=["elevation"],
        ),
        "rgb": _write(d / "rgb.tif", rgb, nodata=0, descriptions=["red", "green", "blue"]),
        "plain": _write(d / "plain.tif", rgb[:2]),
        "mixed": _write(d / "mixed.tif", rgb[:2], scales=[1.0, 2.0]),
    }


def test_engine_is_registered_and_aliased():
    assert GDALBackendEntrypoint is TerrioGdalxarrayEntrypoint
    assert ENGINE in xr.backends.list_engines()
    assert not TerrioGdalxarrayEntrypoint().guess_can_open("dem.tif")


def test_multidim_default_is_refused(files):
    with pytest.raises(NotImplementedError, match="multidim=False"):
        xr.open_dataset(files["dem"], engine=ENGINE)


def test_band_as_dim_layout(files):
    ds = xr.open_dataset(files["rgb"], engine=ENGINE, multidim=False)
    da = ds["band_data"]
    assert da.dims == ("band", "y", "x")
    assert da.shape == (3, 300, 400)
    assert list(ds["band"].values) == [1, 2, 3]
    assert da.attrs["descriptions"] == ["red", "green", "blue"]
    assert da.dtype == np.float32  # masked by CF decoding
    assert da.encoding["_FillValue"] == 0
    assert da.encoding["preferred_chunks"] == {"band": 1, "y": 128, "x": 64}
    assert isinstance(ds.xindexes["x"], RasterIndex)
    assert ds.proj.crs.to_epsg() == 32611
    assert ds.encoding["gdal_driver"] == "GTiff"
    assert ds.encoding["source"] == files["rgb"]
    values = da.isel(y=slice(10, 40), x=slice(20, 80)).values
    assert np.isnan(values).all()
    np.testing.assert_allclose(ds["x"].values[:2], [500015.0, 500045.0])
    np.testing.assert_allclose(ds["y"].values[:2], [4099985.0, 4099955.0])


def test_scale_offset_decoding(files):
    ds = xr.open_dataset(files["dem"], engine=ENGINE, multidim=False)
    raw = xr.open_dataset(files["dem"], engine=ENGINE, multidim=False, mask_and_scale=False)
    assert raw["band_data"].dtype == np.int16
    assert raw["band_data"].attrs["scale_factor"] == 0.5
    assert raw["band_data"].attrs["add_offset"] == 100.0
    assert raw["band_data"].attrs["_FillValue"] == -32768
    r = raw["band_data"].values.astype("float64")
    expected = np.where(r == -32768, np.nan, r * 0.5 + 100.0)
    np.testing.assert_allclose(ds["band_data"].values, expected)


def test_identity_scale_keeps_dtype(files):
    ds = xr.open_dataset(files["plain"], engine=ENGINE, multidim=False)
    assert ds["band_data"].dtype == np.uint8
    assert "_FillValue" not in ds["band_data"].encoding


def test_differing_band_scales_stay_raw(files):
    with pytest.warns(UserWarning, match="differing scale"):
        ds = xr.open_dataset(files["mixed"], engine=ENGINE, multidim=False)
    assert ds["band_data"].dtype == np.uint8
    assert list(ds["band_scale_factor"].values) == [1.0, 2.0]


def test_bands_as_variables(files):
    ds = xr.open_dataset(files["rgb"], engine=ENGINE, multidim=False, band_as_dim=False)
    assert list(ds.data_vars) == ["red", "green", "blue"]
    assert ds["green"].dims == ("y", "x")
    with rio.open(files["rgb"]) as src:
        green = src.read(2).astype("float32")
    green[green == 0] = np.nan
    np.testing.assert_array_equal(ds["green"].values, green)
    plain = xr.open_dataset(files["plain"], engine=ENGINE, multidim=False, band_as_dim=False)
    assert list(plain.data_vars) == ["band_1", "band_2"]


def test_drop_variables(files):
    ds = xr.open_dataset(
        files["rgb"], engine=ENGINE, multidim=False, band_as_dim=False, drop_variables=["red"]
    )
    assert list(ds.data_vars) == ["green", "blue"]
    ds = xr.open_dataset(files["rgb"], engine=ENGINE, multidim=False, drop_variables="band_data")
    assert not ds.data_vars


def test_chunks(files):
    ds = xr.open_dataset(files["rgb"], engine=ENGINE, multidim=False, chunks={})
    assert ds["band_data"].chunks == ((1, 1, 1), (128, 128, 44), (64,) * 6 + (16,))
    ds = xr.open_dataset(files["rgb"], engine=ENGINE, multidim=False, chunks={"x": 100})
    assert ds["band_data"].chunks[2] == (100,) * 4
    eager = xr.open_dataset(files["rgb"], engine=ENGINE, multidim=False)
    xr.testing.assert_identical(ds.compute(), eager.compute())


def test_config_options_and_pickle(files):
    ds = xr.open_dataset(
        files["rgb"],
        engine=ENGINE,
        multidim=False,
        config_options={"GDAL_HTTP_MAX_RETRY": 3},
    )
    again = pickle.loads(pickle.dumps(ds))
    xr.testing.assert_identical(again.compute(), ds.compute())


def test_unopenable_path_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="Could not open"):
        xr.open_dataset(str(tmp_path / "missing.tif"), engine=ENGINE, multidim=False)


def test_compat_aliases_gdalxarray():
    """`import gdalxarray` gives terrio.gdalxarray once terrio.compat is installed."""
    code = """
        import sys
        import terrio.compat
        terrio.compat.install()
        import gdalxarray
        from gdalxarray.backend import GDALBackendEntrypoint
        assert gdalxarray is sys.modules["terrio.gdalxarray"]
        assert GDALBackendEntrypoint is sys.modules["terrio.gdalxarray"].GDALBackendEntrypoint
        assert terrio.compat.installed()["gdalxarray"] == "terrio.gdalxarray"
        """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
