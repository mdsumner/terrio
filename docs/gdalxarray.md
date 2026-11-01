# gdalxarray

`terrio.gdalxarray` brings [gdalxarray](https://github.com/hypertidy/gdalxarray)'s xarray engine
to terrio, without GDAL: the engine `terrio_gdalxarray` takes gdalxarray's `open_dataset`
arguments and gives the Datasets gdalxarray 0.5.1 gives, with pixels read lazily by terrio.

```sh
pip install "terrio[gdalxarray]"   # terrio, xarray, rasterix, xproj and dask
```

```python
import xarray as xr

ds = xr.open_dataset("dem.tif", engine="terrio_gdalxarray", multidim=False)
ds["band_data"]          # (band, y, x), nodata masked and scale/offset applied
ds.xindexes["x"]         # rasterix RasterIndex
ds.proj.crs              # xproj CRSIndex on spatial_ref

ds = xr.open_dataset("dem.tif", engine="terrio_gdalxarray", multidim=False, chunks={})  # dask, by block
ds = xr.open_dataset("bands.tif", engine="terrio_gdalxarray", multidim=False, band_as_dim=False)
```

## Supported arguments

| Argument | Behavior |
|----------|----------|
| `multidim` | Only `False` (classic rasters) so far; `True`, gdalxarray's default, raises `NotImplementedError`. |
| `band_as_dim` | `True`: one `band_data` variable with a `band` dimension. `False`: one `(y, x)` variable per band, named after its description or `band_N`. |
| `mask_and_scale` | CF decoding of `_FillValue`, `scale_factor` and `add_offset` (default `True`). Bands with differing nodata, scales or offsets keep raw values with per-band `band_nodata`, `band_scale_factor` and `band_add_offset` coordinates, and a warning, as in gdalxarray. |
| `chunks` | `None`: lazy without dask. `{}`: the raster's block sizes. A mapping: those sizes. |
| `drop_variables` | Variables to omit. |
| `config_options` | GDAL configuration options terrio reads (`AWS_NO_SIGN_REQUEST`, `GDAL_HTTP_*`...), set for the reading thread around the open and every read, as [`terrio.rio.Env`](rasterio.md) sets them. |
| `group`, `decode_times`, `decode_timedelta`, `use_cftime`, `concat_characters`, `decode_coords` | Accepted; unused for classic rasters. |

Paths are those terrio opens: local files, URLs, `/vsicurl/`, `/vsis3/` and the other prefixes in
[Remote Data](remote-data.md). `gdalxarray.warp()` and the multidimensional data model are not
provided yet.

## Code written for gdalxarray

The engine is named `terrio_gdalxarray`, not `gdalxarray`, so that gdalxarray and terrio can be
installed together. `terrio.compat.install()` makes `import gdalxarray` (and
`gdalxarray.backend`) give `terrio.gdalxarray`, whose `GDALBackendEntrypoint` is terrio's engine
class; code written for gdalxarray then needs its `engine=` argument changed:

```python
import terrio.compat

terrio.compat.install()

import gdalxarray
import xarray as xr

ds = xr.open_dataset("dem.tif", engine=gdalxarray.GDALBackendEntrypoint, multidim=False)
```

`engine="terrio_gdalxarray"` is chosen by name only: the engine does not claim files for xarray's
automatic engine detection.

## Comparison with gdalxarray

`tests/test_gdalxarray_vs_gdalxarray.py` opens files written with GDAL, and the GeoTIFFs of
`TERRIO_TEST_DATA`, with both engines and compares the Datasets with
`xarray.testing.assert_identical`, with the encodings of each variable.
