# Written to match the behavior of:
#   gdalxarray 0.5.1:
#     gdalxarray/
# No code is translated from these sources.

"""gdalxarray's API on terrio, without GDAL.

The xarray engine `terrio_gdalxarray` takes gdalxarray's `open_dataset` arguments and gives its
Dataset layout: `band_data (band, y, x)` (or one variable per band with `band_as_dim=False`), a
rasterix `RasterIndex` on `x`/`y` and an xproj `CRSIndex` on `spatial_ref`::

    import xarray as xr

    ds = xr.open_dataset("dem.tif", engine="terrio_gdalxarray", multidim=False)

The engine is not named `gdalxarray`, so that both can be installed. `terrio.compat.install()`
makes `import gdalxarray` give this module; code written for gdalxarray then needs
`engine="terrio_gdalxarray"` (or `engine=gdalxarray.GDALBackendEntrypoint`).

Only classic rasters (`multidim=False`) are read so far. Requires xarray, rasterix and xproj, as
gdalxarray does (and dask for `chunks`). The API follows gdalxarray (Apache-2.0, Copyright
2025-2026 Michael Sumner).
"""

from __future__ import annotations

try:
    import rasterix as _rasterix  # type: ignore[import-untyped]  # noqa: F401
    import xarray as _xarray  # noqa: F401
    import xproj as _xproj  # type: ignore[import-untyped]  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "terrio.gdalxarray needs xarray, rasterix and xproj: pip install 'terrio[gdalxarray]'"
    ) from e

from terrio.gdalxarray.backend import GDALBackendEntrypoint, TerrioGdalxarrayEntrypoint

#: The gdalxarray version whose behavior this module matches.
__version__ = "0.5.1"

__all__ = ["GDALBackendEntrypoint", "TerrioGdalxarrayEntrypoint", "__version__"]
