# SPDX-FileCopyrightText: Copyright 2025-2026 Michael Sumner
# SPDX-FileCopyrightText: Copyright (c) 2026 Qiusheng Wu and Terrio contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Derived from (adapted and modified):
#   gdalxarray 0.5.1:
#     gdalxarray/backend.py

"""The `terrio_gdalxarray` xarray engine: gdalxarray's `open_dataset` on terrio.

Classic rasters open as gdalxarray opens them with GDAL's classic raster API (gdalxarray 0.5.1,
Apache-2.0, Copyright 2025-2026 Michael Sumner): the same variables, dimensions, attributes,
indexes and encoding. Pixels are read by window through terrio.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import os
import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
import xarray as xr
import xproj  # type: ignore[import-untyped]  # noqa: F401  (registers .proj)
from rasterix import RasterIndex  # type: ignore[import-untyped]
from xarray.backends import BackendArray, BackendEntrypoint
from xarray.core import indexing

from terrio.rio.env import Env
from terrio.xarray._io import TerrioArrayWrapper, _open_source, _Source


def _expand_tilde(dsn: Any) -> Any:
    """Expands a leading `~` of a path, and nothing else, so that URLs keep their slashes.

    Args:
        dsn: The path or URI.

    Returns:
        Any: The path, expanded.
    """
    if isinstance(dsn, os.PathLike):
        dsn = os.fspath(dsn)
    if isinstance(dsn, str) and dsn.startswith("~"):
        return os.path.expanduser(dsn)
    return dsn


def _config_context(
    config_options: dict[str, str] | None,
) -> contextlib.AbstractContextManager[Any]:
    """The configuration options, set for the calling thread only.

    Args:
        config_options: GDAL configuration option names and values.

    Returns:
        A context manager.
    """
    if not config_options:
        return contextlib.nullcontext()
    options: dict[str, Any] = {str(k): str(v) for k, v in config_options.items()}
    return Env(**options)


class _ConfiguredArray(BackendArray):
    """A terrio array read with configuration options set, on whichever thread reads it."""

    def __init__(self, array: TerrioArrayWrapper, config_options: dict[str, str]) -> None:
        """Creates the array.

        Args:
            array: The array to read.
            config_options: The options to set around each read.
        """
        self.array = array
        self.config_options = config_options
        self.shape = array.shape
        self.dtype = array.dtype

    def __getitem__(self, key: indexing.ExplicitIndexer) -> Any:
        """Reads the pixels of an indexer with the options set.

        Args:
            key: An xarray explicit indexer.

        Returns:
            numpy.ndarray: The pixels.
        """
        with _config_context(self.config_options):
            return self.array[key]


class TerrioGdalxarrayEntrypoint(BackendEntrypoint):
    """Opens rasters with terrio, as gdalxarray's `gdalxarray` engine opens them with GDAL."""

    description = "Open rasters with terrio, with gdalxarray's arguments and layout"
    url = "https://terrio.dev/gdalxarray/"

    def open_dataset(
        self,
        filename_or_obj: Any,
        *,
        drop_variables: str | Iterable[str] | None = None,
        chunks: dict[str, int] | None = None,
        multidim: bool = True,
        group: str | None = None,
        band_as_dim: bool = True,
        config_options: dict[str, Any] | None = None,
        mask_and_scale: bool | None = None,
        decode_times: Any = None,
        decode_timedelta: Any = None,
        use_cftime: Any = None,
        concat_characters: Any = None,
        decode_coords: Any = None,
    ) -> xr.Dataset:
        """Opens a raster as a Dataset.

        Args:
            filename_or_obj: A path or URL (`/vsicurl/`, `/vsis3/`... are accepted).
            drop_variables: Variables to omit.
            chunks: Chunk sizes (None: lazy without dask; `{}`: the raster's block sizes).
            multidim: Use the multidimensional data model (not supported yet: pass False).
            group: A group of a multidimensional dataset (unused for classic rasters).
            band_as_dim: Bands as a `band` dimension of `band_data` (True) or one variable each.
            config_options: GDAL configuration options, set around the open and every read.
            mask_and_scale: Apply CF masking and scaling (None: True).
            decode_times: Unused for classic rasters.
            decode_timedelta: Unused for classic rasters.
            use_cftime: Unused for classic rasters.
            concat_characters: Unused for classic rasters.
            decode_coords: Unused for classic rasters.

        Returns:
            xarray.Dataset: The raster.

        Raises:
            NotImplementedError: If `multidim` is True.
        """
        if multidim:
            raise NotImplementedError(
                "terrio_gdalxarray reads classic rasters only so far: pass multidim=False"
            )
        filename_or_obj = _expand_tilde(filename_or_obj)
        mask_and_scale = True if mask_and_scale is None else bool(mask_and_scale)
        options = {str(k): str(v) for k, v in (config_options or {}).items()}
        if isinstance(drop_variables, str):
            drop_variables = [drop_variables]
        with _config_context(options):
            ds = self._open_raster(
                filename_or_obj,
                drop_variables,
                band_as_dim=band_as_dim,
                mask_and_scale=mask_and_scale,
                config_options=options,
            )
        return _maybe_chunk_dataset(ds, chunks, filename_or_obj, options)

    def guess_can_open(self, filename_or_obj: Any) -> bool:
        """Never: the engine is chosen by name, so that it does not take files from `terrio`.

        Args:
            filename_or_obj: The path.

        Returns:
            bool: False.
        """
        return False

    def _open_raster(
        self,
        filename_or_obj: str,
        drop_variables: Iterable[str] | None,
        *,
        band_as_dim: bool,
        mask_and_scale: bool,
        config_options: dict[str, str],
    ) -> xr.Dataset:
        """Opens a classic raster.

        Args:
            filename_or_obj: The path.
            drop_variables: Variables to omit.
            band_as_dim: Bands as a dimension, or one variable each.
            mask_and_scale: Apply CF masking and scaling.
            config_options: Options to set around each read.

        Returns:
            xarray.Dataset: The raster.

        Raises:
            ValueError: If the raster has no bands.
        """
        try:
            src = _open_source(filename_or_obj, None)
        except Exception as e:
            raise ValueError(f"Could not open {filename_or_obj} with terrio: {e}") from e
        # terrio reads complex integers as complex64, as GDAL's ReadAsArray does.
        src = dataclasses.replace(src, dtypes=tuple(_NUMPY_DTYPES.get(d, d) for d in src.dtypes))
        if src.count == 0:
            if src.subdatasets:
                lines = [f"\n  {path}" for path in src.subdatasets]
                raise ValueError(
                    f"{filename_or_obj!r} has no raster bands at the top level "
                    f"but contains {len(src.subdatasets)} subdataset(s). "
                    f"Use multidim=True to read the full hierarchy as an xarray "
                    f"Dataset, or pass one of the subdataset paths below as the "
                    f"filename to read a single 2D view in classic mode:" + "".join(lines)
                )
            raise ValueError(
                f"{filename_or_obj!r} opens but has no raster bands and no "
                f"subdatasets. Cannot be read as a classic raster."
            )
        drop = set(drop_variables or ())
        if band_as_dim:
            ds = _raster_as_band_dim(src, drop, mask_and_scale, config_options)
        else:
            ds = _raster_as_vars(src, drop, mask_and_scale, config_options)
        index = RasterIndex.from_transform(src.transform, width=src.width, height=src.height)
        ds = ds.assign_coords(xr.Coordinates.from_xindex(index))
        if src.crs is not None:
            ds = ds.proj.assign_crs(crs=src.crs.to_wkt())
        ds.encoding["source"] = filename_or_obj
        if src.driver:
            ds.encoding["gdal_driver"] = src.driver
        return ds


#: The NumPy type of the pixel type NumPy does not have, as terrio reads it.
_NUMPY_DTYPES = {"complex_int16": "complex64"}

#: The name gdalxarray's API gives the engine class.
GDALBackendEntrypoint = TerrioGdalxarrayEntrypoint


def _lazy(src: _Source, bands: list[int], config_options: dict[str, str]) -> Any:
    """The raw pixels of bands as a lazy `(band, y, x)` array.

    Args:
        src: The raster.
        bands: The bands, from 1.
        config_options: Options to set around each read.

    Returns:
        xarray.core.indexing.LazilyIndexedArray: The array.
    """
    array: BackendArray = TerrioArrayWrapper(src, bands)
    if config_options:
        array = _ConfiguredArray(array, config_options)  # type: ignore[arg-type]
    return indexing.LazilyIndexedArray(array)


def _preferred_chunks(src: _Source, band: int) -> dict[str, int]:
    """The block size of a band, as dask chunks.

    Args:
        src: The raster.
        band: The band, from 1.

    Returns:
        dict: Chunk sizes of `y` and `x`.
    """
    rows, cols = src.block_shapes[band - 1] if src.block_shapes else (0, 0)
    return {
        "y": rows if rows > 0 else src.height,
        "x": cols if cols > 0 else src.width,
    }


def _description(src: _Source, band: int) -> str:
    """A band's description, or `band_N`.

    Args:
        src: The raster.
        band: The band, from 1.

    Returns:
        str: The name.
    """
    return src.descriptions[band - 1] or f"band_{band}"


def _scale(src: _Source, band: int) -> float:
    value = src.scales[band - 1]
    return 1.0 if value is None else float(value)


def _offset(src: _Source, band: int) -> float:
    value = src.offsets[band - 1]
    return 0.0 if value is None else float(value)


def _raster_as_band_dim(
    src: _Source, drop: set[str], mask_and_scale: bool, config_options: dict[str, str]
) -> xr.Dataset:
    """All bands as the `band` dimension of `band_data`.

    Args:
        src: The raster.
        drop: Variables to omit.
        mask_and_scale: Apply CF masking and scaling.
        config_options: Options to set around each read.

    Returns:
        xarray.Dataset: The Dataset.
    """
    bands = list(range(1, src.count + 1))
    nodatas = [src.nodatavals[b - 1] for b in bands]
    scales = [_scale(src, b) for b in bands]
    offsets = [_offset(src, b) for b in bands]
    attrs: dict[str, Any] = {"descriptions": [_description(src, b) for b in bands]}
    coords: dict[str, Any] = {"band": np.array(bands)}
    heterogeneous = []
    if all(n is None for n in nodatas):
        pass
    elif None not in nodatas and np.unique(np.array(nodatas, dtype=float)).size == 1:
        attrs["_FillValue"] = nodatas[0]
    else:
        coords["band_nodata"] = (
            "band",
            np.array([n if n is not None else np.nan for n in nodatas]),
        )
        heterogeneous.append("nodata")
    if all(s == 1.0 for s in scales):
        pass
    elif len(set(scales)) == 1:
        attrs["scale_factor"] = scales[0]
    else:
        coords["band_scale_factor"] = ("band", np.array(scales))
        heterogeneous.append("scale")
    if all(o == 0.0 for o in offsets):
        pass
    elif len(set(offsets)) == 1:
        attrs["add_offset"] = offsets[0]
    else:
        coords["band_add_offset"] = ("band", np.array(offsets))
        heterogeneous.append("offset")
    if heterogeneous:
        warnings.warn(
            f"Bands of {src.name!r} report differing "
            f"{'/'.join(heterogeneous)} values. CF decoding cannot apply "
            "per-band values to a single band_data variable, so those "
            "values are left raw, with the per-band metadata available "
            "in the band_nodata/band_scale_factor/band_add_offset "
            "coordinates. Open with band_as_dim=False to decode each "
            "band independently.",
            UserWarning,
            stacklevel=5,
        )
    if "band_data" in drop:
        return xr.Dataset()
    da = xr.DataArray(
        _lazy(src, bands, config_options),
        dims=["band", "y", "x"],
        coords=coords,
        attrs=attrs,
        name="band_data",
    )
    da.encoding["preferred_chunks"] = {"band": 1, **_preferred_chunks(src, 1)}
    ds = xr.Dataset({"band_data": da})
    return xr.decode_cf(ds, decode_times=False, mask_and_scale=mask_and_scale)


def _raster_as_vars(
    src: _Source, drop: set[str], mask_and_scale: bool, config_options: dict[str, str]
) -> xr.Dataset:
    """Each band as a `(y, x)` variable named after its description.

    Args:
        src: The raster.
        drop: Variables to omit.
        mask_and_scale: Apply CF masking and scaling.
        config_options: Options to set around each read.

    Returns:
        xarray.Dataset: The Dataset.
    """
    data_vars: dict[str, xr.DataArray] = {}
    for b in range(1, src.count + 1):
        name = _description(src, b)
        if name in drop:
            continue
        attrs: dict[str, Any] = {}
        nodata = src.nodatavals[b - 1]
        if nodata is not None:
            attrs["_FillValue"] = nodata
        scale = _scale(src, b)
        if scale != 1.0:
            attrs["scale_factor"] = scale
        offset = _offset(src, b)
        if offset != 0.0:
            attrs["add_offset"] = offset
        variable = xr.Variable(("band", "y", "x"), _lazy(src, [b], config_options))[0]
        da = xr.DataArray(variable, attrs=attrs)
        da.encoding["preferred_chunks"] = _preferred_chunks(src, b)
        data_vars[name] = da
    ds = xr.Dataset(data_vars)
    return xr.decode_cf(ds, decode_times=False, mask_and_scale=mask_and_scale)


def _maybe_chunk_dataset(
    ds: xr.Dataset,
    chunks: dict[str, int] | None,
    filename_or_obj: Any,
    config_options: dict[str, str] | None = None,
) -> xr.Dataset:
    """Chunks the Dataset with xarray's chunk manager, with a token from the path and options.

    Args:
        ds: The Dataset.
        chunks: Chunk sizes (None: no chunking; `{}`: the variables' preferred chunks).
        filename_or_obj: The path, for the token.
        config_options: The options, for the token.

    Returns:
        xarray.Dataset: The Dataset.

    Raises:
        ImportError: If dask is not installed.
    """
    if chunks is None:
        return ds
    try:
        import dask  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "opening with chunks= requires dask: pip install 'terrio[gdalxarray]' "
            "or add dask to the environment"
        ) from e
    token_src = str(filename_or_obj) + repr(sorted((config_options or {}).items()))
    token = hashlib.sha256(token_src.encode("utf-8", "replace")).hexdigest()[:16]
    if chunks == {}:
        merged: dict[str, int] = {}
        for var in ds.data_vars.values():
            for dim, size in (var.encoding.get("preferred_chunks") or {}).items():
                merged.setdefault(dim, size)
        chunks = merged
    return ds.chunk(dict(chunks), name_prefix="terrio-gdalxarray-", token=token)
