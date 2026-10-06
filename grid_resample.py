"""Resampling onto a target grid, for all bands at once.

Two methods, both computing the source-to-target lookup once per source and
reusing it for every band:

    nearest   each target pixel takes the nearest source pixel within a radius
    bilinear  each target pixel is interpolated from the surrounding valid source
              pixels, within a radius (missing neighbours are skipped. Where no valid
              neighbour exists, the nearest-neighbour value is used). On a regular
              region grid this is true bilinear interpolation. On the HYPSO swath it is an
              inverse-distance mean of the 4 nearest source pixels, since
              bilinear needs a regular grid.

Targets are either the HYPSO pixel grid (a SwathDefinition) or a regular grid
bounded by a region (an AreaDefinition, built by region_grid).
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from pyproj import Proj
from pyresample import geometry
from pyresample.bilinear.xarr import XArrayBilinearResampler
from pyresample.geometry import AreaDefinition, SwathDefinition
from pyresample.kd_tree import get_neighbour_info, get_sample_from_neighbour_info

METHODS = ("nearest", "bilinear")

# Bilinear output is NaN where the valid share of the bilinear weight is below this.
BILINEAR_MIN_VALID_WEIGHT = 0.5


def region_grid(bbox: tuple[float, float, float, float], res_m: float = 300.0,
                area_id: str = "region") -> AreaDefinition:
    """Regular stereographic grid covering bbox = (lon_min, lat_min, lon_max, lat_max),
    with square pixels of res_m metres."""
    lon_min, lat_min, lon_max, lat_max = bbox
    lon_0 = (lon_max - lon_min) / 2 + lon_min
    lat_ts = (lat_max - lat_min) / 2 + lat_min
    proj = Proj(proj="stere", ellps="bessel", lat_0=90.0, lon_0=lon_0, lat_ts=lat_ts)
    projection = {"proj": "stere", "ellps": "bessel", "lat_0": 90.0,
                  "lon_0": lon_0, "lat_ts": lat_ts, "units": "m"}
    x0, y0 = proj(lon_min, lat_min)
    x1, y1 = proj(lon_max, lat_max)
    extent = (x0, y0, x1, y1)
    width = int(round((x1 - x0) / res_m))
    height = int(round((y1 - y0) / res_m))
    return geometry.AreaDefinition(area_id, area_id, area_id, projection, width, height, extent)


def _as_source(lons, lats) -> SwathDefinition:
    return SwathDefinition(lons=np.asarray(lons), lats=np.asarray(lats))


def resample(source_lons, source_lats, data, target, radius_m: float, method: str = "nearest"):
    """Resample data from a source swath onto target (SwathDefinition or AreaDefinition).

    data : 2D (lines, pixels) or 3D (lines, pixels, bands). NaN is no-data.
    radius_m : maximum distance from target pixel to source pixel. Outside the
        radius the output is NaN.

    Returns an array with the target's shape (lines, pixels) or
    (lines, pixels, bands).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    source = _as_source(source_lons, source_lats)
    data = np.asarray(data)
    if data.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D array, got shape {data.shape}")
    shape = target.shape

    if method == "nearest":
        valid_in, valid_out, index_array, _ = get_neighbour_info(source, target, radius_m, neighbours=1)
        sample = lambda band: get_sample_from_neighbour_info(
            "nn", shape, band, valid_in, valid_out, index_array, fill_value=np.nan)
    elif isinstance(target, AreaDefinition):
        brs = XArrayBilinearResampler(source, target, radius_of_influence=radius_m)
        brs.get_bil_info()
        def _interp(values):
            return brs.get_sample_from_bil_info(
                data=xr.DataArray(values, dims=("y", "x")), fill_value=np.nan, output_shape=shape).values

        # Nearest-neighbour lookup, used only to fill target pixels where the bilinear
        # neighbourhood has no usable source pixels (holes in a regular pattern on
        # swaths with many missing pixels).
        nn_in, nn_out, nn_index, _ = get_neighbour_info(source, target, radius_m, neighbours=1)

        def sample(band):
            # Bilinear weights of the valid neighbours only: interpolate the data
            # (NaN -> 0) and the validity mask with the same weights, then divide.
            # Without this, one missing neighbour blanks the whole output pixel.
            finite = np.isfinite(band)
            num = _interp(np.where(finite, band, 0.0))
            weight = _interp(finite.astype(np.float64))
            with np.errstate(invalid="ignore", divide="ignore"):
                out = num / weight
            out[~(weight >= BILINEAR_MIN_VALID_WEIGHT)] = np.nan
            filled = get_sample_from_neighbour_info(
                "nn", shape, band, nn_in, nn_out, nn_index, fill_value=np.nan)
            return np.where(np.isnan(out), filled, out)
    else:
        # pyresample's bilinear needs a regular target grid, so on a swath target
        # use the 4 nearest source pixels, weighted by 1/distance.
        valid_in, valid_out, index_array, dist = get_neighbour_info(source, target, radius_m, neighbours=4)
        weights = 1.0 / np.maximum(dist, 1e-6)

        def sample(band):
            values = band.ravel()[valid_in][index_array]
            w = np.where(np.isfinite(values), weights, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.nansum(np.where(w > 0, values, 0.0) * w, axis=1) / np.sum(w, axis=1)
            out = np.full(shape[0] * shape[1], np.nan)
            out[valid_out] = mean
            return out.reshape(shape)

    if data.ndim == 2:
        return sample(data)
    return np.stack([sample(data[:, :, b]) for b in range(data.shape[2])], axis=-1)
