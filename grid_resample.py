"""Nearest-neighbour swath-to-swath resampling onto a target grid, for all bands
at once.

The nearest-neighbour lookup (which target pixel takes which source pixel)
depends only on the two geometries, so it is computed once per source and
applied to every band. Doing the lookup per band costs roughly the number of
bands times more time for the same output.
"""
from __future__ import annotations

import numpy as np
from pyresample.geometry import SwathDefinition
from pyresample.kd_tree import get_neighbour_info, get_sample_from_neighbour_info


def resample_nearest(source_lons, source_lats, data, target_lons, target_lats,
                     radius_of_influence_m: float):
    """Resample `data` from a source swath onto the target grid.

    data : 2D array (lines, pixels) or 3D array (lines, pixels, bands). Each
        band is resampled with the same lookup. NaN marks no-data.
    radius_of_influence_m : a target pixel takes a source pixel only if the
        source is within this distance (m). Otherwise the output is NaN.

    Returns an array shaped (target lines, target pixels) or
    (target lines, target pixels, bands).
    """
    source = SwathDefinition(lons=np.asarray(source_lons), lats=np.asarray(source_lats))
    target = SwathDefinition(lons=np.asarray(target_lons), lats=np.asarray(target_lats))
    valid_in, valid_out, index_array, _ = get_neighbour_info(
        source, target, radius_of_influence_m, neighbours=1)

    data = np.asarray(data)
    if data.ndim == 2:
        return get_sample_from_neighbour_info(
            "nn", target.shape, data, valid_in, valid_out, index_array, fill_value=np.nan)
    if data.ndim == 3:
        return np.stack(
            [get_sample_from_neighbour_info("nn", target.shape, data[:, :, b],
                                            valid_in, valid_out, index_array, fill_value=np.nan)
             for b in range(data.shape[2])],
            axis=-1)
    raise ValueError(f"expected a 2D or 3D array, got shape {data.shape}")
