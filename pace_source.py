"""Full-swath PACE OCI L2 AOP reading, copied from
hypso-processing-pipeline's hypso_pipeline/colocation/pace_source.py (per
user request, into this poster project's own matchup_code/ rather than
importing it via sys.path like sentinel3_colocate.py does for cdse.py/
sentinel3_source.py).

One real fix made here vs. the upstream version: the upstream
read_l2_aop_swath raises if none of _CHLOR_A_CANDIDATES ("chlor_a",
"chl_oci", "chla") are found, since chlorophyll naming was unverified
against real data there. Verified now, against a real downloaded granule
(PACE_OCI.20250514T112858.L2.OC_AOP.V3_2.nc, aeronetvenice_2025-05-14
capture): none of those names exist - this product's geophysical_data group
has no chlorophyll variable at all (only Rrs, Rrs_unc, angstrom, aot_865,
avw, band_index, l2_flags, nflh). So chlor_a is optional here (None when
absent) rather than a hard error - this poster's own PACE figure only needs
Rrs + geolocation, not chlorophyll.

PACE's native chlorophyll instead lives in a separate product, L2 BGC
(PACE_OCI_L2_BGC) - confirmed against a real downloaded granule
(PACE_OCI.20250511T112624.L2.OC_BGC.V3_2.nc, same timestamp as the AOP
granule for image64N9E_2025-05-11T12-00-37Z): geophysical_data/chlor_a,
units mg m^-3 (linear, no log10 conversion needed - unlike Sentinel-3's
chl_nn/chl_oc4me), valid range [0.001, 100]. read_l2_bgc_swath below reads
that file's own navigation_data/chlor_a.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

_CHLOR_A_CANDIDATES = ["chlor_a", "chl_oci", "chla"]


@dataclass
class PaceSwath:
    latitude: np.ndarray
    longitude: np.ndarray
    rrs: np.ndarray  # shape (lines, pixels, wavelength)
    rrs_wavelengths: np.ndarray
    chlor_a: np.ndarray | None  # shape (lines, pixels), None if not present


def read_l2_aop_swath(file_path) -> PaceSwath:
    """Read the full PACE OCI L2 AOP swath - geolocation, Rrs (all
    wavelengths), and chlor_a if present - with no spatial windowing."""
    with xr.open_dataset(file_path, group="navigation_data") as ds_nav:
        latitude = ds_nav["latitude"].values
        longitude = ds_nav["longitude"].values

    with xr.open_dataset(file_path, group="sensor_band_parameters") as ds_bands:
        rrs_wavelengths = ds_bands["wavelength_3d"].values

    with xr.open_dataset(file_path, group="geophysical_data") as ds_data:
        rrs = ds_data["Rrs"].values
        chlor_a = None
        for name in _CHLOR_A_CANDIDATES:
            if name in ds_data:
                chlor_a = ds_data[name].values
                break

    return PaceSwath(
        latitude=latitude, longitude=longitude,
        rrs=rrs, rrs_wavelengths=rrs_wavelengths, chlor_a=chlor_a,
    )


@dataclass
class PaceBgcSwath:
    latitude: np.ndarray
    longitude: np.ndarray
    chlor_a: np.ndarray  # shape (lines, pixels), mg m^-3, linear


def read_l2_bgc_swath(file_path) -> PaceBgcSwath:
    """Read the full PACE OCI L2 BGC swath - geolocation + chlor_a - with no
    spatial windowing. This is PACE's native chlorophyll product (the L2 AOP
    product read_l2_aop_swath reads has none - see this module's docstring)."""
    with xr.open_dataset(file_path, group="navigation_data") as ds_nav:
        latitude = ds_nav["latitude"].values
        longitude = ds_nav["longitude"].values

    with xr.open_dataset(file_path, group="geophysical_data") as ds_data:
        chlor_a = ds_data["chlor_a"].values

    return PaceBgcSwath(latitude=latitude, longitude=longitude, chlor_a=chlor_a)
