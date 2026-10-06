"""Full-swath Sentinel-3 OLCI L2 WFR reading for matchup_code.

Mirrors pace_source.py's shape (full swath, not a windowed box - see that
module's docstring for why) but for CDSE's Sentinel-3 product instead of
PACE. See granules.py for search/download (the CDSE API side);
this module only reads an already-downloaded product directory.

File layout and semantics below are LIVE-VERIFIED against a real downloaded
WFR product (S3B_OL_2_WFR____20260615T092031_..., 2026-06-15 baseline) via
its own netCDF variable attributes - not taken from documentation, which
turned out to disagree with the real product on two points:

- Oa##_reflectance.nc's own attrs say `long_name: "BRDF-corrected water
  remote-sensing reflectance..."`, `units: "sr^-1"` - this IS ALREADY Rrs,
  not water-leaving reflectance rho_w. No /pi conversion (unlike this repo's
  HYPSO DPS/Polymer convention - see extraction.py's divide_by_pi - which
  does not apply here).
- The OC4ME chlorophyll file was named chlor_a.nc (variable `chlor_a`), NOT
  chl_oc4me.nc as some documentation examples show - _CHL_OC4ME_FILENAMES
  tries both, since an older/different processing baseline might still use
  the other name.

Both chlorophyll variables (chlor_a.nc's chlor_a, chl_nn.nc's CHL_NN) are
log10-scaled (`units: "lg(re mg.m-3)"`, confirmed via real data: NaN-masked
finite range roughly [-2, 3]) - read_wfr_swath converts to linear mg/m^3
(10**value) before returning, matching this repo's existing convention of
exposing chlorophyll in linear scale (see config.ProcessingConfig.polymer_version's
docstring on Polymer v2's log-scale output needing the same reconstruction).

The full search (granules.py) -> download (granules.py) -> read_wfr_swath ->
grid_resample.resample_nearest chain has been run end to end against the
real product referenced above with a synthetic target grid; output shapes,
physical value ranges (Rrs ~[-0.1, 0.35] sr^-1, chlorophyll ~[0.01, 1000]
mg/m^3), and NaN propagation (this granule's footprint spans mostly land/
cloud - only ~7.5% of pixels have valid retrievals at all, which the
resampled output correctly reflects) all check out. Not yet exercised:
quality-flag masking (wqsf.nc is not read/applied here, unlike
extraction.py's PACE l2_flags handling for the AERONET-OC matchup stages -
a deliberate scope cut for the first version, not an oversight) and a
second real product to rule out this being the only baseline/variant CDSE
serves.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

_CHL_OC4ME_FILENAMES = ["chlor_a.nc", "chl_oc4me.nc"]
_CHL_OC4ME_CANDIDATES = ["chlor_a", "CHL_OC4ME", "chl_oc4me"]
_CHL_NN_CANDIDATES = ["CHL_NN", "chl_nn"]

# Official OLCI band center wavelengths (nm) - ESA-published instrument
# characteristic, not derived from any one product file. Used only to label
# whichever Oa##_reflectance.nc bands are actually found on disk (see
# read_wfr_swath) - a band missing from this table is skipped with a
# warning printed rather than silently mislabeled. The set of bands actually
# present (1-12, 16-18, 21 - gas-absorption bands 13-15/19-20 excluded) was
# confirmed against the real downloaded product referenced above.
OLCI_BAND_WAVELENGTHS_NM = {
    1: 400.0, 2: 412.5, 3: 442.5, 4: 490.0, 5: 510.0, 6: 560.0, 7: 620.0,
    8: 665.0, 9: 673.75, 10: 681.25, 11: 708.75, 12: 753.75, 16: 778.75,
    17: 865.0, 18: 885.0, 21: 1020.0,
}

_BAND_FILENAME_RE = re.compile(r"^Oa(\d{2})_reflectance\.nc$")


@dataclass
class Sentinel3Swath:
    latitude: np.ndarray
    longitude: np.ndarray
    rrs: np.ndarray  # shape (lines, pixels, wavelength)
    rrs_wavelengths: np.ndarray
    chl_oc4me: np.ndarray  # shape (lines, pixels), linear mg/m^3
    chl_nn: np.ndarray  # shape (lines, pixels), linear mg/m^3


def _find_variable(ds: xr.Dataset, candidates: list[str], file_path) -> np.ndarray:
    for name in candidates:
        if name in ds:
            return ds[name].values
    raise KeyError(f"None of {candidates} found in {file_path} (available: {sorted(ds.data_vars)}) "
                    f"- update the candidate list in sentinel3_source.py.")


def _find_chl_file(sen3_dir: Path, filenames: list[str]) -> Path:
    for name in filenames:
        path = sen3_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"None of {filenames} found in {sen3_dir}")


def read_wfr_swath(sen3_dir) -> Sentinel3Swath:
    """Read the full Sentinel-3 OLCI WFR swath - geolocation, per-band Rrs,
    and both chlorophyll products (converted to linear mg/m^3) - from an
    already-downloaded product directory (see
    granules.download_s3)."""
    sen3_dir = Path(sen3_dir)

    with xr.open_dataset(sen3_dir / "geo_coordinates.nc") as ds_geo:
        latitude = ds_geo["latitude"].values
        longitude = ds_geo["longitude"].values

    band_files = sorted(sen3_dir.glob("Oa*_reflectance.nc"))
    if not band_files:
        raise FileNotFoundError(f"No Oa##_reflectance.nc files found in {sen3_dir}")

    rrs_bands = []
    wavelengths = []
    for band_file in band_files:
        match = _BAND_FILENAME_RE.match(band_file.name)
        if not match:
            continue
        band_num = int(match.group(1))
        if band_num not in OLCI_BAND_WAVELENGTHS_NM:
            print(f"[WARNING] {band_file.name}: no known wavelength for OLCI band {band_num} - skipping.")
            continue

        var_name = f"Oa{band_num:02d}_reflectance"
        with xr.open_dataset(band_file) as ds_band:
            # Already Rrs (sr^-1) - see this module's docstring, confirmed
            # via the file's own units/long_name attributes. No /pi here.
            rrs_bands.append(_find_variable(ds_band, [var_name], band_file))
        wavelengths.append(OLCI_BAND_WAVELENGTHS_NM[band_num])

    order = np.argsort(wavelengths)
    rrs = np.stack([rrs_bands[i] for i in order], axis=-1)
    rrs_wavelengths = np.array(wavelengths)[order]

    chl_oc4me_file = _find_chl_file(sen3_dir, _CHL_OC4ME_FILENAMES)
    with xr.open_dataset(chl_oc4me_file) as ds_chl:
        chl_oc4me = 10.0 ** _find_variable(ds_chl, _CHL_OC4ME_CANDIDATES, chl_oc4me_file)
    with xr.open_dataset(sen3_dir / "chl_nn.nc") as ds_chl:
        chl_nn = 10.0 ** _find_variable(ds_chl, _CHL_NN_CANDIDATES, sen3_dir / "chl_nn.nc")

    return Sentinel3Swath(
        latitude=latitude, longitude=longitude,
        rrs=rrs, rrs_wavelengths=rrs_wavelengths,
        chl_oc4me=chl_oc4me, chl_nn=chl_nn,
    )
