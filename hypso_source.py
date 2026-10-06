"""Read a HYPSO L1D or L2A product (all bands) and the capture metadata, standalone.

Reads the netCDF directly, so the hypso package is not needed.

Level selects the file and the band variable prefix:
    l1d                      <name>-moved_unmasked-l1d.nc,  rhot_<nm>
    l2a, method=polymer      <name>-moved_unmasked-l2a-polymer.nc,  rho_w_<nm>
    l2a, method=dps          <name>-moved_unmasked-l2a-dps.nc,  Rrs_<nm>
    l2a, method=ocsmart      <name>-moved_unmasked-l2a-ocsmart.nc,  Rrs_<nm>
    l2a, method=acolite_l2r  <name>-moved_unmasked-l2a-acolite_l2r.nc,  rhos_<nm>
Only variables named <prefix><digits> are bands (chla, chlor_a_* are skipped).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd


@dataclass
class HypsoCapture:
    name: str
    time: pd.Timestamp  # UTC acquisition time
    center_lat: float
    center_lon: float
    latitude: np.ndarray  # (lines, samples)
    longitude: np.ndarray  # (lines, samples)
    rhot: np.ndarray  # (lines, samples, bands), all HYPSO bands, sorted by wavelength
    wavelengths: np.ndarray  # nm, length = bands


def read_meta(capture_dir: Path) -> tuple[pd.Timestamp, float, float]:
    name = capture_dir.name
    meta = json.loads((capture_dir / f"{name}-meta.json").read_text())
    return (pd.Timestamp(meta["timestamp_acquired_string"]),
            float(meta["target_latitude"]), float(meta["target_longitude"]))


L2A_PREFIX = {
    "polymer": "rho_w_",
    "dps": "Rrs_",
    "ocsmart": "Rrs_",
    "acolite_l2r": "rhos_",
}


def find_product(capture_dir: Path, level: str = "l1d", method: str | None = None) -> tuple[Path, str]:
    """Return (path, band prefix) for the requested HYPSO level."""
    name = capture_dir.name
    if level == "l1d":
        path, prefix = capture_dir / f"{name}-moved_unmasked-l1d.nc", "rhot_"
    elif level == "l2a":
        if method not in L2A_PREFIX:
            raise ValueError(f"L2A method must be one of {sorted(L2A_PREFIX)}, got {method!r}")
        path = capture_dir / f"{name}-moved_unmasked-l2a-{method}.nc"
        prefix = L2A_PREFIX[method]
    else:
        raise ValueError(f"level must be 'l1d' or 'l2a', got {level!r}")
    if not path.is_file():
        raise FileNotFoundError(f"No {level} product at {path}")
    return path, prefix


def read_hypso(capture_dir: Path, level: str = "l1d", method: str | None = None) -> HypsoCapture:
    """Read all HYPSO bands for the given level (see module docstring) and the
    geometry group's lat/lon."""
    capture_dir = Path(capture_dir).resolve()
    time, center_lat, center_lon = read_meta(capture_dir)
    path, prefix = find_product(capture_dir, level, method)

    with netCDF4.Dataset(path) as ds:
        geometry = ds.groups["geometry"]
        latitude = np.asarray(geometry["latitude"][:])
        longitude = np.asarray(geometry["longitude"][:])
        products = ds.groups["products"]
        names = [v for v in products.variables
                 if v.startswith(prefix) and v[len(prefix):].isdigit()]
        wavelengths = np.array([float(v[len(prefix):]) for v in names])
        order = np.argsort(wavelengths)
        wavelengths = wavelengths[order]
        bands = np.stack([np.asarray(products[names[i]][:], dtype=np.float32) for i in order], axis=-1)

    return HypsoCapture(
        name=capture_dir.name, time=time, center_lat=center_lat, center_lon=center_lon,
        latitude=latitude, longitude=longitude, rhot=bands, wavelengths=wavelengths,
    )
