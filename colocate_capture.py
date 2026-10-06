"""Colocate one HYPSO capture with PACE OCI and Sentinel-3 OLCI, onto the HYPSO grid.

Usage:
    python colocate_capture.py <capture_dir> [--level l1d|l2a] [--method polymer|dps|ocsmart|acolite_l2r] [--no-download]

Inputs (found in capture_dir, or searched and downloaded for the capture's own
location and time when missing):
    HYPSO    <name>-moved_unmasked-l1d.nc (--level l1d) or
             <name>-moved_unmasked-l2a-<method>.nc (--level l2a), plus
             <name>-meta.json (no download)
    PACE     PACE_OCI.*.L2.OC_AOP*.nc (Rrs, all bands)
             PACE_OCI.*.L2.OC_BGC*.nc (chlor_a, optional)
    S3       S3*_OL_2_WFR*/ (Rrs, all OLCI bands; chlorophyll optional)

Output: <capture_dir>/<name>-colocated-<l1d|l2a-method>.nc, on the HYPSO pixel grid. It holds
all HYPSO bands, all PACE Rrs bands, and all Sentinel-3 Rrs bands, each with
its own wavelength axis. Nearest-neighbour resampling, one lookup per source.

Credentials: NASA Earthdata via ~/.netrc or EARTHDATA_USERNAME/PASSWORD for
PACE; CDSE_ACCESS_KEY and CDSE_SECRET_KEY for the Sentinel-3 download (the
search needs none).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import granules
import grid_resample
from pyresample import geometry
import hypso_source
import pace_source
import sentinel3_source

PACE_RADIUS_M = 1500.0
S3_RADIUS_M = 500.0
TIME_WINDOW_MIN = 180.0


def _local(capture_dir: Path, pattern: str) -> list[Path]:
    return sorted(capture_dir.glob(pattern))


def _ensure_pace(capture_dir: Path, capture, download: bool):
    aop = _local(capture_dir, "PACE_OCI.*.L2.OC_AOP*.nc")
    bgc = _local(capture_dir, "PACE_OCI.*.L2.OC_BGC*.nc")
    if (aop or not download):
        return aop[0] if aop else None, bgc[0] if bgc else None

    found = granules.search_pace(capture.time, TIME_WINDOW_MIN, capture.center_lat, capture.center_lon)
    chosen = []
    for product in ("L2_AOP", "L2_BGC"):
        best = granules.nearest_within_window(
            [g for g in found if g.product == product], capture.time, TIME_WINDOW_MIN)
        if best is None:
            print(f"[warning] no PACE {product} granule within {TIME_WINDOW_MIN:.0f} min")
        chosen.append(best)
    wanted = [g for g in chosen if g is not None]
    files = granules.download_pace(wanted, capture_dir)
    aop_path = next((files[g.granule_id] for g in wanted if g.product == "L2_AOP" and g.granule_id in files), None)
    bgc_path = next((files[g.granule_id] for g in wanted if g.product == "L2_BGC" and g.granule_id in files), None)
    return aop_path, bgc_path


def _ensure_s3(capture_dir: Path, capture, download: bool):
    local = [p for p in _local(capture_dir, "S3*_OL_2_WFR*") if p.is_dir()]
    if local or not download:
        return local[0] if local else None

    found = granules.search_s3(capture.time, TIME_WINDOW_MIN, capture.center_lat, capture.center_lon)
    best = granules.nearest_within_window(found, capture.time, TIME_WINDOW_MIN)
    if best is None:
        print(f"[warning] no Sentinel-3 OLCI WFR granule within {TIME_WINDOW_MIN:.0f} min")
        return None
    print(f"downloading Sentinel-3 {best.name}")
    return granules.download_s3(best, capture_dir)


def _valid_fraction(array: np.ndarray) -> float:
    return float(np.isfinite(array).any(axis=-1).mean()) if array.ndim == 3 else float(np.isfinite(array).mean())


HYPSO_RADIUS_M = 300.0


def run(capture_dir: Path, download: bool = True, level: str = "l1d", method: str | None = None,
        grid: str = "hypso", bbox=None, res_m: float = 300.0, resample_method: str = "nearest") -> Path:
    capture_dir = Path(capture_dir).resolve()
    capture = hypso_source.read_hypso(capture_dir, level=level, method=method)
    product = f"{level}" if level == "l1d" else f"{level}-{method}"
    print(f"{capture.name}: HYPSO {product}, {capture.rhot.shape[-1]} bands, "
          f"grid {capture.latitude.shape}, time {capture.time}")

    if grid == "hypso":
        target = geometry.SwathDefinition(lons=capture.longitude, lats=capture.latitude)
        lats, lons = capture.latitude, capture.longitude
        grid_tag = "hypso"
    elif grid == "region":
        if bbox is None:
            raise ValueError("grid=region needs a bbox (lon_min, lat_min, lon_max, lat_max)")
        target = grid_resample.region_grid(tuple(bbox), res_m)
        lons, lats = target.get_lonlats()
        grid_tag = f"region{res_m:g}m"
    else:
        raise ValueError(f"grid must be 'hypso' or 'region', got {grid!r}")
    print(f"target grid: {grid} {target.shape}, resampling: {resample_method}")

    def resample(src_lon, src_lat, data, radius):
        # On a region grid a target pixel is res_m wide, so the search radius must cover at least one pixel.
        if grid == "region":
            radius = max(radius, res_m)
        return grid_resample.resample(src_lon, src_lat, data, target, radius, resample_method)

    out: dict[str, np.ndarray] = {}
    coords: dict[str, np.ndarray] = {}

    # HYPSO: on its own pixel grid it is written unchanged; otherwise it is resampled too.
    if grid == "hypso":
        out["hypso_bands"] = capture.rhot
    else:
        out["hypso_bands"] = resample(capture.longitude, capture.latitude, capture.rhot, HYPSO_RADIUS_M)
    coords["hypso_wavelength"] = capture.wavelengths

    # PACE
    aop_path, bgc_path = _ensure_pace(capture_dir, capture, download)
    if aop_path is not None:
        swath = pace_source.read_l2_aop_swath(aop_path)
        out["pace_rrs"] = resample(swath.longitude, swath.latitude, swath.rrs, PACE_RADIUS_M)
        coords["pace_wavelength"] = np.asarray(swath.rrs_wavelengths)
        print(f"PACE AOP: {swath.rrs.shape[-1]} bands, valid on HYPSO grid "
              f"{_valid_fraction(out['pace_rrs']):.1%}")
    else:
        print("PACE AOP: not available for this capture")
    if bgc_path is not None:
        bgc = pace_source.read_l2_bgc_swath(bgc_path)
        out["pace_chlor_a"] = resample(bgc.longitude, bgc.latitude, bgc.chlor_a, PACE_RADIUS_M)
        print(f"PACE BGC chlor_a valid on HYPSO grid {_valid_fraction(out['pace_chlor_a']):.1%}")

    # Sentinel-3
    s3_dir = _ensure_s3(capture_dir, capture, download)
    if s3_dir is not None:
        s3 = sentinel3_source.read_wfr_swath(s3_dir)
        out["s3_rrs"] = resample(s3.longitude, s3.latitude, s3.rrs, S3_RADIUS_M)
        coords["s3_wavelength"] = np.asarray(s3.rrs_wavelengths)
        out["s3_chl_oc4me"] = resample(s3.longitude, s3.latitude, s3.chl_oc4me, S3_RADIUS_M)
        out["s3_chl_nn"] = resample(s3.longitude, s3.latitude, s3.chl_nn, S3_RADIUS_M)
        print(f"Sentinel-3: {s3.rrs.shape[-1]} bands, valid on HYPSO grid "
              f"{_valid_fraction(out['s3_rrs']):.1%}")
    else:
        print("Sentinel-3: not available for this capture")

    dims = {
        "hypso_bands": ("lines", "samples", "hypso_wavelength"),
        "pace_rrs": ("lines", "samples", "pace_wavelength"),
        "s3_rrs": ("lines", "samples", "s3_wavelength"),
    }
    data_vars = {}
    for name, array in out.items():
        if array.ndim == 3:
            data_vars[name] = (dims.get(name, ("lines", "samples", f"{name}_band")), array)
        else:
            data_vars[name] = (("lines", "samples"), array)
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "latitude": (("lines", "samples"), lats),
            "longitude": (("lines", "samples"), lons),
            **{k: (k, v) for k, v in coords.items()},
        },
        attrs={"hypso_capture": capture.name, "hypso_product": product, "time": str(capture.time),
               "grid": grid_tag, "resampling": resample_method},
    )
    output = capture_dir / f"{capture.name}-colocated-{product}-{grid_tag}-{resample_method}.nc"
    ds.to_netcdf(output)
    print(f"wrote {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--level", choices=["l1d", "l2a"], default="l1d", help="HYPSO product level")
    parser.add_argument("--method", choices=["polymer", "dps", "ocsmart", "acolite_l2r"],
                        help="L2A method (required with --level l2a)")
    parser.add_argument("--grid", choices=["hypso", "region"], default="hypso",
                        help="target grid: the HYPSO pixel grid, or a regular grid bounded by --bbox")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                        help="region for --grid region")
    parser.add_argument("--res-m", type=float, default=300.0, help="pixel size in metres for --grid region")
    parser.add_argument("--resampling", choices=["nearest", "bilinear"], default="nearest")
    parser.add_argument("--no-download", action="store_true", help="use local granules only")
    args = parser.parse_args()
    if args.grid == "region" and args.bbox is None:
        parser.error("--grid region needs --bbox")
    if args.level == "l2a" and args.method is None:
        parser.error("--level l2a needs --method")
    run(args.capture_dir, download=not args.no_download, level=args.level, method=args.method,
        grid=args.grid, bbox=args.bbox, res_m=args.res_m, resample_method=args.resampling)
