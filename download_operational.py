"""Download operational PACE OCI and Sentinel-3 OLCI L2 products for one HYPSO capture.

Usage:
    python download_operational.py <capture_dir> [--pace AOP,BGC,IOP] [--s3 WFR,LFR]
                                   [--window 180] [--all]

Searches around the capture's own time and location (from <name>-meta.json),
then downloads into <capture_dir>.

    --pace   PACE OCI L2 products: AOP (Rrs), BGC (chlorophyll and biogeochemistry),
             IOP (absorption and backscatter). Default AOP,BGC,IOP. Use "none" to skip.
    --s3     Sentinel-3 OLCI L2 products: WFR (full resolution), LFR (reduced
             resolution). Default WFR. Use "none" to skip.
    --window Search half-width in minutes around the capture time. Default 180.
    --all    Download every granule found, not only the one closest in time.

PACE needs NASA Earthdata credentials (~/.netrc or EARTHDATA_USERNAME /
EARTHDATA_PASSWORD). Sentinel-3 needs CDSE_ACCESS_KEY and CDSE_SECRET_KEY.
Searches need neither.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import granules
import hypso_source

PACE_CHOICES = {"AOP": "L2_AOP", "BGC": "L2_BGC", "IOP": "L2_IOP"}
S3_CHOICES = {"WFR": "OL_2_WFR", "LFR": "OL_2_LFR"}


def _parse(value: str, choices: dict[str, str], default: str) -> list[str]:
    if value.lower() == "none":
        return []
    keys = [v.strip().upper() for v in value.split(",") if v.strip()]
    unknown = [k for k in keys if k not in choices]
    if unknown:
        raise SystemExit(f"unknown product(s) {unknown}; choose from {sorted(choices)}")
    return keys or default.split(",")


def _pick(found: list[granules.Granule], reference, window: float, download_all: bool):
    if download_all:
        return found
    best = granules.nearest_within_window(found, reference, window)
    return [best] if best is not None else []


def run(capture_dir: Path, pace: list[str], s3: list[str], window: float, download_all: bool) -> None:
    capture_dir = Path(capture_dir).resolve()
    time, lat, lon = hypso_source.read_meta(capture_dir)
    print(f"{capture_dir.name}: searching +/-{window:.0f} min of {time} at ({lat}, {lon})")

    if pace:
        products = [PACE_CHOICES[k] for k in pace]
        found = granules.search_pace(time, window, lat, lon, products=products)
        chosen = []
        for product in products:
            chosen += _pick([g for g in found if g.product == product], time, window, download_all)
        for g in chosen:
            print(f"  PACE {g.product}: {g.name}  ({g.begin})")
        if chosen:
            files = granules.download_pace(chosen, capture_dir)
            print(f"  downloaded {len(files)} PACE granule(s) to {capture_dir}")
        else:
            print("  no PACE granules in the window")

    for key in s3:
        product = S3_CHOICES[key]
        found = granules.search_s3(time, window, lat, lon, product=product)
        chosen = _pick(found, time, window, download_all)
        if not chosen:
            print(f"  Sentinel-3 {product}: no granule in the window")
            continue
        for g in chosen:
            print(f"  Sentinel-3 {product}: {g.name}  ({g.begin})")
            path = granules.download_s3(g, capture_dir)
            print(f"    -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--pace", default="AOP,BGC,IOP", help="PACE products: AOP,BGC,IOP or none")
    parser.add_argument("--s3", default="WFR", help="Sentinel-3 products: WFR,LFR or none")
    parser.add_argument("--window", type=float, default=180.0, help="search half-width in minutes")
    parser.add_argument("--all", action="store_true", help="download every granule in the window")
    args = parser.parse_args()
    run(args.capture_dir,
        pace=_parse(args.pace, PACE_CHOICES, "AOP,BGC,IOP"),
        s3=_parse(args.s3, S3_CHOICES, "WFR"),
        window=args.window, download_all=args.all)
