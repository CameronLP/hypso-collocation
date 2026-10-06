"""Granule search and download for PACE OCI and Sentinel-3 OLCI, standalone.

PACE (NASA CMR via earthaccess):
    search_pace(...)        find L2 AOP and L2 BGC granules near a point and time.
    download_pace(...)      download granules by concept id into a directory.
    Needs Earthdata credentials: ~/.netrc for urs.earthdata.nasa.gov, or
    EARTHDATA_USERNAME / EARTHDATA_PASSWORD.

Sentinel-3 OLCI WFR (Copernicus Data Space Ecosystem):
    search_s3(...)          OData catalogue search. No credentials needed.
    download_s3(...)        fetch only the files the reader needs, via CDSE's
                            S3-compatible store. Needs CDSE_ACCESS_KEY and
                            CDSE_SECRET_KEY environment variables (S3 keys from
                            the CDSE user dashboard).

nearest_within_window(...) picks the granule closest in time, but only within
a hard time window.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

PACE_PRODUCTS = {
    "L2_AOP": {"short_name": "PACE_OCI_L2_AOP", "version": "3.2"},  # remote sensing reflectance
    "L2_BGC": {"short_name": "PACE_OCI_L2_BGC", "version": "3.2"},  # chlorophyll and biogeochemistry
    "L2_IOP": {"short_name": "PACE_OCI_L2_IOP", "version": "3.2"},  # absorption and backscatter
}

# Sentinel-3 OLCI L2 product types. WFR is full resolution, LFR is reduced.
S3_PRODUCT_TYPES = ("OL_2_WFR", "OL_2_LFR")

CDSE_CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"
CDSE_S3_BUCKET = "eodata"
_S3_COLLECTION = "SENTINEL-3"

# Only these files are needed by sentinel3_source.read_wfr_swath.
# chlor_a.nc and chl_oc4me.nc are both requested: the product name varies by
# processing baseline.
_S3_REQUIRED_FILENAMES = {"geo_coordinates.nc", "chl_oc4me.nc", "chlor_a.nc", "chl_nn.nc"}
_S3_BAND_FILE_RE = re.compile(r"^Oa\d{2}_reflectance\.nc$")

_SEARCH_RETRIES = 3
_SEARCH_RETRY_DELAY_S = 2.0


@dataclass
class Granule:
    source: str  # "PACE" or "S3"
    product: str  # "L2_AOP", "L2_BGC" or "OL_2_WFR"
    granule_id: str  # earthaccess concept id (PACE) or CDSE product id (S3)
    name: str
    begin: pd.Timestamp
    s3_prefix: str | None = None  # S3 only: object-key prefix of the product


def nearest_within_window(granules: list[Granule], reference: pd.Timestamp,
                          max_minutes: float) -> Granule | None:
    """Closest granule by begin time, or None if none is within max_minutes."""
    if not granules:
        return None
    reference = pd.Timestamp(reference)
    if granules[0].begin.tzinfo is not None and reference.tzinfo is None:
        reference = reference.tz_localize(granules[0].begin.tzinfo)
    elif granules[0].begin.tzinfo is None and reference.tzinfo is not None:
        reference = reference.tz_localize(None)
    closest = min(granules, key=lambda g: abs(g.begin - reference))
    if abs(closest.begin - reference) > pd.Timedelta(minutes=max_minutes):
        return None
    return closest


# ---------------------------------------------------------------------------
# PACE
# ---------------------------------------------------------------------------

def search_pace(center: pd.Timestamp, window_minutes: float,
                latitude: float, longitude: float, products=None) -> list[Granule]:
    """Search NASA CMR for PACE L2 granules covering a point within
    +/- window_minutes of center. products: keys of PACE_PRODUCTS (default all)."""
    import earthaccess

    center = pd.Timestamp(center)
    window = pd.Timedelta(minutes=window_minutes)
    temporal = ((center - window).strftime("%Y-%m-%dT%H:%M:%S"),
                (center + window).strftime("%Y-%m-%dT%H:%M:%S"))

    granules = []
    for product, spec in PACE_PRODUCTS.items():
        if products is not None and product not in products:
            continue
        results = earthaccess.search_data(
            point=(longitude, latitude), temporal=temporal,
            short_name=spec["short_name"], version=spec["version"],
        )
        for r in results:
            granules.append(Granule(
                source="PACE", product=product,
                granule_id=r["meta"]["concept-id"],
                name=r["umm"].get("GranuleUR", r["meta"]["concept-id"]),
                begin=pd.to_datetime(r["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]),
            ))
    return granules


def download_pace(granules: list[Granule], dest_dir: str | Path) -> dict[str, Path]:
    """Download PACE granules into dest_dir. Returns granule_id -> local file.
    Files already on disk are reused (earthaccess's default)."""
    import earthaccess
    import xarray as xr

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted({g.granule_id for g in granules})
    if not ids:
        return {}

    results = earthaccess.search_data(concept_id=ids)
    if not results:
        print(f"[warning] no PACE granule resolved for ids {ids}")
        return {}
    files = earthaccess.download(results, local_path=str(dest_dir), show_progress=True)

    out = {}
    for result, file in zip(results, files):
        try:
            with xr.open_dataset(file):
                pass
        except Exception as ex:
            print(f"[warning] {file} did not open ({ex}); skipping")
            continue
        out[result["meta"]["concept-id"]] = Path(file)
    return out


# ---------------------------------------------------------------------------
# Sentinel-3 OLCI WFR (CDSE)
# ---------------------------------------------------------------------------

def search_s3(center: pd.Timestamp, window_minutes: float,
              latitude: float, longitude: float, product: str = "OL_2_WFR") -> list[Granule]:
    """Search the CDSE catalogue for Sentinel-3 OLCI products (product: one of
    S3_PRODUCT_TYPES) covering a point within +/- window_minutes of center.
    No credentials needed."""
    center = pd.Timestamp(center)
    window = pd.Timedelta(minutes=window_minutes)
    start = (center - window).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end = (center + window).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    filter_str = (
        f"Collection/Name eq '{_S3_COLLECTION}' and "
        f"contains(Name,'{product}') and "
        f"ContentDate/Start gt {start} and ContentDate/Start lt {end} and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;POINT({longitude} {latitude})')"
    )

    last_error = None
    for attempt in range(1, _SEARCH_RETRIES + 1):
        try:
            response = requests.get(CDSE_CATALOGUE_URL, params={"$filter": filter_str, "$top": 20},
                                    timeout=60)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as ex:
            last_error = ex
            if attempt < _SEARCH_RETRIES:
                time.sleep(_SEARCH_RETRY_DELAY_S)
    else:
        raise last_error

    granules = []
    for item in response.json().get("value", []):
        granules.append(Granule(
            source="S3", product=product,
            granule_id=item["Id"], name=item["Name"],
            begin=pd.to_datetime(item["ContentDate"]["Start"]),
            s3_prefix=item["S3Path"].removeprefix("/eodata/").rstrip("/"),
        ))
    return granules


def download_s3(granule: Granule, dest_dir: str | Path) -> Path:
    """Download the files sentinel3_source.read_wfr_swath needs for one S3
    granule into dest_dir/<product name>/. Returns that product directory.
    Requires CDSE_ACCESS_KEY and CDSE_SECRET_KEY in the environment."""
    import boto3

    access_key = os.environ.get("CDSE_ACCESS_KEY")
    secret_key = os.environ.get("CDSE_SECRET_KEY")
    if not access_key or not secret_key:
        raise RuntimeError("Set CDSE_ACCESS_KEY and CDSE_SECRET_KEY to download Sentinel-3 data.")
    if granule.s3_prefix is None:
        raise ValueError(f"{granule.name} has no S3 prefix; cannot download.")

    s3 = boto3.client("s3", endpoint_url=CDSE_S3_ENDPOINT,
                      aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    product_dir = Path(dest_dir) / granule.name
    product_dir.mkdir(parents=True, exist_ok=True)

    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=CDSE_S3_BUCKET, Prefix=granule.s3_prefix + "/"):
        for obj in page.get("Contents", []):
            filename = obj["Key"].rsplit("/", 1)[-1]
            if filename in _S3_REQUIRED_FILENAMES or _S3_BAND_FILE_RE.match(filename):
                keys.append(obj["Key"])
    if not keys:
        raise FileNotFoundError(f"No matching files under s3://{CDSE_S3_BUCKET}/{granule.s3_prefix}/")

    for key in keys:
        target = product_dir / key.rsplit("/", 1)[-1]
        if not target.is_file():
            s3.download_file(CDSE_S3_BUCKET, key, str(target))
    return product_dir
