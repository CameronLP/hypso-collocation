# hypso-collocation

Colocates one HYPSO-2 capture with PACE OCI and Sentinel-3 OLCI. It finds the
PACE and Sentinel-3 granules closest in time and place to the capture, reads
all of their bands, resamples them onto the HYPSO pixel grid, and writes one
NetCDF file.

The code does not import the `hypso` or `hypso_pipeline` packages. HYPSO
products are read directly with netCDF4.

## Contents

| File | Purpose |
|---|---|
| `colocate_capture.py` | Command-line entry point. Finds or downloads the granules, resamples, and writes the output. |
| `hypso_source.py` | Reads a HYPSO L1D or L2A product (all bands) and the capture metadata. |
| `pace_source.py` | Reads a PACE OCI L2 AOP swath (Rrs, all bands) and L2 BGC swath (chlor_a). |
| `sentinel3_source.py` | Reads a Sentinel-3 OLCI WFR product directory (Rrs, all bands, chlorophyll). |
| `granules.py` | Searches and downloads PACE (NASA CMR) and Sentinel-3 (Copernicus Data Space). |
| `grid_resample.py` | Nearest-neighbour resampling. One neighbour lookup per source, shared by all bands. |
| `plot_colocated.py` | Map plots of the colocated file (band, RGB, chlorophyll), or of a native PACE or Sentinel-3 product. |
| `download_operational.py` | Downloads operational PACE L2 (AOP, BGC, IOP) and Sentinel-3 OLCI L2 (WFR, LFR) products for a capture. |

## Setup

Requires Python 3.10 or later.

```bash
conda create -n hypso-colloc python=3.12 -y
conda activate hypso-colloc
conda install -c conda-forge numpy xarray netcdf4 pandas pyresample -y
pip install earthaccess requests boto3
```

Tested with numpy 2.5, xarray 2026.7, pyresample 1.35 and earthaccess 0.18.

### Credentials

Only the downloads need credentials. Searching works without them.

**PACE (NASA Earthdata)**, used by `granules.download_pace`. Either:
- add a `~/.netrc` entry for `urs.earthdata.nasa.gov`, or
- set `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`.

**Sentinel-3 (Copernicus Data Space)**, used by `granules.download_s3`. Set
these two environment variables. The keys are the S3 access keys from your
CDSE user dashboard, not your login password.

```bash
export CDSE_ACCESS_KEY=...
export CDSE_SECRET_KEY=...
```

## Input layout

The script takes a capture directory, named after the capture. It needs:

```
<capture>/
  <capture>-meta.json                        # acquisition time, target lat/lon
  <capture>-moved_unmasked-l1d.nc            # HYPSO L1D (for --level l1d)
  <capture>-moved_unmasked-l2a-<method>.nc   # HYPSO L2A (for --level l2a)
```

The `meta.json` must contain `timestamp_acquired_string`, `target_latitude`
and `target_longitude`.

Any PACE or Sentinel-3 granules already in the capture directory are used
as they are. These are matched by name:

```
PACE_OCI.*.L2.OC_AOP*.nc     # PACE L2 AOP (required for PACE)
PACE_OCI.*.L2.OC_BGC*.nc     # PACE L2 BGC (optional, for chlor_a)
S3*_OL_2_WFR*/               # Sentinel-3 OLCI WFR product directory
```

## Usage

From the capture directory, or with a path to it:

```bash
python colocate_capture.py <capture_dir> --level l1d
python colocate_capture.py <capture_dir> --level l2a --method polymer
```

Options:

| Option | Meaning |
|---|---|
| `--level l1d` or `--level l2a` | HYPSO product level. Default `l1d`. |
| `--method` | Required with `--level l2a`: `polymer`, `dps`, `ocsmart` or `acolite_l2r`. |
| `--no-download` | Use only granules already in the capture directory. Nothing is searched for or downloaded. |

### What happens when you run it

1. Reads the HYPSO product and capture metadata.
2. For PACE, uses the local AOP and BGC granules. If the AOP granule is missing, searches NASA CMR for PACE L2 AOP and BGC granules within ±180 minutes of the capture time, centered on the capture location. It takes the closest of each and downloads it into the capture directory.
3. For Sentinel-3, uses a local `S3*_OL_2_WFR*` directory. If none exists, searches the Copernicus Data Space catalogue within the same window and downloads the closest product's files.
4. Resamples all PACE Rrs bands, the Sentinel-3 Rrs bands and both chlorophyll products onto the HYPSO grid, using a 1500 m radius for PACE and 500 m for Sentinel-3.
5. Writes the output file.

If a granule is not found, the script prints a warning and continues without it.

## Output

`<capture>/<capture>-colocated-<level>.nc`, where `<level>` is `l1d` or
`l2a-<method>`. Dimensions are `lines` × `samples` (the HYPSO grid).

| Variable | Dimensions | Contents |
|---|---|---|
| `latitude`, `longitude` | lines, samples | HYPSO pixel geolocation |
| `hypso_bands` | lines, samples, hypso_wavelength | All HYPSO bands (120 for HYPSO-2) |
| `pace_rrs` | lines, samples, pace_wavelength | PACE OCI Rrs, all bands (172) |
| `pace_chlor_a` | lines, samples | PACE chlorophyll (mg m⁻³) from L2 BGC |
| `s3_rrs` | lines, samples, s3_wavelength | Sentinel-3 OLCI Rrs, all bands present (16) |
| `s3_chl_oc4me`, `s3_chl_nn` | lines, samples | Sentinel-3 chlorophyll (mg m⁻³) |

The wavelength coordinates give each band's center wavelength in nm. Pixels
with no source within the search radius are NaN.

## Example

```bash
cd ~/HYPSO_DATA_FROHAVET_REPROCESS/image64N9E_2025-05-11T12-00-37Z
python ~/hypso-collocation/colocate_capture.py . --level l2a --method polymer --no-download
```

Output:

```
image64N9E_2025-05-11T12-00-37Z: HYPSO l2a-polymer, 120 bands, grid (598, 1092), time 2025-05-11 12:00:37+00:00
PACE AOP: 172 bands, valid on HYPSO grid 41.0%
PACE BGC chlor_a valid on HYPSO grid 40.2%
Sentinel-3: 16 bands, valid on HYPSO grid 49.3%
wrote .../image64N9E_2025-05-11T12-00-37Z-colocated-l2a-polymer.nc
```

Reading the output:

```python
import xarray as xr
ds = xr.open_dataset("image64N9E_2025-05-11T12-00-37Z-colocated-l2a-polymer.nc")
pace_550 = ds["pace_rrs"].sel(pace_wavelength=550, method="nearest")
```

## Downloading operational products

`download_operational.py` downloads the operational L2 products for a capture,
without resampling them. It searches within ±180 minutes of the capture, centred
on its location, and downloads the closest granule for each product into the
capture directory.

```bash
python download_operational.py <capture_dir>                           # PACE AOP, BGC, IOP and Sentinel-3 WFR
python download_operational.py <capture_dir> --pace BGC --s3 LFR      # chosen products
python download_operational.py <capture_dir> --pace none --s3 WFR     # Sentinel-3 only
python download_operational.py <capture_dir> --all                    # every granule in the window
```

| Source | Option | Product |
|---|---|---|
| PACE OCI | `--pace AOP` | L2 AOP: remote sensing reflectance (Rrs) |
| PACE OCI | `--pace BGC` | L2 BGC: chlorophyll and biogeochemistry |
| PACE OCI | `--pace IOP` | L2 IOP: absorption and backscatter |
| Sentinel-3 OLCI | `--s3 WFR` | Full resolution L2 Water (includes chl_oc4me and chl_nn) |
| Sentinel-3 OLCI | `--s3 LFR` | Reduced resolution L2 Water |

Downloads need credentials, as described under Setup. Searches do not. The PACE
search uses NASA CMR, and the Sentinel-3 search uses the Copernicus catalogue.

## Plotting

`plot_colocated.py` makes PNG maps. Output goes next to the input file unless
`--out` is given.

```bash
python plot_colocated.py <colocated.nc> --band 560    # HYPSO, PACE and Sentinel-3 Rrs at ~560 nm, one colour scale
python plot_colocated.py <colocated.nc> --rgb         # RGB composite per source (R 665, G 560, B 490 nm)
python plot_colocated.py <colocated.nc> --chl         # PACE BGC and Sentinel-3 chlorophyll, log colour scale
```

Single products, plotted from the original files:

```bash
python plot_colocated.py <PACE OC_AOP .nc> --native --band 560
python plot_colocated.py <PACE OC_BGC .nc> --native --chl
python plot_colocated.py <Sentinel-3 .SEN3 directory> --native --band 560
python plot_colocated.py <Sentinel-3 .SEN3 directory> --native --chl
```

The band and chlorophyll panels share a colour scale, set by the 2nd to 98th
percentile of valid pixels. RGB uses a separate 2 to 98 percent stretch for each
channel. No-data pixels are grey. Maps use plain longitude and latitude axes with
no coastlines, so no map data is downloaded.

## Notes and limitations

- The common grid is the HYPSO pixel grid. Every product is resampled onto it, and HYPSO bands are written unchanged.
- The search uses only the capture's own time and location. A granule is taken only if it lies within the time window. If nothing is in the window, nothing is used.
- Automatic download has not been tested end to end yet. Local-granule runs have been tested on `image64N9E_2025-05-11T12-00-37Z`. Search has been tested against the live PACE and Copernicus catalogues.
- Sentinel-3 quality flags (`wqsf`) are not applied. PACE `l2_flags` are not applied either.
- Output size is large: about 820 MB for a 598 × 1092 capture with all bands.
