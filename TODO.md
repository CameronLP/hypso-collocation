# TODO

## Plotting
- [ ] Full-granule map: HYPSO, PACE and Sentinel-3 footprints on one map, with one band or chlorophyll per source.
- [ ] Shared colour scale: bright HYPSO coastal pixels stretch the 2nd to 98th percentile range for all panels. Add a fixed range or per-panel scales.
- [ ] Relief (hillshade) shading. Needs the Natural Earth shaded-relief raster, which is not on this machine.

## Downloads
- [ ] Test `download_operational.py` and the automatic download in `colocate_capture.py` end to end. Only searches have been tested.
- [ ] Check that NASA Earthdata and CDSE credentials are available on the processing machine.

## Colocation
- [ ] Sentinel-3 quality flags (`wqsf`) and PACE `l2_flags` are not applied.
- [ ] Decide whether `--grid region` should default to a latitude-longitude grid for full-granule extents, since stereographic distorts large areas.
- [ ] Check the bilinear output on the HYPSO swath against true bilinear on a regular grid.

## Repository
- [ ] Add a test fixture for a small capture, so the readers and resamplers can be tested without the full data.
