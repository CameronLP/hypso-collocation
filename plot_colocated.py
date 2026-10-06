"""Map plots for a colocated HYPSO / PACE / Sentinel-3 file, or a native product.

Usage:
    python plot_colocated.py <colocated.nc> --band 560      # one band, three panels
    python plot_colocated.py <colocated.nc> --rgb           # RGB composite per source
    python plot_colocated.py <colocated.nc> --chl           # chlorophyll, three panels
    python plot_colocated.py <PACE AOP .nc> --native --band 560     # one native product
    python plot_colocated.py <PACE BGC .nc> --native --chl          # native chlorophyll
    python plot_colocated.py <S3 product dir> --native --band 560
    python plot_colocated.py <S3 product dir> --native --chl

Output PNG is written next to the input unless --out is given.

Band and chlorophyll panels share one colour scale (2nd to 98th percentile of
all valid pixels). RGB uses a per-channel 2-98 percent stretch. Red, green and
blue are the bands nearest 665, 560 and 490 nm.

Maps use plain longitude/latitude axes. Coastlines are not drawn, so no map
data is downloaded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

RGB_BANDS = (665.0, 560.0, 490.0)


def _nearest(wavelengths: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(wavelengths) - target)))


def _stretch(values: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    a, b = np.percentile(finite, [lo, hi])
    if b <= a:
        b = a + 1e-12
    return np.clip((values - a) / (b - a), 0, 1)


def _shared_limits(arrays: list[np.ndarray]) -> tuple[float, float]:
    finite = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays if np.isfinite(a).any()])
    return tuple(np.percentile(finite, [2, 98]))


_BASEMAP_CACHE: dict = {}


def _clipped(category: str, name: str, extent):
    """Natural Earth 10 m geometries inside extent = (lon_min, lon_max, lat_min, lat_max).
    Data come from cartopy's local cache (no download)."""
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import box
    key = (category, name)
    if key not in _BASEMAP_CACHE:
        path = shpreader.natural_earth(resolution="10m", category=category, name=name)
        _BASEMAP_CACHE[key] = list(shpreader.Reader(path).geometries())
    window = box(*extent)
    clipped = []
    for geom in _BASEMAP_CACHE[key]:
        if geom.intersects(window):
            clipped.append(geom.intersection(window))
    return clipped


def basemap(ax, lon, lat, land=True, coast=True, ocean=False):
    """Draw land fill and coastlines over the axes' current data extent.

    land    grey land polygons (Natural Earth 10 m land)
    coast   black coastline (Natural Earth 10 m coastline)
    ocean   light blue ocean fill, drawn under the data (off by default)
    """
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.patches import Polygon
    extent = (float(np.nanmin(lon)), float(np.nanmax(lon)), float(np.nanmin(lat)), float(np.nanmax(lat)))
    if ocean:
        ax.add_patch(Polygon([(extent[0], extent[2]), (extent[1], extent[2]),
                              (extent[1], extent[3]), (extent[0], extent[3])],
                             closed=True, facecolor="#dbe9f6", edgecolor="none", zorder=0))
    if land:
        patches = []
        for geom in _clipped("physical", "land", extent):
            polys = geom.geoms if hasattr(geom, "geoms") else [geom]
            for poly in polys:
                if poly.is_empty or poly.geom_type != "Polygon":
                    continue
                patches.append(Polygon(np.asarray(poly.exterior.coords), closed=True))
                patches.extend(Polygon(np.asarray(r.coords), closed=True) for r in poly.interiors)
        ax.add_collection(PatchCollection(patches, facecolor="0.55", edgecolor="none", zorder=3))
    if coast:
        lines = []
        for geom in _clipped("physical", "coastline", extent):
            parts = geom.geoms if hasattr(geom, "geoms") else [geom]
            lines.extend(np.asarray(part.coords) for part in parts if not part.is_empty)
        ax.add_collection(LineCollection(lines, colors="black", linewidths=0.4, zorder=4))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])


def _panel(ax, lon, lat, z, title, cmap, vmin, vmax, norm=None):
    mesh = ax.pcolormesh(lon, lat, z, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, shading="auto")
    basemap(ax, lon, lat)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect(1.0 / np.cos(np.deg2rad(float(np.nanmean(lat)))))
    ax.grid(True, alpha=0.3)
    return mesh


def _rgb_image(cube: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """RGBA image: no-data pixels have alpha 0, so the axes background shows."""
    channels = [cube[:, :, _nearest(wavelengths, nm)] for nm in RGB_BANDS]
    rgb = np.stack([_stretch(c) for c in channels], axis=-1)
    valid = np.isfinite(np.stack(channels, axis=-1)).any(axis=-1)
    rgb[~valid] = 0.0
    return np.dstack([rgb, valid.astype(float)])


def plot_colocated(path: Path, mode: str, band: float | None, out: Path) -> Path:
    ds = xr.open_dataset(path)
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    sources = []
    if "hypso_bands" in ds:
        sources.append(("HYPSO", ds["hypso_bands"].values, ds["hypso_wavelength"].values))
    if "pace_rrs" in ds:
        sources.append(("PACE OCI Rrs", ds["pace_rrs"].values, ds["pace_wavelength"].values))
    if "s3_rrs" in ds:
        sources.append(("Sentinel-3 OLCI Rrs", ds["s3_rrs"].values, ds["s3_wavelength"].values))

    if mode == "rgb":
        fig, axes = plt.subplots(1, len(sources), figsize=(5.2 * len(sources), 5), squeeze=False)
        for ax, (name, cube, wl) in zip(axes[0], sources):
            img = _rgb_image(cube, wl)
            ax.set_facecolor("white")
            ax.imshow(img, extent=[np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)],
                      origin="upper", aspect="auto", interpolation="nearest")
            basemap(ax, lon, lat)
            ax.set_title(f"{name}\nR {RGB_BANDS[0]:.0f} G {RGB_BANDS[1]:.0f} B {RGB_BANDS[2]:.0f} nm", fontsize=10)
            ax.set_xlabel("longitude")
            ax.set_ylabel("latitude")
        suptitle = f"RGB composite, {ds.attrs.get('hypso_capture', path.stem)}"

    elif mode == "band":
        fields = []
        for name, cube, wl in sources:
            i = _nearest(wl, band)
            fields.append((f"{name}, {wl[i]:.1f} nm", cube[:, :, i]))
        vmin, vmax = _shared_limits([f for _, f in fields])
        fig, axes = plt.subplots(1, len(fields), figsize=(5.2 * len(fields), 5), squeeze=False)
        mesh = None
        for ax, (title, z) in zip(axes[0], fields):
            mesh = _panel(ax, lon, lat, z, title, "viridis", vmin, vmax)
        fig.colorbar(mesh, ax=axes[0].tolist(), shrink=0.8, label="Rrs (sr$^{-1}$)")
        suptitle = f"Rrs at ~{band:.0f} nm, {ds.attrs.get('hypso_capture', path.stem)}"

    elif mode == "chl":
        fields = []
        for name, key in [("PACE BGC chlor_a", "pace_chlor_a"), ("S3 OC4ME chlor_a", "s3_chl_oc4me"),
                          ("S3 CHL_NN", "s3_chl_nn")]:
            if key in ds:
                fields.append((name, ds[key].values))
        if not fields:
            raise SystemExit("no chlorophyll variables in this file")
        vmin, vmax = _shared_limits([f for _, f in fields])
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=max(vmin, 1e-3), vmax=max(vmax, 1e-2))
        fig, axes = plt.subplots(1, len(fields), figsize=(5.2 * len(fields), 5), squeeze=False)
        mesh = None
        for ax, (title, z) in zip(axes[0], fields):
            mesh = _panel(ax, lon, lat, z, title, "viridis", None, None, norm=norm)
        fig.colorbar(mesh, ax=axes[0].tolist(), shrink=0.8, label="chlorophyll (mg m$^{-3}$)")
        suptitle = f"Chlorophyll, {ds.attrs.get('hypso_capture', path.stem)}"
    else:
        raise SystemExit(f"unknown mode {mode}")

    fig.suptitle(suptitle, fontsize=11)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_native(path: Path, mode: str, band: float | None, out: Path) -> Path:
    import pace_source
    import sentinel3_source

    path = Path(path)
    if path.is_dir():
        swath = sentinel3_source.read_wfr_swath(path)
        lat, lon = swath.latitude, swath.longitude
        if mode == "chl":
            panels = [("S3 OC4ME chlor_a", swath.chl_oc4me), ("S3 CHL_NN", swath.chl_nn)]
            label, cmap_log = "chlorophyll (mg m$^{-3}$)", True
        else:
            i = _nearest(swath.rrs_wavelengths, band)
            panels = [(f"Sentinel-3 OLCI Rrs, {swath.rrs_wavelengths[i]:.1f} nm", swath.rrs[:, :, i])]
            label, cmap_log = "Rrs (sr$^{-1}$)", False
        title = path.name
    elif "OC_BGC" in path.name:
        bgc = pace_source.read_l2_bgc_swath(path)
        lat, lon = bgc.latitude, bgc.longitude
        if mode != "chl":
            raise SystemExit("a BGC file only has chlorophyll; use --chl")
        panels = [("PACE BGC chlor_a", bgc.chlor_a)]
        label, cmap_log = "chlorophyll (mg m$^{-3}$)", True
        title = path.name
    elif "OC_AOP" in path.name:
        aop = pace_source.read_l2_aop_swath(path)
        lat, lon = aop.latitude, aop.longitude
        if mode == "chl":
            if aop.chlor_a is None:
                raise SystemExit("this AOP file has no chlorophyll; use --chl with the BGC file")
            panels = [("PACE AOP chlor_a", aop.chlor_a)]
            label, cmap_log = "chlorophyll (mg m$^{-3}$)", True
        else:
            i = _nearest(aop.rrs_wavelengths, band)
            panels = [(f"PACE OCI Rrs, {aop.rrs_wavelengths[i]:.1f} nm", aop.rrs[:, :, i])]
            label, cmap_log = "Rrs (sr$^{-1}$)", False
        title = path.name
    else:
        raise SystemExit(f"unrecognised product {path.name}: expected a PACE OC_AOP or OC_BGC file, "
                         f"or a Sentinel-3 product directory")

    vmin, vmax = _shared_limits([z for _, z in panels])
    from matplotlib.colors import LogNorm
    norm = LogNorm(vmin=max(vmin, 1e-3), vmax=max(vmax, 1e-2)) if cmap_log else None
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 5), squeeze=False)
    mesh = None
    for ax, (name, z) in zip(axes[0], panels):
        mesh = _panel(ax, lon, lat, z, name, "viridis", None if norm else vmin, None if norm else vmax, norm=norm)
    fig.colorbar(mesh, ax=axes[0].tolist(), shrink=0.8, label=label)
    fig.suptitle(title, fontsize=11)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--band", type=float, help="wavelength in nm")
    group.add_argument("--rgb", action="store_true")
    group.add_argument("--chl", action="store_true")
    parser.add_argument("--native", action="store_true", help="plot a native PACE or Sentinel-3 product")
    parser.add_argument("--out", type=Path, help="output PNG (default: next to the input)")
    args = parser.parse_args()

    mode = "rgb" if args.rgb else "chl" if args.chl else "band"
    if args.native and mode == "rgb":
        parser.error("--rgb is for colocated files; use --band for native products")
    stem = args.path.name if args.path.is_dir() else args.path.stem
    out = args.out or args.path.parent / f"{stem}-{mode}{'' if args.band is None else int(args.band)}.png"
    result = plot_native(args.path, mode, args.band, out) if args.native else plot_colocated(args.path, mode, args.band, out)
    print(f"wrote {result}")
