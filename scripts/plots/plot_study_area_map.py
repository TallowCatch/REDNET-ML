from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import contextily as cx

from shapely.geometry import Point
from shapely import wkt
from pyproj import Transformer

from sentinel_tiles import sentinel_tiles


# ============================================================
# CONFIG
# ============================================================
CSV_PATH = Path("runs/datasets/fusion_training_with_plants_splitlabels.csv")
OUT_PNG  = Path("runs/plots/study_area_chlor_a_final.png")

COLOR_BY = "chlor_a"
AGG = "median"
BUFFER_KM = 250

BASEMAP = cx.providers.Esri.WorldImagery

PLANTS = [
    {"name": "A", "lat": 21.9310725, "lon": 59.6321193},
    {"name": "B", "lat": 22.622075,  "lon": 59.452973},
    {"name": "C", "lat": 25.6716585, "lon": 56.2667608},
    {"name": "D", "lat": 23.6018997, "lon": 58.4137212},
]

TILE_RE = re.compile(r"_T(\d{2}[A-Z]{3})_")


# ============================================================
# HELPERS
# ============================================================
def extract_mgrs(tile: str) -> str | None:
    m = TILE_RE.search(str(tile))
    return m.group(1) if m else None


def agg_series(s: pd.Series, how: str) -> float:
    if how == "median":
        return float(s.median())
    if how == "mean":
        return float(s.mean())
    if how == "max":
        return float(s.max())
    raise ValueError(how)


def to_geom(x):
    return wkt.loads(x.wkt) if hasattr(x, "wkt") else wkt.loads(str(x))


def add_lonlat_ticks(ax, n=5):
    """
    Add geographic (lon/lat) ticks that stay INSIDE the map extent.
    Designed for EPSG:3857 axes.
    """
    transformer = Transformer.from_crs(3857, 4326, always_xy=True)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # inset so labels never sit on the frame
    inset_x = 0.02 * (xmax - xmin)
    inset_y = 0.02 * (ymax - ymin)

    xticks = np.linspace(xmin + inset_x, xmax - inset_x, n)
    yticks = np.linspace(ymin + inset_y, ymax - inset_y, n)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    lon_labels = [
        f"{transformer.transform(x, ymin)[0]:.1f}°E"
        for x in xticks
    ]
    lat_labels = [
        f"{transformer.transform(xmin, y)[1]:.1f}°N"
        for y in yticks
    ]

    ax.set_xticklabels(lon_labels, fontsize=8)
    ax.set_yticklabels(lat_labels, fontsize=8)

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=4,
        width=0.8,
        bottom=True,
        left=True,
        top=False,
        right=False,
        pad=4,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


# ============================================================
# MAIN
# ============================================================
def main():

    # ------------------------
    # Load & aggregate data
    # ------------------------
    df = pd.read_csv(CSV_PATH)
    df["mgrs"] = df["tile"].apply(extract_mgrs)
    df = df.dropna(subset=["mgrs"])

    if "valid_px" in df.columns:
        df = df[df["valid_px"] > 0]

    tile_vals = (
        df.groupby("mgrs")[COLOR_BY]
          .apply(lambda s: agg_series(s, AGG))
          .reset_index(name="val")
    )

    geoms = [to_geom(sentinel_tiles.footprint(t)) for t in tile_vals["mgrs"]]
    gdf = gpd.GeoDataFrame(tile_vals, geometry=geoms, crs="EPSG:4326")

    plants = gpd.GeoDataFrame(
        PLANTS,
        geometry=[Point(p["lon"], p["lat"]) for p in PLANTS],
        crs="EPSG:4326"
    )

    gdf = gdf.to_crs(epsg=3857)
    plants = plants.to_crs(epsg=3857)

    # ------------------------
    # Spatial filtering
    # ------------------------
    buffer = plants.buffer(BUFFER_KM * 1000).unary_union
    gdf = gdf[gdf.geometry.intersects(buffer)]

    vmin = gdf["val"].quantile(0.05)
    vmax = gdf["val"].quantile(0.95)

    # ------------------------
    # Plot
    # ------------------------
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(left=0.06, right=0.86, top=0.93, bottom=0.06)

    gdf.plot(
        column="val",
        ax=ax,
        cmap="viridis",
        alpha=0.45,
        edgecolor="black",
        linewidth=0.7,
        vmin=vmin,
        vmax=vmax,
        zorder=3,
    )

    cx.add_basemap(ax, source=BASEMAP, attribution=False, zorder=1)

    plants.plot(
        ax=ax,
        marker="^",
        markersize=140,
        color="white",
        edgecolor="black",
        linewidth=1.2,
        zorder=6,
    )

    # Plant labels
    for _, r in plants.iterrows():
        t = ax.text(
            r.geometry.x,
            r.geometry.y + 14_000,
            r["name"],
            fontsize=8.5,
            weight="bold",
            color="white",
            ha="center",
            va="bottom",
            zorder=7,
        )
        t.set_path_effects([
            pe.Stroke(linewidth=1.2, foreground="black"),
            pe.Normal()
        ])

    # ------------------------
    # Lock map extent (prevents tick drift)
    # ------------------------
    xmin, ymin, xmax, ymax = gdf.total_bounds
    pad = 40_000  # meters
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    # ------------------------
    # Colorbar (tight + clean)
    # ------------------------
    cax = fig.add_axes([0.840, 0.25, 0.014, 0.50])
    cb = fig.colorbar(ax.collections[0], cax=cax)
    cb.set_label(
        "chlorophyll-a (median, mg m⁻³)",
        fontsize=9,
        labelpad=10
    )

    # ------------------------
    # Title & coordinates
    # ------------------------
    ax.set_title(
        "Study Area and Sentinel-2 MGRS Tiles Used for HAB Fusion Model",
        fontsize=12
    )

    add_lonlat_ticks(ax)

    # ------------------------
    # Save
    # ------------------------
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300)
    print(f"Saved → {OUT_PNG}")


if __name__ == "__main__":
    main()
