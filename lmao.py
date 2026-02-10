#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = Path("data/sentinel2/chlor_a_pixels.csv")
MONTH = "2023-11"

OUT_PNG = Path("runs/plots/sentinel2_oci_chlor_a_nov2023.png")

GRID_RES_DEG = 0.01   # ~1 km grid
VMIN = 0.0
VMAX = 3.0            # Sentinel-2 lower than MODIS (this is CORRECT)
CMAP = "turbo"

# ============================================================
# LOAD + FILTER
# ============================================================
df = pd.read_csv(CSV_PATH)

required = {"lat", "lon", "chlor_a", "datetime"}
missing = required - set(df.columns)
assert not missing, f"Missing columns: {missing}"

df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df = df[df["datetime"].dt.strftime("%Y-%m") == MONTH]
df = df[df["chlor_a"] > 0]

# ============================================================
# GRID (MODIS-like compositing)
# ============================================================
lon_bins = np.arange(df.lon.min(), df.lon.max(), GRID_RES_DEG)
lat_bins = np.arange(df.lat.min(), df.lat.max(), GRID_RES_DEG)

df["lon_bin"] = pd.cut(df.lon, lon_bins, labels=lon_bins[:-1])
df["lat_bin"] = pd.cut(df.lat, lat_bins, labels=lat_bins[:-1])

grid = (
    df.groupby(["lat_bin", "lon_bin"])
      .chlor_a
      .median()
      .reset_index()
)

grid["lon"] = grid.lon_bin.astype(float)
grid["lat"] = grid.lat_bin.astype(float)

# ============================================================
# TO XARRAY
# ============================================================
chl = grid.set_index(["lat", "lon"]).to_xarray()["chlor_a"]

# Optional light smoothing (visual only)
chl = chl.rolling(lat=2, lon=2, center=True).mean()

# ============================================================
# PLOT
# ============================================================
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

pcm = ax.pcolormesh(
    chl.lon,
    chl.lat,
    chl,
    cmap=CMAP,
    vmin=VMIN,
    vmax=VMAX,
    shading="auto"
)

ax.coastlines(resolution="10m", linewidth=1)
ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=3)

gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
gl.top_labels = False
gl.right_labels = False

cb = plt.colorbar(pcm, ax=ax, shrink=0.8)
cb.set_label("Chlorophyll-a (mg m$^{-3}$) — Sentinel-2 OCI-style")

ax.set_title(
    "Sentinel-2 Chlorophyll-a (OCI-style)\nMonthly Composite — November 2023",
    fontsize=12
)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
print(f"Saved → {OUT_PNG}")
