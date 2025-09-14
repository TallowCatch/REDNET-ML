# src/tools/make_chl_tiles_from_nc.py
# Build PNG tiles + labels CSV from OBPG L3m chlorophyll NetCDFs (AQUA_MODIS ... x_chlor_a.nc)
# Usage (from repo root, with env active):
#   python src/tools/make_chl_tiles_from_nc.py
# Optional args:
#   python src/tools/make_chl_tiles_from_nc.py --src data/chl_tiles/requested_files --out data/chl_tiles/tiles_png --csv data/labels/regression.csv

from __future__ import annotations
from pathlib import Path
import argparse
import math
import random
import sys

import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image

# --------------------------
# Defaults (can be overridden by CLI)
# --------------------------
DEF_SRC = Path("data/chl_tiles/requested_files")       # where your .nc files live
DEF_OUT = Path("data/chl_tiles/tiles_png")             # where PNG tiles are written
DEF_CSV = Path("data/labels/regression.csv")           # labels file (filepath,split,chl)
TILE_PX = 256                                          # tile size (pixels)
# Oman bounding box (lat, lon)
LAT_N, LON_W, LAT_S, LON_E = 26.5, 52.0, 16.0, 60.5

random.seed(42)

def parse_args():
    p = argparse.ArgumentParser(description="Make CHL tiles + labels from OBPG L3m NetCDFs")
    p.add_argument("--src", type=Path, default=DEF_SRC, help="Directory with .nc files")
    p.add_argument("--out", type=Path, default=DEF_OUT, help="Output directory for PNG tiles")
    p.add_argument("--csv", type=Path, default=DEF_CSV, help="Output CSV with labels")
    p.add_argument("--tile", type=int, default=TILE_PX, help="Tile size in pixels")
    p.add_argument("--vmin", type=float, default=0.0, help="min chl-a for 8-bit scaling")
    p.add_argument("--vmax", type=float, default=10.0, help="max chl-a for 8-bit scaling")
    p.add_argument("--nan_ratio", type=float, default=0.2, help="skip tiles with > this NaN fraction")
    return p.parse_args()

def to_uint8_gray(arr: np.ndarray, vmin=0.0, vmax=10.0) -> np.ndarray:
    """Scale float array to 0..255 grayscale."""
    arr = np.asarray(arr, dtype="float32")
    arr = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)

def open_nc_safe(p: Path) -> xr.Dataset | None:
    """
    Try multiple xarray engines (h5netcdf, netcdf4, scipy).
    Returns an opened Dataset or None if all fail.
    """
    for eng in ("h5netcdf", "netcdf4", "scipy"):
        try:
            ds = xr.open_dataset(
                p,
                engine=eng,
                mask_and_scale=True,
                decode_cf=True,
                decode_coords="all",
            )
            # quick sanity
            if len(ds.data_vars) == 0:
                ds.close()
                raise ValueError("no data_vars")
            return ds
        except Exception as e:
            print(f"Engine {eng} failed for {p.name}: {e}")
    return None

def find_chl_var(ds: xr.Dataset) -> str | None:
    """
    OBPG L3m files usually expose 'chlor_a' (and sometimes a 'palette' var).
    """
    for k in ds.data_vars:
        lk = k.lower()
        if "chlor_a" in lk or lk.startswith("chlor"):
            return k
    return None

def find_lon_lat_names(ds: xr.Dataset) -> tuple[str | None, str | None]:
    lon = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
    lat = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
    return lon, lat

def main():
    args = parse_args()
    src_dir: Path = args.src
    out_dir: Path = args.out
    csv_path: Path = args.csv
    tile_px: int = args.tile
    vmin, vmax = args.vmin, args.vmax
    nan_ratio = args.nan_ratio

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    nc_files = sorted(src_dir.glob("*.nc"))
    if not nc_files:
        print(f"No .nc files found in {src_dir.resolve()}")
        sys.exit(0)

    rows: list[tuple[str, float]] = []
    print(f"Found {len(nc_files)} NetCDF files in {src_dir}")

    for nc in nc_files:
        ds = open_nc_safe(nc)
        if ds is None:
            print(f"Skip {nc.name}: cannot open")
            continue

        chl_name = find_chl_var(ds)
        if chl_name is None:
            print(f"Skip {nc.name}: no chlorophyll variable among {list(ds.data_vars)}")
            ds.close()
            continue

        lon_name, lat_name = find_lon_lat_names(ds)
        if lon_name is None or lat_name is None:
            print(f"Skip {nc.name}: missing lon/lat coords; coords={list(ds.coords)}")
            ds.close()
            continue

        # Clip to Oman bbox; handle lat orientation (some files descend in latitude)
        lat_vals = ds[lat_name].values
        lat_asc = bool(lat_vals[-1] > lat_vals[0])

        sub = ds.sel(
            {
                lat_name: slice(LAT_S, LAT_N) if lat_asc else slice(LAT_N, LAT_S),
                lon_name: slice(LON_W, LON_E),
            }
        )

        # Some monthly mapped products have a singleton time dim; squeeze it out
        da = sub[chl_name].squeeze()

        # Replace fill/masked with NaN
        arr = np.asarray(da, dtype="float32")
        arr[~np.isfinite(arr)] = np.nan

        if arr.ndim != 2:
            print(f"Skip {nc.name}: expected 2D field after squeeze, got shape {arr.shape}")
            ds.close()
            continue

        H, W = arr.shape
        nrows = H // tile_px
        ncols = W // tile_px

        made = 0
        for i in range(nrows):
            for j in range(ncols):
                r0, c0 = i * tile_px, j * tile_px
                tile = arr[r0 : r0 + tile_px, c0 : c0 + tile_px]
                if tile.size == 0:
                    continue
                if np.isnan(tile).mean() > nan_ratio:
                    continue

                chl_mean = float(np.nanmean(tile))
                base = f"{nc.stem}_r{i:03d}_c{j:03d}"
                png_path = out_dir / f"{base}.png"
                Image.fromarray(to_uint8_gray(tile, vmin=vmin, vmax=vmax)).save(png_path)
                rows.append((str(png_path), chl_mean))
                made += 1

        print(f"{nc.name}: wrote {made} tiles")
        ds.close()

    # Train/val/test split and write CSV
    random.shuffle(rows)
    n = len(rows)
    n_tr = int(0.7 * n)
    n_va = int(0.15 * n)
    splits = (["train"] * n_tr) + (["val"] * n_va) + (["test"] * (n - n_tr - n_va))

    with open(csv_path, "w") as f:
        f.write("filepath,split,chl\n")
        for (fp, chl), sp in zip(rows, splits):
            f.write(f"{fp},{sp},{chl:.4f}\n")

    print(f"\nTiled {len(rows)} patches → {out_dir}")
    print(f"Wrote labels → {csv_path}")

if __name__ == "__main__":
    main()
