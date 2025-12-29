#!/usr/bin/env python3
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Try a bunch of common column names seen in chip index exports
BBOX_CANDIDATES = [
    ("lon_min", "lat_min", "lon_max", "lat_max"),
    ("xmin", "ymin", "xmax", "ymax"),
    ("min_lon", "min_lat", "max_lon", "max_lat"),
    ("west", "south", "east", "north"),
    ("left", "bottom", "right", "top"),
]

CENTROID_CANDIDATES = [
    ("lon", "lat"),
    ("center_lon", "center_lat"),
    ("centroid_lon", "centroid_lat"),
    ("x_center", "y_center"),
]

TILE_COL_CANDIDATES = ["tile", "chip", "chip_id", "filename", "tile_name"]

def find_first_present(cols, candidates):
    cols_set = set([c.lower() for c in cols])
    for cand in candidates:
        if all(c.lower() in cols_set for c in cand):
            # return actual-cased column names
            mapping = {c.lower(): c for c in cols}
            return tuple(mapping[c.lower()] for c in cand)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips_glob", required=True, help="Glob for chip_indices_clean.csv files")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.chips_glob))
    if not paths:
        raise SystemExit(f"❌ No files matched chips_glob: {args.chips_glob}")

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["_source_file"] = str(p)
        frames.append(df)

    chips = pd.concat(frames, ignore_index=True)
    cols = list(chips.columns)

    tile_col = None
    for c in TILE_COL_CANDIDATES:
        if c in chips.columns:
            tile_col = c
            break
    if tile_col is None:
        raise SystemExit(f"❌ Could not find a tile column. Tried: {TILE_COL_CANDIDATES}\nColumns: {cols}")

    # 1) if centroid columns exist, use them
    cent = find_first_present(cols, CENTROID_CANDIDATES)
    if cent is not None:
        lon_c, lat_c = cent
        out = chips[[tile_col, lat_c, lon_c]].copy()
        out = out.rename(columns={tile_col: "tile", lat_c: "tile_lat", lon_c: "tile_lon"})
    else:
        # 2) else compute centroid from bbox columns
        bbox = find_first_present(cols, BBOX_CANDIDATES)
        if bbox is None:
            # Dump columns to help you immediately see what you have
            raise SystemExit(
                "❌ Could not find centroid or bbox columns in chip index files.\n"
                f"Columns found:\n{json.dumps(cols, indent=2)}\n\n"
                "Fix: tell me what bbox/coord columns exist in chip_indices_clean.csv "
                "and I’ll adapt this script in 1 minute."
            )
        x0, y0, x1, y1 = bbox
        out = chips[[tile_col, y0, x0, y1, x1]].copy()
        out["tile_lat"] = (out[y0].astype(float) + out[y1].astype(float)) / 2.0
        out["tile_lon"] = (out[x0].astype(float) + out[x1].astype(float)) / 2.0
        out = out.rename(columns={tile_col: "tile"})

    # Keep only unique tile -> centroid (first occurrence wins)
    out = out.dropna(subset=["tile_lat", "tile_lon"])
    out = out.drop_duplicates(subset=["tile"]).reset_index(drop=True)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"✅ wrote {out_path} rows={len(out)}")

if __name__ == "__main__":
    main()
