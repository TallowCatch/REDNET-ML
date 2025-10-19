#!/usr/bin/env python3
import argparse, glob, os
from pathlib import Path
import pandas as pd
import numpy as np

KEY_ORDER = [
    "tag","source_csv","scene_id","tile","datetime","year","month","season",
    "lon","lat","xmin","xmax","ymin","ymax","valid_px",
    "chlor_a","flh","kd490","fai_mean","rednir_mean","ndwi_mean",
    "hab_label"
]

def season_of_month(m):
    return ("winter","spring","summer","autumn")[(m%12)//3]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # nflh → flh
    if "flh" not in df.columns and "nflh" in df.columns:
        df = df.rename(columns={"nflh":"flh"})
    # Kd_490 → kd490
    if "kd490" not in df.columns and "Kd_490" in df.columns:
        df = df.rename(columns={"Kd_490":"kd490"})
    # numeric coercion
    for c in ["chlor_a","flh","kd490","fai_mean","rednir_mean","ndwi_mean","valid_px",
              "lon","lat","xmin","xmax","ymin","ymax","hab_label"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # time fields
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df["datetime"] = dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df["year"]  = dt.dt.year
        df["month"] = dt.dt.month
        df["season"]= dt.dt.month.apply(season_of_month)
    return df

def main():
    ap = argparse.ArgumentParser(description="Concatenate *_hab.csv into a single dataset.")
    ap.add_argument("--glob", default="data/aerial_*_20*/chip_indices_clean_hab.csv",
                    help="Glob of input CSVs")
    ap.add_argument("--out_csv", default="data/all_hab.csv", help="Output CSV path")
    ap.add_argument("--dedup", choices=["none","scene_tile","scene_tile_time"], default="scene_tile",
                    help="Deduplication strategy")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched: {args.glob}")

    frames = []
    for f in files:
        tag = Path(f).parent.name
        df = pd.read_csv(f)
        df = normalize_columns(df)
        df.insert(0, "tag", tag)
        df.insert(1, "source_csv", os.path.relpath(f))
        frames.append(df)

    # union of columns, aligned automatically by concat
    big = pd.concat(frames, ignore_index=True, sort=False)

    # robust de-dup
    if args.dedup != "none":
        keys = []
        if "scene_id" in big.columns: keys.append("scene_id")
        if "tile"     in big.columns: keys.append("tile")
        if args.dedup == "scene_tile_time" and "datetime" in big.columns:
            keys.append("datetime")
        if keys:
            before = len(big)
            big = big.drop_duplicates(subset=keys)
            print(f"De-duplicated by {keys}: {before} → {len(big)}")

    # reorder columns (keep everything else afterward)
    ordered = [c for c in KEY_ORDER if c in big.columns]
    remaining = [c for c in big.columns if c not in ordered]
    big = big[ordered + remaining]

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    big.to_csv(args.out_csv, index=False)
    # basic summary
    n = len(big)
    pos = int(pd.to_numeric(big.get("hab_label", pd.Series([0]*n)), errors="coerce").fillna(0).sum())
    print(f"✓ wrote {args.out_csv}  rows={n}, positives={pos}, files={len(files)}")

if __name__ == "__main__":
    main()
