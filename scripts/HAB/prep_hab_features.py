#!/usr/bin/env python3
import argparse, glob, os
from pathlib import Path
import pandas as pd
import numpy as np

def season_of_month(m):  # 1..12
    return ("winter","winter","spring","spring","spring",
            "summer","summer","summer","autumn","autumn","autumn","winter")[m-1]

def add_time_features(df):
    if "datetime" not in df.columns: return df
    dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.copy()
    df["year"]  = dt.dt.year
    df["month"] = dt.dt.month
    # cyclical encoding (avoid ordinal trap)
    ang = 2*np.pi*(df["month"].astype(float)/12.0)
    df["month_sin"] = np.sin(ang)
    df["month_cos"] = np.cos(ang)
    # coarse season string + one-hot
    df["season"] = df["month"].fillna(1).astype(int).clip(1,12).map(season_of_month)
    for s in ["winter","spring","summer","autumn"]:
        df[f"season_{s}"] = (df["season"]==s).astype(int)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/aerial_*_20*/chip_indices_clean_hab.csv")
    ap.add_argument("--out_csv", default="runs/datasets/hab_train_nonleaky.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched: {args.glob}")

    rows = []
    for f in files:
        tag = Path(f).parent.name
        df = pd.read_csv(f)
        df["__tag__"] = tag
        df["__src__"] = f
        rows.append(df)
    full = pd.concat(rows, ignore_index=True)

    # normalize likely numeric columns
    for c in ["fai_mean","rednir_mean","ndwi_mean","chlor_a","kd490","flh","nflh","valid_px"]:
        if c in full.columns:
            full[c] = pd.to_numeric(full[c], errors="coerce")
    # unifying FLH
    if "flh" not in full.columns and "nflh" in full.columns:
        full = full.rename(columns={"nflh":"flh"})

    # keep only labeled rows
    if "hab_label" not in full.columns:
        raise SystemExit("Input CSVs must have 'hab_label'. Run make_hab_labels.py first.")
    full["hab_label"] = (pd.to_numeric(full["hab_label"], errors="coerce")>0.5).astype(int)

    # add time features
    full = add_time_features(full)

    # carry forward a scene grouping if available
    if "scene_id" not in full.columns:
        full["scene_id"] = full["__src__"].str.extract(r'/(S2[ABC]_MSIL2A_[^/_]+)')[0].fillna("NA")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out, index=False)
    print(f"✓ wrote {out}  (rows={len(full)}, positives={int(full['hab_label'].sum())})")

if __name__ == "__main__":
    main()
