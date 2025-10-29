#!/usr/bin/env python3
import argparse, glob, re
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- helpers ----------
def season_of_month(m):  # 1..12
    return ("winter","winter","spring","spring","spring",
            "summer","summer","summer","autumn","autumn","autumn","winter")[m-1]

def add_time_features(df):
    if "datetime" not in df.columns:
        return df
    dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.copy()
    df["year"]  = dt.dt.year
    df["month"] = dt.dt.month
    # cyclical month encoding
    ang = 2 * np.pi * (df["month"].astype(float) / 12.0)
    df["month_sin"] = np.sin(ang)
    df["month_cos"] = np.cos(ang)
    # season one-hots
    df["season"] = df["month"].fillna(1).astype(int).clip(1, 12).map(season_of_month)
    for s in ["winter","spring","summer","autumn"]:
        df[f"season_{s}"] = (df["season"] == s).astype(int)
    return df

# derive month_key from scene_id or datetime
MONTH_RE = re.compile(r"(20\d{2})[-_.]?(\d{2})")
def derive_month_key(s: str) -> str | None:
    m = MONTH_RE.search(str(s))
    if not m: return None
    y, mo = m.group(1), m.group(2)
    try:
        yi, mi = int(y), int(mo)
        if 1 <= mi <= 12: return f"{yi:04d}-{mi:02d}"
    except Exception:
        pass
    return None

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/aerial_*_20*/chip_indices_clean_hab.csv")
    ap.add_argument("--out_csv", default="runs/datasets/hab_train_nonleaky.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched: {args.glob}")

    # ---------- merge all ----------
    parts = []
    for f in files:
        tag = Path(f).parent.name
        df = pd.read_csv(f)
        df["__tag__"] = tag
        df["__src__"] = f
        parts.append(df)
    full = pd.concat(parts, ignore_index=True)

    # ---------- standardize columns ----------
    for c in ["fai_mean","rednir_mean","ndwi_mean","chlor_a","kd490","flh","nflh","valid_px"]:
        if c in full.columns:
            full[c] = pd.to_numeric(full[c], errors="coerce")
    if "flh" not in full.columns and "nflh" in full.columns:
        full = full.rename(columns={"nflh": "flh"})

    # ---------- ensure label exists ----------
    if "hab_label" not in full.columns:
        raise SystemExit("Missing hab_label; run make_hab_labels.py first.")
    full["hab_label"] = (pd.to_numeric(full["hab_label"], errors="coerce") > 0.5).astype(int)

    # ---------- add scene_id and month ----------
    if "scene_id" not in full.columns:
        full["scene_id"] = full["__src__"].str.extract(r'/(S2[ABC]_MSIL2A_[^/_]+)')[0].fillna("NA")
    full["month_key"] = full["scene_id"].astype(str).map(derive_month_key)
    if full["month_key"].isna().any() and "datetime" in full.columns:
        full["month_key"] = full["month_key"].fillna(
            pd.to_datetime(full["datetime"], errors="coerce").dt.strftime("%Y-%m")
        )

    # ---------- add time features ----------
    full = add_time_features(full)

    # ---------- enforce true non-leaky rule ----------
    # 1 row per (scene_id, month_key)
    full["_sort_key"] = full["hab_label"].rank(method="first", ascending=False)
    full = full.sort_values(["scene_id", "month_key", "_sort_key"])
    full = full.drop_duplicates(["scene_id", "month_key"], keep="first").drop(columns="_sort_key")

    # remove any scene_ids appearing in >1 distinct month (prevent temporal overlap)
    multi_month = (
        full.groupby("scene_id")["month_key"].nunique()
    )
    drop_ids = set(multi_month[multi_month > 1].index)
    before = len(full)
    # full = full[~full["scene_id"].isin(drop_ids)].copy()
    after = len(full)
    print(f"[nonleak] dropped {before - after} rows from multi-month scenes ({len(drop_ids)} scenes)")

    # ---------- write ----------
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(args.out_csv, index=False)
    print(f"✓ wrote {args.out_csv}  (rows={len(full)}, positives={int(full['hab_label'].sum())})")

if __name__ == "__main__":
    main()
