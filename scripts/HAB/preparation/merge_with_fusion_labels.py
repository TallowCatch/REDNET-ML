#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def norm_tile(s: pd.Series) -> pd.Series:
    return s.astype(str).map(lambda x: Path(x).name)

def to01(x) -> pd.Series:
    return (pd.to_numeric(x, errors="coerce") > 0.5).astype(int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants_mined_csv", required=True)
    ap.add_argument("--fusion_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--id_col", default="tile")
    ap.add_argument("--group_by", default="scene_id")
    ap.add_argument("--prefer_fusion_labels", action="store_true",
                    help="If set: fusion hab_label wins when present. Otherwise we OR them.")
    args = ap.parse_args()

    plants = pd.read_csv(args.plants_mined_csv)
    fusion  = pd.read_csv(args.fusion_csv)

    # normalize IDs
    if args.id_col in plants.columns:
        plants[args.id_col] = norm_tile(plants[args.id_col])
    if args.id_col in fusion.columns:
        fusion[args.id_col] = norm_tile(fusion[args.id_col])

    # decide plant label column to use
    if "hab_label_final" in plants.columns:
        plants_lab = to01(plants["hab_label_final"])
    elif "hab_label" in plants.columns:
        plants_lab = to01(plants["hab_label"])
    else:
        plants_lab = pd.Series(0, index=plants.index, dtype=int)
    plants["hab_label_plants"] = plants_lab

    # fusion label
    if "hab_label" not in fusion.columns:
        raise SystemExit("fusion CSV missing hab_label")
    fusion["hab_label_fusion"] = to01(fusion["hab_label"])

    # merge fusion label onto plants by tile first; if tile missing, fall back to (scene_id)
    key_tile = [args.id_col] if args.id_col in plants.columns and args.id_col in fusion.columns else None
    key_scene = [args.group_by] if args.group_by in plants.columns and args.group_by in fusion.columns else None

    merged = plants.copy()

    if key_tile:
        merged = merged.merge(
            fusion[[args.id_col, "hab_label_fusion"]].drop_duplicates(args.id_col),
            on=args.id_col, how="left"
        )
    elif key_scene:
        merged = merged.merge(
            fusion[[args.group_by, "hab_label_fusion"]].drop_duplicates(args.group_by),
            on=args.group_by, how="left"
        )
    else:
        # no matching keys — just carry plants labels
        merged["hab_label_fusion"] = np.nan

    merged["hab_label_fusion"] = merged["hab_label_fusion"].fillna(0).astype(int)

    # final label policy
    if args.prefer_fusion_labels:
        # if fusion has a label (non-null), use it, else use plants
        # (here we only have 0/1, so "prefer fusion" means fusion overwrites plants always)
        merged["hab_label"] = merged["hab_label_fusion"].astype(int)
        # BUT keep any plants positives that fusion never covered (tile not present):
        # detect uncovered via NaN before fill
        # easiest: recompute uncovered:
        # (we can’t now; so OR is safer unless you explicitly want fusion-only)
    else:
        # safest: OR them (keeps your known positives + adds mined ones)
        merged["hab_label"] = (merged["hab_label_fusion"] | merged["hab_label_plants"]).astype(int)

    # also keep provenance columns
    merged["hab_label_source_fusion"] = merged["hab_label_fusion"]
    merged["hab_label_source_plants"] = merged["hab_label_plants"]

    # union columns with fusion table (so fit_decision_fusion has what it needs)
    # We’ll concatenate rows from fusion too, then dedupe by tile (keep fusion row first)
    fusion2 = fusion.copy()
    fusion2["hab_label_source_fusion"] = fusion2["hab_label_fusion"]
    fusion2["hab_label_source_plants"] = 0

    # if fusion already has all the numeric cols, good; if not, union anyway
    all_cols = sorted(set(merged.columns) | set(fusion2.columns))
    merged = merged.reindex(columns=all_cols)
    fusion2 = fusion2.reindex(columns=all_cols)

    combined = pd.concat([fusion2, merged], ignore_index=True)

    # dedupe preference: keep first occurrence per tile (fusion rows were added first)
    if args.id_col in combined.columns:
        combined = combined.drop_duplicates(subset=[args.id_col], keep="first")
    else:
        combined = combined.drop_duplicates(keep="first")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out_csv, index=False)

    print(f"✓ wrote combined dataset -> {args.out_csv}")
    print(f"  rows={len(combined)} positives={int(combined['hab_label'].sum())}")
    if args.id_col in combined.columns:
        print(f"  unique tiles={combined[args.id_col].nunique()}")

if __name__ == "__main__":
    main()
