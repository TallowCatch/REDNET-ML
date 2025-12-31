#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import argparse, re, sys

def normalize_tile(s):
    """Strip extensions and trailing frame suffixes (_0000, .jpg, etc.)"""
    s = str(s)
    s = re.sub(r'\.jpg$', '', s, flags=re.I)
    s = re.sub(r'_\d{4}$', '', s)  # removes _0000, _0001, etc.
    return s

ap = argparse.ArgumentParser()
ap.add_argument("--labels_csv", required=True)
ap.add_argument("--det_csv", required=True)
ap.add_argument("--out_csv", required=True)
ap.add_argument("--label_col", default=None)
args = ap.parse_args()

labels = pd.read_csv(args.labels_csv)
det = pd.read_csv(args.det_csv)

# --- find label col ---
label_col = args.label_col
if label_col is None:
    for c in ["hab_label_final", "hab_label", "label"]:
        if c in labels.columns:
            label_col = c
            break
if label_col is None:
    sys.exit("❌ No label column found in labels CSV.")

# --- normalise ID cols ---
if "tile" not in labels.columns:
    sys.exit("❌ labels CSV must have a 'tile' column.")
if "chip_id" in det.columns and "tile" not in det.columns:
    det = det.rename(columns={"chip_id": "tile"})

labels["tile_norm"] = labels["tile"].map(normalize_tile)
det["tile_norm"] = det["tile"].map(normalize_tile)

merged = labels.merge(det, on="tile_norm", how="left", suffixes=("", "_det"))

# rename for consistency
merged = merged.rename(columns={label_col: "hab_label"})
merged = merged.drop(columns=["tile_norm"])

Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(args.out_csv, index=False)

match_col = [c for c in det.columns if c.startswith("p_") or c.endswith("_count")]
matched = merged[match_col[0]].notna().sum() if match_col else 0

print(f"✅ Merged {Path(args.out_csv).name} using label_col='{label_col}' — {matched} matched out of {len(merged)}")
