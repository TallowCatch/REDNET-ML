#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

old_csv = Path("runs/fusion/fusion_training_table_clean.csv")
new_csv = Path("runs/fusion/fusion_training_table_clean_populated.csv")
out_csv = Path("runs/fusion/fusion_training_table_merged.csv")

print("Loading…")
old = pd.read_csv(old_csv)
new = pd.read_csv(new_csv)

# normalize tile keys
old["tile"] = old["tile"].astype(str)
new["tile"] = new["tile"].astype(str)

# merge on tile
merged = new.merge(old, on="tile", how="left", suffixes=("", "_old"))

# fill missing values from old table
for col in new.columns:
    old_col = f"{col}_old"
    if old_col in merged.columns:
        merged[col] = merged[col].combine_first(merged[old_col])
        merged = merged.drop(columns=[old_col])

# drop duplicate rows if any
merged = merged.drop_duplicates()

merged.to_csv(out_csv, index=False)

print("\n✓ merged table saved:", out_csv)
print("rows:", len(merged))
print("columns:", len(merged.columns))

print("\ncoverage:")
for c in merged.columns:
    if merged[c].dtype != object:
        pct = merged[c].notna().mean() * 100
        print(f"{c:25s} {pct:6.1f}%")
