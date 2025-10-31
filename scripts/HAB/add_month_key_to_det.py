#!/usr/bin/env python3
"""
Add a `month_key` column to MODIS detector CSVs by parsing the chip_id field.

Usage:
  python scripts/HAB/add_month_key_to_det.py input.csv output.csv
Example:
  python scripts/HAB/add_month_key_to_det.py runs/fusion/p_frcnn_r50.csv runs/fusion/p_frcnn_r50_mkey.csv
"""

import re
import pandas as pd
import sys
from pathlib import Path

if len(sys.argv) != 3:
    sys.exit("Usage: python add_month_key_to_det.py input.csv output.csv")

in_csv, out_csv = sys.argv[1], sys.argv[2]

df = pd.read_csv(in_csv)

# Typical chip_id looks like:
# AQUA_MODIS.20150301_20150331.L3m.MO.CHL.x_chlor_a_r002_c002
# We’ll extract '2015-03' from that pattern
pat = re.compile(r'\.(20\d{2})(\d{2})\d{2}_')

def get_month_key(chip):
    if pd.isna(chip):
        return None
    m = pat.search(str(chip))
    if not m:
        return None
    year, month = m.group(1), m.group(2)
    return f"{year}-{month}"

if "chip_id" not in df.columns:
    sys.exit("❌ Input file must have a 'chip_id' column")

df["month_key"] = df["chip_id"].map(get_month_key)

# reorder for readability
cols = list(df.columns)
if "month_key" in cols:
    cols.insert(1, cols.pop(cols.index("month_key")))
df = df[cols]

Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)

print(f"✅ wrote {out_csv}")
print(f"   total rows: {len(df)}")
print(f"   unique month_keys: {df['month_key'].nunique()}")
print(df["month_key"].value_counts().head())
