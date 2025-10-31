#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

fusion_path = Path("runs/fusion/fusion_training_table.csv")
df = pd.read_csv(fusion_path)

# Define feature groups
sentinel_feats = ["fai_mean", "rednir_mean", "ndwi_mean", "kd490", "chlor_a", "nflh"]
detector_feats = ["p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med", "p_tab"]

# Drop rows missing ALL Sentinel data
df = df.dropna(subset=sentinel_feats, how="all")

# Drop rows missing >2 Sentinel features (too incomplete)
df = df[df[sentinel_feats].isna().sum(axis=1) <= 2]

# Drop rows missing ANY detector score
df = df.dropna(subset=detector_feats, how="any")

# Fill derived features with medians
for col in ["month_sin", "month_cos", "ndwi_std", "rednir_std"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Save cleaned file
out = fusion_path.with_name("fusion_training_table_clean.csv")
df.to_csv(out, index=False)
print(f"✅ Cleaned and saved to {out}  ({len(df)} rows kept)")
