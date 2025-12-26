#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

fusion_path = Path("runs/fusion/fusion_training_table.csv")
df = pd.read_csv(fusion_path)

sentinel_feats_all = ["fai_mean", "rednir_mean", "ndwi_mean", "kd490", "chlor_a", "nflh", "sst"]
detector_feats_all = ["p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med", "p_tab"]

# Only keep columns that actually exist
sentinel_feats = [c for c in sentinel_feats_all if c in df.columns]
detector_feats = [c for c in detector_feats_all if c in df.columns]

if sentinel_feats:
    df = df.dropna(subset=sentinel_feats, how="all")
    df = df[df[sentinel_feats].isna().sum(axis=1) <= 2]

if detector_feats:
    df = df.dropna(subset=detector_feats, how="any")

for col in ["month_sin", "month_cos", "ndwi_std", "rednir_std"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

out = fusion_path.with_name("fusion_training_table_clean.csv")
df.to_csv(out, index=False)
print(f"✅ Cleaned and saved to {out}  ({len(df)} rows kept)")
print("   columns:", list(df.columns))
