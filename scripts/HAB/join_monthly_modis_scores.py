#!/usr/bin/env python3
"""
Join MODIS detector monthly scores with Sentinel-2 labeled rows.
- Sentinel: runs/datasets/hab_train_mined_aslabel.csv   (has hab_label_final)
- MODIS det: runs/fusion/p_frcnn_r50_mkey.csv, p_frcnn_mb_mkey.csv, p_frcnn_ssd_mkey.csv

Result: runs/fusion/fusion_training_table.csv
"""

import pandas as pd
from pathlib import Path

# 1) load Sentinel-2 labels/features
sent_path = "runs/datasets/hab_train_mined_aslabel.csv"
sent = pd.read_csv(sent_path)

# pick the correct label col
label_col = None
if "hab_label_final" in sent.columns:
    label_col = "hab_label_final"
elif "hab_label" in sent.columns:
    label_col = "hab_label"
else:
    raise SystemExit("❌ Sentinel CSV has no hab_label / hab_label_final")

# 2) load MODIS detector scores (already with month_key)
r50 = pd.read_csv("runs/fusion/p_frcnn_r50_mkey.csv")
mb  = pd.read_csv("runs/fusion/p_frcnn_mb_mkey.csv")
ssd = pd.read_csv("runs/fusion/p_frcnn_ssd_mkey.csv")

# 3) collapse to per-month medians
r50_m = r50.groupby("month_key")["p_frcnn_r50"].median().rename("p_frcnn_r50_med")
mb_m  = mb.groupby("month_key")["p_frcnn_mb"].median().rename("p_frcnn_mb_med")

# SSD can be named two ways
ssd_score_col = None
for cand in ["p_frcnn_ssd", "p_ssd_mb"]:
    if cand in ssd.columns:
        ssd_score_col = cand
        break
if ssd_score_col is None:
    raise SystemExit("❌ SSD CSV is missing p_frcnn_ssd / p_ssd_mb")

ssd_m = ssd.groupby("month_key")[ssd_score_col].median().rename("p_ssd_mb_med")

# 4) join 3 detector summaries
det_month = (
    pd.concat([r50_m, mb_m, ssd_m], axis=1)
    .reset_index()
)

# 5) merge with Sentinel tiles by month_key
fusion_df = sent.merge(det_month, on="month_key", how="left")

# 6) make a single label column called hab_label
fusion_df["hab_label"] = fusion_df[label_col].astype(int)

# if there was ALSO a plain hab_label and we used hab_label_final, drop the old one
if label_col == "hab_label_final" and "hab_label" in sent.columns:
    # we already wrote the new hab_label above, so drop the sentinel's old 'hab_label'
    fusion_df = fusion_df.drop(columns=["hab_label_final"], errors="ignore")
else:
    # we used plain hab_label, just drop duplicate name if any
    fusion_df = fusion_df.loc[:, ~fusion_df.columns.duplicated()]

# 7) keep useful columns
keep_cols = [
    "tile", "scene_id", "datetime", "month_key",
    "fai_mean", "rednir_mean", "ndwi_mean",
    "kd490", "chlor_a", "nflh",
    "month_sin", "month_cos", "ndwi_std", "rednir_std",
    "hab_label",
    "p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med",
]
fusion_df = fusion_df[[c for c in keep_cols if c in fusion_df.columns]]

# fusion_df["p_tab"] = fusion_df["rednir_mean"]

# 8) write
out_csv = "runs/fusion/fusion_training_table.csv"
Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
fusion_df.to_csv(out_csv, index=False)

print(f"✅ wrote {out_csv} with {len(fusion_df)} rows")
print("   class balance (final):\n", fusion_df["hab_label"].value_counts(dropna=False))
print("   MODIS coverage per col:\n", fusion_df[["p_frcnn_r50_med","p_frcnn_mb_med","p_ssd_mb_med"]].notna().sum())
