#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

def main():
    pred_path = Path("runs/fusion/fused_sets/B_mined_timecv_norm_f1/merged_features_debug.csv")
    outdir = Path("runs/fusion/fused_sets")
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path)
    print(f"Loaded {len(df)} rows from {pred_path}")

    # detect available detector columns
    detectors = [c for c in df.columns if c.startswith("frcnn_")]
    if not detectors:
        raise SystemExit("❌ No frcnn_* columns found in merged_features_debug.csv")

    # base columns
    base_cols = ["tile", "hab_label"]
    for col in detectors:
        short = col.split("_", 1)[1]  # e.g. mb, r50, ssd
        out_csv = outdir / f"p_frcnn_with_HAB_label_{short}.csv"
        out = df[base_cols + [col]].rename(columns={col: f"p_frcnn_{short}"})
        out.to_csv(out_csv, index=False)
        print(f"✓ Wrote {out_csv} ({len(out)} rows)")

    print("✅ Extraction complete. Ready for build_detector_comparison.py")

if __name__ == "__main__":
    main()
