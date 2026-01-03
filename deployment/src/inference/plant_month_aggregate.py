#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def risk_band(p: float) -> str:
    # simple default bands (tune later using your CV4 thresholds if you want)
    if p >= 0.53277: return "high"
    if p >= 0.39263: return "medium"
    return "low"

def main():
    ap = argparse.ArgumentParser("Aggregate chip-level inference into plant-month risk")
    ap.add_argument("--in_csv", required=True, help="deployment/outputs/.../inference_all_months.csv")
    ap.add_argument("--out_csv", required=True, help="Output plant_month_risk.csv")
    ap.add_argument("--plant_id", default=None, help="Plant id/name to store in output (optional)")
    ap.add_argument("--prob_col", default="hab_prob")
    ap.add_argument("--month_col", default="month")  # you already write this
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    if args.prob_col not in df.columns:
        raise SystemExit(f"Missing prob_col={args.prob_col}")
    if args.month_col not in df.columns:
        raise SystemExit(f"Missing month_col={args.month_col}")

    # Ensure numeric
    df[args.prob_col] = pd.to_numeric(df[args.prob_col], errors="coerce")

    def p95(x):
        x = x.dropna().to_numpy()
        return float(np.percentile(x, 95)) if len(x) else np.nan

    out = (
        df.groupby(args.month_col, as_index=False)
          .agg(
              n_chips=(args.prob_col, "size"),
              hab_mean=(args.prob_col, "mean"),
              hab_p95=(args.prob_col, p95),
              hab_max=(args.prob_col, "max"),
              frac_ge_05=(args.prob_col, lambda s: float((s >= 0.5).mean())),
              frac_ge_08=(args.prob_col, lambda s: float((s >= 0.8).mean())),
          )
    )

    out["risk"] = out["hab_p95"].map(risk_band)

    if args.plant_id:
        out.insert(0, "plant_id", args.plant_id)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"✅ wrote {args.out_csv} rows={len(out)}")

if __name__ == "__main__":
    main()
