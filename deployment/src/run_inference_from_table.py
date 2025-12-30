#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_joblib", required=True, help="deployment/artifacts/model/fusion_model_cv4.joblib")
    ap.add_argument("--in_csv", required=True, help="Per-tile feature table for inference (like training table schema)")
    ap.add_argument("--out_csv", required=True, help="Output CSV with hab_prob")
    args = ap.parse_args()

    bundle = joblib.load(args.model_joblib)
    model = bundle["model"]
    features = bundle["features"]
    calibrator = bundle.get("calibrator", None)

    df = pd.read_csv(args.in_csv)

    # basic checks
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise SystemExit(f"❌ Missing required feature columns: {missing[:20]}{'...' if len(missing)>20 else ''}")

    X = df[features]

    # CatBoost: predict_proba works on pandas
    proba = model.predict_proba(X)[:, 1]

    if calibrator is not None:
        # if you ever used calibration in the future
        proba = calibrator.predict_proba(proba.reshape(-1, 1))[:, 1]

    out = df.copy()
    out["hab_prob"] = proba

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"✅ wrote {args.out_csv} rows={len(out)}")

if __name__ == "__main__":
    main()
