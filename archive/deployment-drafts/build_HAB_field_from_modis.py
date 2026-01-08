#!/usr/bin/env python3
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modis_csv", required=True, help="CSV with MODIS grid values")
    ap.add_argument("--model", required=True, help="fusion_model.joblib")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--time_col", default="datetime")
    ap.add_argument("--lat_col", default="lat")
    ap.add_argument("--lon_col", default="lon")
    args = ap.parse_args()

    df = pd.read_csv(args.modis_csv)
    bundle = joblib.load(args.model)

    model = bundle["model"]
    features = bundle["features"]
    fill = bundle.get("fill_values", {})

    # Ensure features exist
    for c in features:
        if c not in df.columns:
            df[c] = fill.get(c, 0.0)

    X = df[features].fillna(0.0)
    df["hab_prob"] = model.predict_proba(X)[:, 1]

    df[[args.lat_col, args.lon_col, args.time_col, "hab_prob"]].to_csv(
        args.out_csv, index=False
    )

    print(f"✅ wrote HAB field: {args.out_csv} ({len(df)} points)")

if __name__ == "__main__":
    main()
