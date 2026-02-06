#!/usr/bin/env python3
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df.get("datetime"), errors="coerce", utc=True)
    df["month_num"] = dt.dt.month
    ang = 2 * np.pi * (df["month_num"].astype(float) / 12.0)
    df["month_sin"] = np.sin(ang)
    df["month_cos"] = np.cos(ang)
    return df

def harmonize_modis_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Kd_490" in df.columns and "kd490" not in df.columns:
        df = df.rename(columns={"Kd_490": "kd490"})
    if "nflh" not in df.columns and "flh" in df.columns:
        df["nflh"] = df["flh"]
    if "flh" not in df.columns and "nflh" in df.columns:
        df["flh"] = df["nflh"]
    return df

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["kd490","chlor_a","nflh","sst","fai_mean","ndwi_mean","rednir_mean"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "kd490" in df.columns:
        df["log_kd490"] = np.log(np.clip(df["kd490"].to_numpy(float), 1e-9, None))
    if "chlor_a" in df.columns:
        df["log_chlor_a"] = np.log(np.clip(df["chlor_a"].to_numpy(float), 1e-9, None))
    if "nflh" in df.columns:
        df["log_nflh"] = np.log(np.clip(df["nflh"].to_numpy(float), 1e-9, None))

    if "chlor_a" in df.columns and "kd490" in df.columns:
        df["ratio_chl_kd"] = df["chlor_a"] / np.clip(df["kd490"], 1e-9, None)
    if "chlor_a" in df.columns and "nflh" in df.columns:
        df["chl_times_nflh"] = df["chlor_a"] * df["nflh"]
    if "nflh" in df.columns and "kd490" in df.columns:
        df["ratio_nflh_kd"] = df["nflh"] / np.clip(df["kd490"], 1e-9, None)
    return df

def safe_fill(df: pd.DataFrame, feature_cols: list[str], bundle: dict) -> pd.DataFrame:
    df = df.copy()
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    fills = bundle.get("fill_values") or bundle.get("feature_fill_values")
    if isinstance(fills, dict):
        for c, v in fills.items():
            if c in df.columns:
                df[c] = df[c].fillna(v)

    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--prob_col", default="hab_prob")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feats = bundle["features"]
    cal = bundle.get("calibrator")

    df = harmonize_modis_columns(df)
    df = add_time_features(df)
    df = add_engineered_features(df)
    df = safe_fill(df, feats, bundle)

    probs = model.predict_proba(df[feats])[:, 1]
    if cal is not None:
        probs = cal.predict_proba(probs.reshape(-1,1))[:, 1]

    df[args.prob_col] = probs

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"✓ wrote {args.out_csv} (rows={len(df)})")

if __name__ == "__main__":
    main()
