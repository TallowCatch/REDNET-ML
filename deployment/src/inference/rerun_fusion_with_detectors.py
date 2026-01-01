#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

DETECTOR_COLS = [
    "p_frcnn_r50_med",
    "p_frcnn_mb_med",
    "p_ssd_mb_med",
]

def run_one_month(month_dir: Path, bundle: dict):
    inf_csv  = month_dir / "inference.csv"
    det_csv  = month_dir / "detector_scores.csv"
    idx_csv  = month_dir / "chips" / "chip_indices.csv"

    if not inf_csv.exists() or not det_csv.exists() or not idx_csv.exists():
        raise RuntimeError("missing inference.csv, detector_scores.csv, or chip_indices.csv")

    df  = pd.read_csv(inf_csv)
    det = pd.read_csv(det_csv)
    idx = pd.read_csv(idx_csv)

    # ----------------------------
    # Build PNG stem → hashed chip_id map
    # ----------------------------
    if "tile" not in idx.columns or "chip_id" not in idx.columns:
        raise RuntimeError("chip_indices.csv must contain tile and chip_id columns")

    idx = idx.copy()
    idx["png_stem"] = idx["tile"].astype(str).map(lambda p: Path(p).stem)

    png_to_hashed = dict(zip(idx["png_stem"], idx["chip_id"]))

    # ----------------------------
    # Rewrite detector chip_id
    # ----------------------------
    det = det.copy()
    det["png_stem"] = det["chip_id"]
    det["chip_id"] = det["png_stem"].map(png_to_hashed)

    det = det.dropna(subset=["chip_id"])

    if det.empty:
        raise RuntimeError("no detector rows mapped to inference chip_id")

    # ----------------------------
    # Clean inference + merge
    # ----------------------------
    df = df.drop(
        columns=[c for c in df.columns if c.startswith("p_frcnn_") or c.startswith("p_ssd_")],
        errors="ignore",
    )

    df = df.merge(
        det[["chip_id"] + DETECTOR_COLS],
        on="chip_id",
        how="left",
    )

    for c in DETECTOR_COLS:
        df[c] = df[c].fillna(0.0)

    # 🚨 HARD ASSERT
    if df[DETECTOR_COLS].sum().sum() == 0:
        raise RuntimeError("detector merge failed — all zeros")

    # ----------------------------
    # Fusion inference
    # ----------------------------
    model = bundle["model"]
    feats = bundle["features"]
    calibrator = bundle.get("calibrator")

    for c in feats:
        if c not in df.columns:
            df[c] = 0.0

    X = df[feats]
    probs = model.predict_proba(X)[:, 1]

    if calibrator is not None:
        probs = calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]

    df["hab_prob"] = probs
    df.to_csv(inf_csv, index=False)

    print(f"✓ fused → {month_dir.name} ({len(df)} rows)")


def main():
    ap = argparse.ArgumentParser("Re-run fusion inference with detector scores")
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    root = Path(args.root_dir)
    bundle = joblib.load(args.model)

    months = sorted(p for p in root.iterdir() if p.is_dir())
    print(f"\nFound {len(months)} months\n")

    for m in months:
        try:
            run_one_month(m, bundle)
        except Exception as e:
            print(f"✗ {m.name}: {e}")

    print("\n✅ Fusion complete")


if __name__ == "__main__":
    main()
