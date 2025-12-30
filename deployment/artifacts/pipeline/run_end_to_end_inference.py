#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import date, timedelta
import subprocess
import joblib
import pandas as pd
import json
import sys


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def month_range(start: date, end: date):
    cur = date(start.year, start.month, 1)
    while cur <= end:
        yield cur
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)


def run(cmd):
    print("\n▶", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="End-to-end monthly HAB inference (Oman)")
    ap.add_argument("--aoi", required=True, help="AOI GeoJSON (WGS84)")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Load model bundle ONCE
    # ----------------------------------------------------------
    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_cols = bundle["features"]
    calibrator = bundle.get("calibrator", None)

    all_months = []

    # ----------------------------------------------------------
    # Loop month-by-month
    # ----------------------------------------------------------
    for m in month_range(start, end):
        tag = f"{m.year}-{m.month:02d}"
        print(f"\n==================== {tag} ====================")

        m_start = m.isoformat()
        if m.month == 12:
            m_end = date(m.year, 12, 31).isoformat()
        else:
            m_end = (date(m.year, m.month + 1, 1) - timedelta(days=1)).isoformat()

        month_dir = out_root / tag
        chips_dir = month_dir / "chips"
        month_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------
        # 1) CHIP SENTINEL-2 (8-day windows)
        # ------------------------------------------------------
        try:
            run([
                "python", "scripts/download/s2_chip_8day.py",
                "--aoi", args.aoi,
                "--start", m_start,
                "--end", m_end,
                "--cloud", "30",
                "--per_window", "1",
                "--size", "640",
                "--stride", "256",
                "--out", str(chips_dir)
            ])
        except subprocess.CalledProcessError:
            print(f"⚠️ Chipping failed for {tag}, skipping")
            continue

        index_csv = chips_dir / "index.csv"
        if not index_csv.exists() or index_csv.stat().st_size <= 100:
            print(f"⚠️ No chips for {tag}, skipping")
            continue

        # ------------------------------------------------------
        # 2) COMPUTE CHIP INDICES (CORRECT USAGE)
        # ------------------------------------------------------
        try:
            run([
                "python", "scripts/download/s2_compute_chip_indices.py",
                "--folder", str(chips_dir)
            ])
        except subprocess.CalledProcessError:
            print(f"⚠️ Feature computation failed for {tag}, skipping")
            continue

        feat_csv = chips_dir / "chip_indices.csv"
        if not feat_csv.exists():
            print(f"⚠️ chip_indices.csv missing for {tag}, skipping")
            continue

        df = pd.read_csv(feat_csv)
        if df.empty:
            print(f"⚠️ No features for {tag}, skipping")
            continue

        # ------------------------------------------------------
        # 3) RUN FUSION MODEL
        # ------------------------------------------------------
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            print(f"❌ Missing model features in {tag}: {missing}")
            sys.exit(1)

        X = df[feature_cols]
        probs = model.predict_proba(X)[:, 1]

        if calibrator is not None:
            probs = calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]

        df["hab_prob"] = probs
        df["month"] = tag

        out_csv = month_dir / "inference.csv"
        df.to_csv(out_csv, index=False)
        print(f"✅ wrote {out_csv}")

        all_months.append(df)

    # ----------------------------------------------------------
    # Merge all months
    # ----------------------------------------------------------
    if not all_months:
        raise SystemExit("❌ No inference results produced")

    merged = pd.concat(all_months, ignore_index=True)
    merged_csv = out_root / "inference_all_months.csv"
    merged.to_csv(merged_csv, index=False)
    print(f"\n✅ wrote {merged_csv}")

    # ----------------------------------------------------------
    # Build Kepler.gl GeoJSON (TIME SLIDER)
    # ----------------------------------------------------------
    features = []
    for r in merged.itertuples(index=False):
        poly = [
            [r.xmin, r.ymin],
            [r.xmin, r.ymax],
            [r.xmax, r.ymax],
            [r.xmax, r.ymin],
            [r.xmin, r.ymin],
        ]
        features.append({
            "type": "Feature",
            "properties": {
                "scene_id": r.scene_id,
                "time": r.datetime,
                "month": r.month,
                "hab_prob": float(r.hab_prob),
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [poly],
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    out_gj = out_root / "hab_timeslider_all_months.geojson"
    out_gj.write_text(json.dumps(geojson))
    print(f"✅ wrote {out_gj}")


if __name__ == "__main__":
    main()
