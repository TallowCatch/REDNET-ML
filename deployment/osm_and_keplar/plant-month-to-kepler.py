#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser("Make Kepler time-slider GeoJSON from plant_month_risk.csv")
    ap.add_argument("--risk_csv", required=True, help="plant_month_risk.csv")
    ap.add_argument("--plant_lat", type=float, required=True)
    ap.add_argument("--plant_lon", type=float, required=True)
    ap.add_argument("--out_geojson", required=True)
    ap.add_argument("--time_col", default="month")     # 'YYYY-MM'
    args = ap.parse_args()

    df = pd.read_csv(args.risk_csv)

    if args.time_col not in df.columns:
        raise SystemExit(f"Missing time_col={args.time_col}")

    # convert 'YYYY-MM' -> ISO datetime (Kepler likes timestamps)
    # use first day of month at UTC
    time_iso = pd.to_datetime(df[args.time_col].astype(str) + "-01", errors="coerce", utc=True)

    feats = []
    for r, t in zip(df.itertuples(index=False), time_iso):
        if pd.isna(t):
            continue

        props = {c: getattr(r, c) for c in df.columns}
        props["time"] = t.isoformat()

        feats.append({
            "type": "Feature",
            "properties": props,
            "geometry": {"type": "Point", "coordinates": [args.plant_lon, args.plant_lat]},
        })

    out = {"type": "FeatureCollection", "features": feats}
    Path(args.out_geojson).write_text(json.dumps(out))
    print(f"✅ wrote {args.out_geojson} features={len(feats)}")

if __name__ == "__main__":
    main()
