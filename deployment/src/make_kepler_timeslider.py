#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Make Kepler.gl time-slider GeoJSON from an inference table")
    ap.add_argument("--in_csv", required=True, help="CSV like deployment/outputs/inference_with_probs.csv")
    ap.add_argument("--out_geojson", required=True, help="Output GeoJSON for Kepler time slider")
    ap.add_argument("--prob_col", default="hab_prob", help="Probability column name")
    ap.add_argument("--time_col", default="datetime", help="Datetime column name (ISO preferred)")
    ap.add_argument("--lat_col", default="tile_lat", help="Latitude column (WGS84)")
    ap.add_argument("--lon_col", default="tile_lon", help="Longitude column (WGS84)")
    ap.add_argument("--id_col", default="tile", help="Tile id column")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # tolerate common variations
    if args.prob_col not in df.columns:
        # sometimes named p_hab / prob / pred / etc
        for cand in ["prob", "p_hab", "pred", "hab_probability"]:
            if cand in df.columns:
                args.prob_col = cand
                break

    missing = [c for c in [args.lat_col, args.lon_col, args.time_col, args.prob_col] if c not in df.columns]
    if missing:
        raise SystemExit(
            "❌ Your table is missing required columns for mapping/time slider:\n"
            f"   missing={missing}\n\n"
            "Fix: ensure your table has tile_lat/tile_lon in WGS84 + datetime + hab_prob.\n"
            "If you only have scene-level coords, you can map scene centroids instead."
        )

    # Build GeoJSON points (fast + perfect for Kepler time slider)
    feats = []
    for r in df.itertuples(index=False):
        lat = getattr(r, args.lat_col)
        lon = getattr(r, args.lon_col)
        if pd.isna(lat) or pd.isna(lon):
            continue

        props = {
            "id": getattr(r, args.id_col) if args.id_col in df.columns else None,
            "scene_id": getattr(r, "scene_id") if "scene_id" in df.columns else None,
            "time": getattr(r, args.time_col),
            "hab_prob": float(getattr(r, args.prob_col)),
        }

        feats.append({
            "type": "Feature",
            "properties": props,
            "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]}
        })

    out = {"type": "FeatureCollection", "features": feats}
    Path(args.out_geojson).write_text(json.dumps(out))
    print(f"✅ wrote {args.out_geojson} features={len(feats)}")

if __name__ == "__main__":
    main()
