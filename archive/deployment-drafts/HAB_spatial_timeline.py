#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────
def stable_rng(key: str):
    h = hashlib.sha1(key.encode()).digest()
    seed = int.from_bytes(h[:4], "little")
    return np.random.default_rng(seed)

def jitter_latlon(lat, lon, meters, rng):
    # very small-area approximation (OK for AOI scale)
    dlat = meters / 111_000
    dlon = meters / (111_000 * math.cos(math.radians(lat)))
    return (
        lat + rng.uniform(-dlat, dlat),
        lon + rng.uniform(-dlon, dlon),
    )

# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser("HAB spatial cloud + plant risk pipeline")
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--plant_lat", type=float, required=True)
    ap.add_argument("--plant_lon", type=float, required=True)
    ap.add_argument("--out_geojson", required=True)

    ap.add_argument("--points_per_month", type=int, default=200)
    ap.add_argument("--base_radius_m", type=float, default=8000)
    ap.add_argument("--max_radius_m", type=float, default=25000)

    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    feats = []

    for row in df.itertuples(index=False):
        prob = float(row.hab_prob)
        if np.isnan(prob):
            continue

        # scale cloud size with risk
        n_pts = max(5, int(args.points_per_month * prob))
        radius = args.base_radius_m + prob * (args.max_radius_m - args.base_radius_m)

        rng = stable_rng(f"{row.month}_{prob:.3f}")

        for _ in range(n_pts):
            lat, lon = jitter_latlon(
                args.plant_lat,
                args.plant_lon,
                meters=rng.uniform(0, radius),
                rng=rng,
            )

            feats.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "time": row.datetime.isoformat(),
                    "hab_prob": prob,
                    "month": row.month,
                    "chlor_a": getattr(row, "chlor_a", None),
                }
            })

    geojson = {"type": "FeatureCollection", "features": feats}
    Path(args.out_geojson).write_text(json.dumps(geojson))

    print(f"✅ wrote {args.out_geojson}")
    print(f"   features={len(feats)}")

if __name__ == "__main__":
    main()
