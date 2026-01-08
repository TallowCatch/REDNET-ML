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

def meters_to_latlon(lat, dx, dy):
    dlat = dy / 111_000
    dlon = dx / (111_000 * math.cos(math.radians(lat)))
    return dlat, dlon

# ─────────────────────────────────────────────────────────────
# offshore sampler (CORRECT)
# ─────────────────────────────────────────────────────────────
def offshore_point_east(
    plant_lat,
    plant_lon,
    min_forward_m,
    max_forward_m,
    lateral_m,
    rng
):
    """
    Generate a point strictly offshore (east of plant).
    """
    dx = rng.uniform(min_forward_m, max_forward_m)   # EAST only
    dy = rng.normal(0, lateral_m)                    # north/south spread

    dlat, dlon = meters_to_latlon(plant_lat, dx, dy)

    lat = plant_lat + dlat
    lon = plant_lon + dlon

    # hard safety: never allow landward points
    if lon <= plant_lon:
        lon = plant_lon + abs(dlon) + 0.01

    return lat, lon

# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser("HAB offshore cloud (correct)")
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--plant_geojson", required=True)
    ap.add_argument("--out_geojson", required=True)

    ap.add_argument("--points_per_month", type=int, default=300)
    ap.add_argument("--min_forward_m", type=float, default=4000)
    ap.add_argument("--max_forward_m", type=float, default=30000)
    ap.add_argument("--lateral_m", type=float, default=8000)

    args = ap.parse_args()

    # ── Load plant (REAL location)
    plant = json.loads(Path(args.plant_geojson).read_text())
    props = plant["features"][0]["properties"]

    plant_lat = float(props["lat"])
    plant_lon = float(props["lon"])

    print(f"🏭 Plant: {props.get('name','')} lat={plant_lat}, lon={plant_lon}")
    print(f"➡️  Offshore enforced EAST (lon > plant lon)")

    # ── Load inference
    df = pd.read_csv(args.in_csv)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    features = []

    for row in df.itertuples(index=False):
        prob = float(row.hab_prob)
        if not np.isfinite(prob):
            continue

        rng = stable_rng(f"{row.month}_{prob:.3f}")

        n_pts = max(20, int(args.points_per_month * prob))

        for _ in range(n_pts):
            lat, lon = offshore_point_east(
                plant_lat,
                plant_lon,
                min_forward_m=args.min_forward_m,
                max_forward_m=args.max_forward_m,
                lateral_m=args.lateral_m,
                rng=rng,
            )

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "time": row.datetime.isoformat(),
                    "hab_prob": prob,
                    "month": str(row.month),
                }
            })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    Path(args.out_geojson).write_text(json.dumps(geojson, indent=2))

    print("✅ HAB cloud correctly placed offshore")
    print(f"   → {args.out_geojson}")
    print(f"   → features: {len(features)}")

if __name__ == "__main__":
    main()
