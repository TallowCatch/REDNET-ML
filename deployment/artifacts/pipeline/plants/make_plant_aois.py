#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, re
from pathlib import Path
import pandas as pd

def km_to_deg(lat_deg: float, km: float):
    dlat = km / 110.574
    dlon = km / (111.320 * math.cos(math.radians(lat_deg)) + 1e-9)
    return dlat, dlon

def feature_box(lon, lat, half_km, props):
    dlat, dlon = km_to_deg(lat, half_km)
    xmin, xmax = lon - dlon, lon + dlon
    ymin, ymax = lat - dlat, lat + dlat
    poly = [
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin],
        [xmin, ymin],
    ]
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "Polygon", "coordinates": [poly]},
    }

def safe_id(s: str) -> str:
    # turn "osm:way/449632054" -> "osm_way_449632054"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants_csv", required=True, help="CSV with plant_id,name,lat,lon")
    ap.add_argument("--out_dir", required=True, help="Output folder for per-plant AOI GeoJSONs")
    ap.add_argument("--box_km", type=float, default=20.0, help="BOX WIDTH in km (not radius). e.g. 20km box")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plants = pd.read_csv(args.plants_csv)
    half_km = args.box_km / 2.0

    for r in plants.itertuples(index=False):
        plant_id = str(getattr(r, "plant_id"))
        plant_id_safe = safe_id(plant_id)

        name = str(getattr(r, "name")) if "name" in plants.columns else ""
        lat = float(getattr(r, "lat"))
        lon = float(getattr(r, "lon"))

        feat = feature_box(lon, lat, half_km, {
            "plant_id": plant_id,
            "plant_id_safe": plant_id_safe,
            "name": name,
            "lat": lat,
            "lon": lon,
            "box_km": args.box_km,
        })

        gj = {"type": "FeatureCollection", "features": [feat]}
        (out_dir / f"plant_{plant_id_safe}.geojson").write_text(json.dumps(gj))

    print(f"✅ wrote {len(plants)} AOIs to {out_dir}")

if __name__ == "__main__":
    main()
