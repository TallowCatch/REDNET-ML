#!/usr/bin/env python3
import pandas as pd
import json
from pathlib import Path

IN_CSV = "deployment/outputs/by_plant/osm_way_1236881046/inference_all_months.csv"
OUT_GEOJSON = "/tmp/plant_risk.geojson"

PLANT_LAT = 21.9310725
PLANT_LON = 59.6321193

df = pd.read_csv(IN_CSV)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

features = []

for _, row in df.iterrows():
    dt = row["datetime"]

    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [PLANT_LON, PLANT_LAT]
        },
        "properties": {
            # 🔑 THIS is the only required time field
            "time": int(dt.timestamp() * 1000),

            # 🔑 main signal
            "hab_prob": float(row["hab_prob"]),

            # optional metadata (SAFE)
            "year": int(dt.year),
            "month": int(dt.month),
            "month_str": dt.strftime("%Y-%m"),

            # human-readable risk bucket
            "risk_level": (
                "high" if row["hab_prob"] > 0.7 else
                "medium" if row["hab_prob"] > 0.4 else
                "low"
            )
        }
    })

geojson = {
    "type": "FeatureCollection",
    "features": features
}

Path(OUT_GEOJSON).write_text(json.dumps(geojson))
print(f"✅ wrote {OUT_GEOJSON} ({len(features)} points)")
