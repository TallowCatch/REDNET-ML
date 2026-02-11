#!/usr/bin/env python3
"""
Join scene alerts with desalination plant proximity.

Inputs:
- scene_alerts.csv (from hab_scene_audit.py) must include: scene_id, scene_prob (and ideally scene_date)
- plants.csv from OSM script must include: plant_id, name, lat, lon

Outputs:
- scene_alerts_with_proximity.csv (each scene matched to nearest plant)
- plant_risk_events.csv (only scenes within radius_km and over a chosen threshold band)

Usage:
  python deployment/src/scene_proximity_join.py \
    --scene_csv deployment/outputs/scene_alerts.csv \
    --plants_csv deployment/artifacts/plants.csv \
    --radius_km 25 \
    --watch_band_low 0.39 --watch_band_high 0.53 \
    --out_scene_csv deployment/outputs/scene_alerts_with_proximity.csv \
    --out_events_csv deployment/outputs/plant_risk_events.csv
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import re
from math import radians, sin, cos, asin, sqrt

def haversine_km(lat1, lon1, lat2, lon2):
    # great-circle distance
    R = 6371.0
    dlat = radians(lat2-lat1)
    dlon = radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*asin(sqrt(a))

TS_RE = re.compile(r'_(\d{8})T\d{6}_R')  # grabs YYYYMMDD from scene_root like ..._20220928T065651_R063_...

def infer_scene_date(scene_id: str):
    m = TS_RE.search(scene_id)
    if not m:
        return None
    s = m.group(1)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_csv", required=True)
    ap.add_argument("--plants_csv", required=True)
    ap.add_argument("--radius_km", type=float, default=25.0)
    ap.add_argument("--watch_band_low", type=float, default=0.39)
    ap.add_argument("--watch_band_high", type=float, default=0.53)
    ap.add_argument("--out_scene_csv", required=True)
    ap.add_argument("--out_events_csv", required=True)
    args = ap.parse_args()

    scenes = pd.read_csv(args.scene_csv)
    plants = pd.read_csv(args.plants_csv)

    # You need scene geometry (lat/lon) for true proximity.
    # If you don't have it yet, we can only output placeholders.
    if not {"scene_lat","scene_lon"}.issubset(scenes.columns):
        scenes["scene_lat"] = np.nan
        scenes["scene_lon"] = np.nan
        scenes["nearest_plant_id"] = ""
        scenes["nearest_plant_name"] = ""
        scenes["nearest_plant_km"] = np.nan
        scenes["scene_date"] = scenes.get("scene_date") if "scene_date" in scenes.columns else scenes["scene_id"].map(infer_scene_date)
        Path(args.out_scene_csv).parent.mkdir(parents=True, exist_ok=True)
        scenes.to_csv(args.out_scene_csv, index=False)
        # Events can't be computed without geometry
        Path(args.out_events_csv).parent.mkdir(parents=True, exist_ok=True)
        scenes.iloc[0:0].to_csv(args.out_events_csv, index=False)
        print("⚠️  scene_lat/scene_lon not found in scene_csv, wrote placeholders.")
        print("   Fix by adding tile centroids to inference, then aggregating to scene_lat/scene_lon.")
        return

    # Compute nearest plant per scene
    plant_lat = plants["lat"].astype(float).to_numpy()
    plant_lon = plants["lon"].astype(float).to_numpy()
    plant_id = plants["plant_id"].astype(str).to_numpy()
    plant_nm = plants.get("name", pd.Series([""]*len(plants))).astype(str).to_numpy()

    nearest_ids = []
    nearest_names = []
    nearest_km = []

    for _, row in scenes.iterrows():
        lat, lon = float(row["scene_lat"]), float(row["scene_lon"])
        dists = np.array([haversine_km(lat, lon, plat, plon) for plat, plon in zip(plant_lat, plant_lon)])
        j = int(np.argmin(dists))
        nearest_ids.append(plant_id[j])
        nearest_names.append(plant_nm[j])
        nearest_km.append(float(dists[j]))

    scenes["scene_date"] = scenes.get("scene_date") if "scene_date" in scenes.columns else scenes["scene_id"].map(infer_scene_date)
    scenes["nearest_plant_id"] = nearest_ids
    scenes["nearest_plant_name"] = nearest_names
    scenes["nearest_plant_km"] = nearest_km
    scenes["within_radius"] = scenes["nearest_plant_km"] <= args.radius_km

    out_scene = Path(args.out_scene_csv)
    out_scene.parent.mkdir(parents=True, exist_ok=True)
    scenes.to_csv(out_scene, index=False)

    # Watchlist-like events: within radius AND in the watch band OR alerted
    prob = scenes["scene_prob"].astype(float)
    watch = (prob >= args.watch_band_low) & (prob < args.watch_band_high)
    alerted = scenes.get("scene_alert", pd.Series([0]*len(scenes))).astype(int) == 1
    events = scenes[scenes["within_radius"] & (watch | alerted)].copy()

    out_events = Path(args.out_events_csv)
    out_events.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_events, index=False)

    print(f"✅ wrote: {out_scene} (rows={len(scenes)})")
    print(f"✅ wrote: {out_events} (events={len(events)})")

if __name__ == "__main__":
    main()
