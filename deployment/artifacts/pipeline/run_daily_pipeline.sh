#!/usr/bin/env bash
set -euo pipefail

# Example "daily" pipeline runner (no Docker).
# You can call this from cron (Linux) or launchd (macOS) later.

# 1) Update L3 features (your existing script)
#    (keep your OBPG_APPKEY env var)
# bash scripts/run_append_l3_8d.sh

# 2) Inference on new/merged CSV
# python deployment/src/batch_infer_csv.py --in_csv <new_merged_csv> --out_csv deployment/outputs/daily_infer.csv

# 3) Scene aggregation
# python deployment/src/hab_scene_audit.py --tile_pred_csv deployment/outputs/daily_infer.csv --make_scene_root --threshold_from deployment/artifacts/thresholds.json --agg max --out_csv deployment/outputs/daily_scene_alerts.csv

# 4) Plant extraction (weekly/monthly is enough)
# python scripts/osm_extract_desal_plants.py --aoi_geojson deployment/artifacts/aoi.geojson --out_csv deployment/artifacts/plants.csv

# 5) Proximity join (requires scene_lat/scene_lon columns to be meaningful)
# python deployment/src/scene_proximity_join.py --scene_csv deployment/outputs/daily_scene_alerts.csv --plants_csv deployment/artifacts/plants.csv --out_scene_csv deployment/outputs/daily_scene_alerts_with_proximity.csv --out_events_csv deployment/outputs/daily_plant_risk_events.csv

# 6) Dashboard is separate (Streamlit)
# streamlit run deployment/apps/hab_dashboard.py
echo "✅ pipeline skeleton; edit paths + enable the steps you want."
