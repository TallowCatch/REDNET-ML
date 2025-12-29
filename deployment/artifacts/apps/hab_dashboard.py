#!/usr/bin/env python3
"""
Streamlit HAB Alert Dashboard (Phase 1)

Shows:
- AOI polygon
- Desalination plant markers
- Scene points colored by scene_prob for selected date (time slider)
- Sidebar controls for thresholds and radius

Run:
  pip install streamlit pydeck pandas
  streamlit run deployment/apps/hab_dashboard.py

Inputs (default paths):
  deployment/artifacts/aoi.geojson
  deployment/artifacts/plants.csv
  deployment/outputs/scene_alerts.csv   (must contain scene_lat/scene_lon to map)
"""
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

TS_RE = re.compile(r'_(\d{8})T\d{6}_R')

def infer_scene_date(scene_id: str):
    m = TS_RE.search(scene_id)
    if not m:
        return None
    s = m.group(1)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"

def load_geojson(path: Path):
    return json.loads(path.read_text())

def color_from_prob(p: float):
    # Simple blue->red ramp without specifying "style"; returns [r,g,b,alpha]
    # Clamp 0..1
    x = max(0.0, min(1.0, float(p)))
    r = int(255 * x)
    g = int(80 * (1-x))
    b = int(255 * (1-x))
    return [r, g, b, 160]

st.set_page_config(page_title="HAB Scene Alerts", layout="wide")

# Paths
aoi_path = Path(st.sidebar.text_input("AOI GeoJSON", "deployment/artifacts/aoi.geojson"))
plants_path = Path(st.sidebar.text_input("Plants CSV", "deployment/artifacts/plants.csv"))
scenes_path = Path(st.sidebar.text_input("Scene alerts CSV", "deployment/outputs/scene_alerts.csv"))

# Controls
thr_alert = st.sidebar.number_input("Alert threshold", value=0.53277, step=0.01, format="%.5f")
thr_watch_lo = st.sidebar.number_input("Watch band low", value=0.39, step=0.01, format="%.2f")
thr_watch_hi = st.sidebar.number_input("Watch band high", value=0.53, step=0.01, format="%.2f")

st.title("HAB Alerting Dashboard (Phase 1 — 2D)")

if not (aoi_path.exists() and plants_path.exists() and scenes_path.exists()):
    st.error("One or more input files missing. Check sidebar paths.")
    st.stop()

aoi = load_geojson(aoi_path)
plants = pd.read_csv(plants_path)
scenes = pd.read_csv(scenes_path)

# Scene date
if "scene_date" not in scenes.columns:
    scenes["scene_date"] = scenes["scene_id"].astype(str).map(infer_scene_date)

# Validate geometry
has_geo = {"scene_lat","scene_lon"}.issubset(scenes.columns) and scenes["scene_lat"].notna().any()
if not has_geo:
    st.warning("scene_lat/scene_lon not present in scene_alerts.csv yet, so scenes cannot be mapped.\n"
               "Add tile centroids to inference and aggregate to scene_lat/scene_lon, then re-run.")
    # Still show plants + AOI
    scenes_for_map = scenes.iloc[0:0].copy()
else:
    scenes_for_map = scenes.copy()
    scenes_for_map["scene_prob"] = scenes_for_map["scene_prob"].astype(float)
    scenes_for_map["is_alert"] = scenes_for_map["scene_prob"] >= thr_alert
    scenes_for_map["is_watch"] = (scenes_for_map["scene_prob"] >= thr_watch_lo) & (scenes_for_map["scene_prob"] < thr_watch_hi)
    scenes_for_map["color"] = scenes_for_map["scene_prob"].map(color_from_prob)

# Time slider
dates = sorted([d for d in scenes_for_map["scene_date"].dropna().unique()])
if dates:
    dsel = st.slider("Date", min_value=dates[0], max_value=dates[-1], value=dates[-1])
    scenes_day = scenes_for_map[scenes_for_map["scene_date"] == dsel].copy()
else:
    dsel = None
    scenes_day = scenes_for_map.iloc[0:0].copy()

# Map viewport: use AOI bbox center
coords = []
for feat in aoi.get("features", []):
    geom = feat.get("geometry", {})
    if geom.get("type") == "Polygon":
        for ring in geom["coordinates"]:
            coords.extend(ring)
lons = [c[0] for c in coords]
lats = [c[1] for c in coords]
center_lon = float(np.mean(lons)) if lons else float(plants["lon"].mean())
center_lat = float(np.mean(lats)) if lats else float(plants["lat"].mean())

layers = []

# AOI polygon
layers.append(
    pdk.Layer(
        "GeoJsonLayer",
        data=aoi,
        stroked=True,
        filled=False,
        get_line_color=[0, 0, 0, 180],
        line_width_min_pixels=2,
    )
)

# Plants
layers.append(
    pdk.Layer(
        "ScatterplotLayer",
        data=plants,
        get_position=["lon", "lat"],
        get_radius=6000,  # meters
        radius_min_pixels=4,
        pickable=True,
        auto_highlight=True,
        get_fill_color=[0, 120, 0, 180],
    )
)

# Scenes (points)
if has_geo and len(scenes_day):
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=scenes_day,
            get_position=["scene_lon", "scene_lat"],
            get_radius=8000,
            radius_min_pixels=5,
            pickable=True,
            auto_highlight=True,
            get_fill_color="color",
        )
    )

tooltip = {
    "html": """
    <b>{name}</b><br/>
    {plant_id}<br/>
    """,
    "style": {"backgroundColor": "white", "color": "black"}
}

scene_tooltip = {
    "html": """
    <b>Scene</b>: {scene_id}<br/>
    <b>Prob</b>: {scene_prob}<br/>
    <b>Alert</b>: {is_alert}<br/>
    <b>Watch</b>: {is_watch}<br/>
    """,
    "style": {"backgroundColor": "white", "color": "black"}
}

# Render
view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=6, pitch=0)
deck = pdk.Deck(layers=layers, initial_view_state=view, tooltip=scene_tooltip)
st.pydeck_chart(deck, use_container_width=True)

# Tables
c1, c2 = st.columns(2)
with c1:
    st.subheader("Plants in AOI")
    st.dataframe(plants[["plant_id","name","lat","lon"]].head(200), use_container_width=True)
with c2:
    st.subheader("Scenes for selected date")
    if has_geo and dsel:
        show = scenes_day.sort_values("scene_prob", ascending=False).head(100)
        st.dataframe(show[["scene_id","scene_prob","is_alert","is_watch"]], use_container_width=True)
    else:
        st.info("No scene geometry yet, or no dates found.")

