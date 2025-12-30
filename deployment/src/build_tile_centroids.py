#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

# Planetary Computer STAC API (Sentinel-2 L2A lives here)
STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
COLLECTION = "sentinel-2-l2a"

SCENE_RE = re.compile(
    r'^(S2[ABC])_MSIL2A_(\d{8}T\d{6})_R(\d{3})_T([0-9A-Z]{2}[0-9A-Z]{3})'
)

def parse_scene(scene_id: str):
    """
    Returns (platform, sensing_dt_utc, rel_orbit, mgrs_tile)
    Example id:
      S2A_MSIL2A_20220928T065651_R063_T39QXG_20240730T192859
    """
    m = SCENE_RE.match(scene_id)
    if not m:
        return None
    platform = m.group(1)               # S2A/S2B/S2C
    sensing = m.group(2)                # YYYYMMDDThhmmss
    rel_orbit = m.group(3)              # R063
    mgrs = m.group(4)                   # 39QXG (already without leading 'T')
    dt = datetime.strptime(sensing, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    return platform, dt, rel_orbit, mgrs

def geom_centroid_lonlat(geom):
    """
    geom is GeoJSON geometry. We'll compute centroid from bbox as a robust fallback.
    """
    # If STAC provides bbox (it does), use that.
    # bbox is [minLon, minLat, maxLon, maxLat]
    return None

def fetch_best_item(platform, dt, mgrs, rel_orbit=None, minutes_window=30):
    """
    Query STAC by datetime window + MGRS tile + platform.
    Then choose the item with id that best matches prefix.
    """
    start = (dt - timedelta(minutes=minutes_window)).isoformat().replace("+00:00", "Z")
    end   = (dt + timedelta(minutes=minutes_window)).isoformat().replace("+00:00", "Z")

    # STAC search payload
    payload = {
        "collections": [COLLECTION],
        "datetime": f"{start}/{end}",
        "limit": 100,
        "query": {
            # many catalogs use "platform" = "sentinel-2a" etc; PC often exposes both "platform" and "sat:platform_international_designator"
            # We'll rely on mgrs tile first; platform filter is a nice-to-have, not required.
            "mgrs:tile": {"eq": mgrs},
        }
    }

    r = requests.post(STAC_SEARCH, json=payload, timeout=60)
    r.raise_for_status()
    feats = r.json().get("features", [])
    if not feats:
        return None

    # Prefer items whose id starts with the stable prefix:
    # S2A_MSIL2A_YYYYMMDDThhmmss_R063_T39QXG
    prefix = f"{platform}_MSIL2A_{dt.strftime('%Y%m%dT%H%M%S')}_R{rel_orbit}_T{mgrs}"

    # Score candidates
    best = None
    best_score = -1
    for it in feats:
        iid = it.get("id", "")
        score = 0
        if iid.startswith(prefix):
            score += 10
        # closer time gets higher score
        props = it.get("properties", {})
        tstr = props.get("datetime") or props.get("start_datetime") or ""
        try:
            tdt = datetime.fromisoformat(tstr.replace("Z", "+00:00"))
            dt_diff = abs((tdt - dt).total_seconds())
            score += max(0, 5 - dt_diff / 600)  # within 50 min gives something
        except Exception:
            pass
        # platform hint
        plat = (props.get("platform") or "").lower()
        if platform.lower() in plat:
            score += 1

        if score > best_score:
            best = it
            best_score = score

    return best

def centroid_from_item(item):
    # Prefer bbox
    bbox = item.get("bbox", None)
    if bbox and len(bbox) == 4:
        minx, miny, maxx, maxy = bbox
        lon = (minx + maxx) / 2.0
        lat = (miny + maxy) / 2.0
        return lat, lon

    # Fallback: geometry coordinates bbox-like calc (rarely needed)
    geom = item.get("geometry", None)
    if not geom:
        return None
    coords = []

    def walk(x):
        if isinstance(x, (list, tuple)) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x):
            coords.append(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                walk(y)

    walk(geom.get("coordinates", []))
    if not coords:
        return None
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (min(lats) + max(lats)) / 2.0, (min(lons) + max(lons)) / 2.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.infer_csv)
    if "scene_id" not in df.columns:
        raise SystemExit("infer_csv must have scene_id")
    if "tile" not in df.columns:
        raise SystemExit("infer_csv must have tile")

    # unique tiles
    tiles = df[["tile", "scene_id"]].drop_duplicates().reset_index(drop=True)

    out_rows = []
    missing = 0

    for _, row in tiles.iterrows():
        tile = str(row["tile"])
        scene_id = str(row["scene_id"])

        parsed = parse_scene(scene_id)
        if not parsed:
            missing += 1
            out_rows.append((tile, scene_id, None, None))
            continue

        platform, dt, rel_orbit, mgrs = parsed
        item = fetch_best_item(platform, dt, mgrs, rel_orbit=rel_orbit, minutes_window=60)
        if not item:
            missing += 1
            out_rows.append((tile, scene_id, None, None))
            continue

        cen = centroid_from_item(item)
        if not cen:
            missing += 1
            out_rows.append((tile, scene_id, None, None))
            continue

        lat, lon = cen
        out_rows.append((tile, scene_id, lat, lon))

    out = pd.DataFrame(out_rows, columns=["tile", "scene_id", "tile_lat", "tile_lon"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f"✅ wrote {args.out_csv} rows={len(out)} missing_centroids={missing}")
    if missing > 0:
        print("⚠️ Some scenes still missing. If it’s all missing, your machine has no internet access to STAC, or the STAC endpoint is blocked.")

if __name__ == "__main__":
    main()
