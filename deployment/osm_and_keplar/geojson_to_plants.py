#!/usr/bin/env python3
"""
Convert one or more GeoJSON files (downloaded from Overpass/OSM) into plants.csv.

Example:
  python geojson_to_plants_csv.py \
    --aoi_geojson deployment/artifacts/aoi.geojson \
    --geojson deployment/artifacts/plants_query1.geojson \
    --geojson deployment/artifacts/plants_query2.geojson \
    --out_csv deployment/artifacts/plants.csv \
    --dedupe_radius_m 300

Notes:
- No API calls. Offline conversion only.
- Extracts representative lat/lon from:
    * Point geometry
    * properties.center or geometry.center (if present)
    * centroid of LineString/Polygon/Multi* (best-effort)
- Writes schema compatible with your pipeline:
    plant_id,name,lat,lon,country,capacity_m3_day,osm_type,osm_id,operator,tags_json
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ----------------------------
# Geometry helpers
# ----------------------------

def aoi_bbox(aoi_geojson_path: Path) -> Tuple[float, float, float, float]:
    gj = json.loads(aoi_geojson_path.read_text(encoding="utf-8"))
    coords = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry", {}) or {}
        gtype = geom.get("type")
        if gtype == "Polygon":
            for ring in geom.get("coordinates", []):
                coords.extend(ring)
        elif gtype == "MultiPolygon":
            for poly in geom.get("coordinates", []):
                for ring in poly:
                    coords.extend(ring)
    if not coords:
        raise ValueError("AOI geojson had no polygon coordinates")
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    south, west, north, east = min(lats), min(lons), max(lats), max(lons)
    return south, west, north, east

def within_bbox(lat: float, lon: float, bbox: Tuple[float, float, float, float]) -> bool:
    south, west, north, east = bbox
    return (south <= lat <= north) and (west <= lon <= east)

def flatten_coords(coords: Any) -> Iterable[Tuple[float, float]]:
    """
    Yield (lon, lat) pairs from nested GeoJSON coordinate structures.
    """
    if coords is None:
        return
    # If it looks like a single coordinate pair [lon, lat]
    if isinstance(coords, (list, tuple)) and len(coords) >= 2 and all(isinstance(x, (int, float)) for x in coords[:2]):
        yield (float(coords[0]), float(coords[1]))
        return
    # Otherwise recurse
    if isinstance(coords, (list, tuple)):
        for item in coords:
            yield from flatten_coords(item)

def centroid_of_coords(lonlat: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not lonlat:
        return None
    # Simple average centroid (good enough for our purpose)
    lon = sum(p[0] for p in lonlat) / len(lonlat)
    lat = sum(p[1] for p in lonlat) / len(lonlat)
    return (lat, lon)

def get_feature_point(feature: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    Return (lat, lon) for a feature if possible.
    Tries:
      - geometry Point
      - properties.center / geometry.center (from Overpass "out center")
      - centroid from other geometries
    """
    geom = feature.get("geometry") or {}
    props = feature.get("properties") or {}

    gtype = geom.get("type")

    # 1) Point geometry
    if gtype == "Point":
        c = geom.get("coordinates")
        if c and len(c) >= 2:
            lon, lat = float(c[0]), float(c[1])
            return (lat, lon)

    # 2) Overpass-style "center"
    # Sometimes the conversion to GeoJSON stores it in props["center"] = {"lat":..,"lon":..}
    center = props.get("center") or geom.get("center")
    if isinstance(center, dict) and "lat" in center and "lon" in center:
        return (float(center["lat"]), float(center["lon"]))

    # 3) Some exports put lat/lon directly in properties
    for lat_key, lon_key in [("lat", "lon"), ("latitude", "longitude")]:
        if lat_key in props and lon_key in props:
            try:
                return (float(props[lat_key]), float(props[lon_key]))
            except Exception:
                pass

    # 4) Centroid for other geometries
    coords = geom.get("coordinates")
    pts = list(flatten_coords(coords))
    return centroid_of_coords(pts)

# ----------------------------
# Tag extraction helpers
# ----------------------------

NAME_KEYS = ["name", "Name", "title", "plant_name"]
OPERATOR_KEYS = ["operator", "Operator", "owner", "Owner", "company", "Company"]
COUNTRY_KEYS = ["addr:country", "country", "ISO3166-1:alpha2", "ISO3166-1"]
CAPACITY_KEYS = [
    "desalination:capacity",
    "capacity",
    "capacity_m3_day",
    "capacity:m3/day",
    "capacity:volume",
    "plant:output",
    "plant:output:water",
    "water:output",
]

def pick_first(props: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        v = props.get(k)
        if v is None:
            continue
        if isinstance(v, (dict, list)):
            continue
        s = str(v).strip()
        if s:
            return s
    return ""

def normalize_osm_identity(props: Dict[str, Any]) -> Tuple[str, str]:
    """
    Try to find osm_type/osm_id from common exports.
    """
    osm_type = (
        pick_first(props, ["osm_type", "type", "@type"])
        or ""
    ).lower()

    osm_id = pick_first(props, ["osm_id", "id", "@id"])
    # Overpass GeoJSON exports sometimes use "id" as numeric and type is in "type"
    if osm_id and osm_id.startswith(("node/", "way/", "relation/")):
        # e.g. "way/123"
        try:
            t, i = osm_id.split("/", 1)
            return t, i
        except Exception:
            pass

    # If osm_type is weird (like "Feature") ignore it
    if osm_type in {"feature", "features"}:
        osm_type = ""

    # If props has separate "osm_type" and "osm_id", prefer them
    if props.get("osm_type") and props.get("osm_id"):
        return str(props["osm_type"]).lower(), str(props["osm_id"])

    # If not available, leave blank
    return osm_type, osm_id

# ----------------------------
# Dedupe (optional)
# ----------------------------

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def dedupe_rows(rows: List[Dict[str, Any]], radius_m: float) -> List[Dict[str, Any]]:
    """
    Keep the "best" row among near-duplicates (within radius_m).
    Best = more tags, then has name, then has operator.
    """
    kept: List[Dict[str, Any]] = []
    for r in rows:
        lat, lon = float(r["lat"]), float(r["lon"])
        found_idx = None
        for i, k in enumerate(kept):
            d = haversine_m(lat, lon, float(k["lat"]), float(k["lon"]))
            if d <= radius_m:
                found_idx = i
                break

        if found_idx is None:
            kept.append(r)
        else:
            def score(x: Dict[str, Any]) -> int:
                tags = json.loads(x["tags_json"]) if x.get("tags_json") else {}
                s = 0
                s += min(len(tags), 50)  # cap
                if x.get("name"): s += 10
                if x.get("operator"): s += 5
                if x.get("capacity_m3_day"): s += 3
                return s

            if score(r) > score(kept[found_idx]):
                kept[found_idx] = r
    return kept

# ----------------------------
# Main
# ----------------------------

def load_geojson(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def iter_features(gj: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    t = gj.get("type")
    if t == "FeatureCollection":
        yield from (gj.get("features") or [])
    elif t == "Feature":
        yield gj
    else:
        # Some Overpass JSON->GeoJSON exports wrap differently; try best-effort
        feats = gj.get("features")
        if isinstance(feats, list):
            yield from feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi_geojson", required=True, help="AOI polygon GeoJSON (used for bbox filtering)")
    ap.add_argument("--geojson", action="append", required=True, help="Input GeoJSON file (repeatable)")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--debug_json", default=None, help="Optional: write parsed rows as JSON for inspection")
    ap.add_argument("--dedupe_radius_m", type=float, default=300.0, help="Merge near-duplicates within this radius (meters)")
    ap.add_argument("--no_bbox_filter", action="store_true", help="Do not filter to AOI bbox")
    args = ap.parse_args()

    aoi_path = Path(args.aoi_geojson)
    bbox = aoi_bbox(aoi_path)

    rows: List[Dict[str, Any]] = []
    for gj_path_str in args.geojson:
        gj_path = Path(gj_path_str)
        gj = load_geojson(gj_path)
        for feat in iter_features(gj):
            props = feat.get("properties") or {}
            point = get_feature_point(feat)
            if not point:
                continue
            lat, lon = point

            if (not args.no_bbox_filter) and (not within_bbox(lat, lon, bbox)):
                continue

            name = pick_first(props, NAME_KEYS)
            operator = pick_first(props, OPERATOR_KEYS)
            country = pick_first(props, COUNTRY_KEYS)
            capacity = pick_first(props, CAPACITY_KEYS)

            osm_type, osm_id = normalize_osm_identity(props)

            # plant_id: prefer osm_type/osm_id if present
            if osm_type and osm_id:
                plant_id = f"osm:{osm_type}/{osm_id}"
            elif osm_id:
                plant_id = f"osm:unknown/{osm_id}"
            else:
                # fallback deterministic id from coords + name
                plant_id = f"geojson:{round(lat,6)}:{round(lon,6)}:{name[:30]}"

            # if no name, use operator as display label
            display_name = name or operator or ""

            rows.append({
                "plant_id": plant_id,
                "name": display_name,
                "lat": lat,
                "lon": lon,
                "country": country,
                "capacity_m3_day": capacity,
                "osm_type": osm_type,
                "osm_id": osm_id,
                "operator": operator,
                "tags_json": json.dumps(props, ensure_ascii=False),
            })

    # Dedupe
    if args.dedupe_radius_m and args.dedupe_radius_m > 0:
        rows = dedupe_rows(rows, args.dedupe_radius_m)

    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)

    fields = ["plant_id","name","lat","lon","country","capacity_m3_day","osm_type","osm_id","operator","tags_json"]
    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if args.debug_json:
        dbg = Path(args.debug_json)
        dbg.parent.mkdir(parents=True, exist_ok=True)
        dbg.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    south, west, north, east = bbox
    print(f"✅ wrote plants.csv: {outp} (rows={len(rows)})")
    print(f"AOI bbox: south={south:.4f}, west={west:.4f}, north={north:.4f}, east={east:.4f}")
    if args.debug_json:
        print(f"🧪 debug rows JSON: {args.debug_json}")

if __name__ == "__main__":
    main()
