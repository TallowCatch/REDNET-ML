#!/usr/bin/env python3
"""
Extract desalination-related plants from OpenStreetMap via Overpass, within an AOI bbox,
writing plants.csv suitable for a proximity layer.

Key idea (because Oman tagging is inconsistent):
- Query 1: "strict tags" (man_made=desalination_plant, water_works desalination, etc.)
- Query 2: "fuzzy text" (name/operator contains desal/RO/تحلية and key Oman plant locations)
- Merge + dedupe + score candidates
- Filter by AOI bbox in Python (keeps only your AOI)
- Output CSV + optional debug JSON for inspection

Usage:
  python osm_extract_desal_plants.py \
    --aoi_geojson deployment/artifacts/aoi.geojson \
    --out_csv deployment/artifacts/plants.csv \
    --debug_json deployment/artifacts/plants_debug.json \
    --min_score 3
"""

import argparse
import csv
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests


# ---------- AOI helpers ----------

def aoi_bbox(aoi_geojson_path: Path) -> Tuple[float, float, float, float]:
    """
    Compute bbox over all Polygon/MultiPolygon coordinates in AOI GeoJSON.

    Returns (south, west, north, east) == (min_lat, min_lon, max_lat, max_lon)
    """
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
        raise ValueError("AOI geojson had no polygon coordinates (Polygon/MultiPolygon).")

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (min(lats), min(lons), max(lats), max(lons))


def in_bbox(lat: float, lon: float, south: float, west: float, north: float, east: float) -> bool:
    return (south <= lat <= north) and (west <= lon <= east)


# ---------- Overpass plumbing ----------

DEFAULT_ENDPOINTS = [
    # Good general endpoints (any of these can be busy sometimes)
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

# Treat these codes as "try another endpoint"
RETRYABLE_HTTP = {429, 502, 503, 504}
SKIP_ENDPOINT_HTTP = {401, 403}  # often blocked by geo/rate-limit rules


def post_overpass(query: str, endpoints: List[str], max_tries_per_endpoint: int = 2) -> Dict:
    """
    Try multiple Overpass endpoints until one returns 200 with JSON.
    Skips endpoints that return 401/403. Retries rate-limit/busy errors.
    """
    headers = {"Content-Type": "text/plain; charset=utf-8"}

    last_err = None

    # Shuffle endpoints so we don't always hit the same one first.
    eps = endpoints[:]
    random.shuffle(eps)

    for ep in eps:
        for attempt in range(1, max_tries_per_endpoint + 1):
            try:
                resp = requests.post(ep, data=query.encode("utf-8"), headers=headers, timeout=240)
            except requests.RequestException as e:
                last_err = f"Request error from {ep}: {e}"
                # backoff then retry
                time.sleep(1.5 * attempt)
                continue

            if resp.status_code == 200:
                # Sometimes Overpass returns HTML on 200 if upstream is weird; guard it.
                try:
                    return resp.json()
                except Exception:
                    last_err = f"Non-JSON 200 response from {ep}: {resp.text[:500]}"
                    time.sleep(1.5 * attempt)
                    continue

            if resp.status_code in SKIP_ENDPOINT_HTTP:
                last_err = f"HTTP {resp.status_code} from {ep}: {resp.text[:400]}"
                # Don't retry this endpoint — move on
                break

            if resp.status_code in RETRYABLE_HTTP:
                last_err = f"HTTP {resp.status_code} from {ep}: {resp.text[:400]}"
                # Backoff then retry
                time.sleep(2.0 * attempt)
                continue

            # Other non-200: likely query or server trouble; try next endpoint
            last_err = f"HTTP {resp.status_code} from {ep}: {resp.text[:400]}"
            break

    raise RuntimeError(last_err or "Overpass failed for unknown reasons.")


# ---------- Queries ----------

OMAN_AREA_ID = 3600305138  # area id for relation 305138 (Oman)

# Regexes that work well in Oman data:
FUZZY_DESAL_RE = r"desal|desalin|reverse osmosis|\bRO\b|تحلية|محطة تحلية|تحليه"
# Optional Oman-known locations that tend to appear in plant names/operators.
OMAN_LOC_RE = r"Barka|Sohar|Sur|Ghubrah|Qurayyat|Qurayyat|Muscat|Seeb|Salalah|Duqm|Khasab|Shinas"


def build_query_strict(area_id: int, timeout_s: int = 180) -> str:
    """
    Strict tags query inside Oman admin area.
    This avoids bbox heaviness, and we bbox-filter later in Python.
    """
    return f"""
[out:json][timeout:{timeout_s}];
area({area_id})->.om;
(
  nwr["man_made"="desalination_plant"](area.om);
  nwr["man_made"="water_works"]["water_works"="desalination"](area.om);
  nwr["water"="desalination"](area.om);

  nwr["man_made"="works"]["industrial"="water"](area.om);
  nwr["industrial"="water_treatment"](area.om);
  nwr["industrial"="water"](area.om);

  nwr["man_made"="works"]["product"~"water",i](area.om);
);
out center tags;
"""


def build_query_fuzzy(area_id: int, timeout_s: int = 180) -> str:
    """
    Fuzzy query inside Oman admin area:
    - name/operator contain desalination hints (English + Arabic)
    - plus a second clause that catches items named after common Oman plant locations
      (useful when the word "desal" isn't present)
    """
    return f"""
[out:json][timeout:{timeout_s}];
area({area_id})->.om;
(
  nwr["name"~"{FUZZY_DESAL_RE}", i](area.om);
  nwr["operator"~"{FUZZY_DESAL_RE}", i](area.om);
  nwr["description"~"{FUZZY_DESAL_RE}", i](area.om);

  nwr["name"~"{OMAN_LOC_RE}", i](area.om);
);
out center tags;
"""


# ---------- Scoring / filtering ----------

CAPACITY_KEYS = [
    "desalination:capacity",
    "plant:output:water",
    "plant:output",
    "output",
    "capacity",
    "design_capacity",
    "capacity:water",
]

@dataclass
class Candidate:
    osm_type: str
    osm_id: int
    lat: float
    lon: float
    tags: Dict
    score: int
    reasons: List[str]


def get_lat_lon(e: Dict) -> Optional[Tuple[float, float]]:
    if "lat" in e and "lon" in e:
        return e["lat"], e["lon"]
    if "center" in e and isinstance(e["center"], dict):
        c = e["center"]
        if "lat" in c and "lon" in c:
            return c["lat"], c["lon"]
    return None


def normalize_text(s: str) -> str:
    return (s or "").strip()


def score_candidate(tags: Dict) -> Tuple[int, List[str]]:
    """
    Add points for strong evidence it is a desalination plant.
    Tune as needed, but these weights work well:
    - strict tags give big points
    - fuzzy text gives smaller points
    """
    score = 0
    reasons = []

    mm = tags.get("man_made", "")
    water_works = tags.get("water_works", "")
    water = tags.get("water", "")
    industrial = tags.get("industrial", "")

    name = normalize_text(tags.get("name", ""))
    operator = normalize_text(tags.get("operator", ""))
    desc = normalize_text(tags.get("description", ""))

    # Strong tagging signals
    if mm == "desalination_plant":
        score += 5
        reasons.append("man_made=desalination_plant (+5)")

    if mm == "water_works" and water_works == "desalination":
        score += 5
        reasons.append("man_made=water_works + water_works=desalination (+5)")

    if water == "desalination":
        score += 4
        reasons.append("water=desalination (+4)")

    # Weaker/indirect
    if mm == "works" and industrial == "water":
        score += 2
        reasons.append("man_made=works + industrial=water (+2)")

    if industrial in {"water", "water_treatment"}:
        score += 2
        reasons.append(f"industrial={industrial} (+2)")

    # Text evidence
    txt = " | ".join([name, operator, desc])
    if re.search(FUZZY_DESAL_RE, txt, flags=re.IGNORECASE):
        score += 2
        reasons.append("text matches desal/RO/تحلية (+2)")

    # Oman location names give *weak* evidence (good for recall, not precision)
    if re.search(OMAN_LOC_RE, name, flags=re.IGNORECASE):
        score += 1
        reasons.append("name matches Oman location list (+1)")

    # If it explicitly says plant/facility in description/name, small boost
    if re.search(r"\bplant\b|\bstation\b|محطة", txt, flags=re.IGNORECASE):
        score += 1
        reasons.append("text mentions plant/station/محطة (+1)")

    return score, reasons


def extract_capacity(tags: Dict) -> str:
    for k in CAPACITY_KEYS:
        v = tags.get(k)
        if v:
            return str(v)
    return ""


def merge_elements(*datasets: Dict) -> List[Dict]:
    """
    Merge elements from multiple Overpass results and dedupe by (type,id).
    Prefer the element with more tags if duplicates exist.
    """
    best: Dict[Tuple[str, int], Dict] = {}
    for data in datasets:
        for e in data.get("elements", []) or []:
            t = e.get("type")
            i = e.get("id")
            if not t or i is None:
                continue
            key = (t, int(i))
            prev = best.get(key)
            if prev is None:
                best[key] = e
            else:
                # keep the one with more tags (often richer)
                prev_tags = prev.get("tags", {}) or {}
                new_tags = e.get("tags", {}) or {}
                if len(new_tags) > len(prev_tags):
                    best[key] = e
    return list(best.values())


# ---------- Main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi_geojson", required=True, help="AOI geojson file")
    ap.add_argument("--out_csv", required=True, help="Output plants.csv path")
    ap.add_argument("--debug_json", default="", help="Optional debug JSON output path")
    ap.add_argument("--min_score", type=int, default=3, help="Minimum score to keep a candidate (default: 3)")
    ap.add_argument("--timeout", type=int, default=180, help="Overpass timeout seconds (default: 180)")
    ap.add_argument(
        "--overpass_urls",
        default=",".join(DEFAULT_ENDPOINTS),
        help="Comma-separated Overpass interpreter endpoints",
    )
    args = ap.parse_args()

    aoi = Path(args.aoi_geojson)
    south, west, north, east = aoi_bbox(aoi)
    endpoints = [u.strip() for u in args.overpass_urls.split(",") if u.strip()]

    q1 = build_query_strict(OMAN_AREA_ID, timeout_s=args.timeout)
    q2 = build_query_fuzzy(OMAN_AREA_ID, timeout_s=args.timeout)

    # Fetch both datasets (endpoint failover handled inside post_overpass)
    data1 = post_overpass(q1, endpoints=endpoints)
    data2 = post_overpass(q2, endpoints=endpoints)

    merged = merge_elements(data1, data2)

    candidates: List[Candidate] = []
    for e in merged:
        ll = get_lat_lon(e)
        if not ll:
            continue
        lat, lon = ll

        # Filter by your AOI bbox here
        if not in_bbox(lat, lon, south, west, north, east):
            continue

        tags = e.get("tags", {}) or {}
        score, reasons = score_candidate(tags)

        candidates.append(Candidate(
            osm_type=e.get("type", ""),
            osm_id=int(e.get("id")),
            lat=float(lat),
            lon=float(lon),
            tags=tags,
            score=score,
            reasons=reasons,
        ))

    # Keep only strong-enough hits
    kept = [c for c in candidates if c.score >= args.min_score]
    kept.sort(key=lambda c: (-c.score, c.tags.get("name", "")))

    # Write CSV
    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)

    fields = ["plant_id", "name", "lat", "lon", "country", "capacity_m3_day",
              "osm_type", "osm_id", "operator", "score", "reasons", "tags_json"]

    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for c in kept:
            tags = c.tags
            name = tags.get("name") or tags.get("operator") or tags.get("brand") or ""
            operator = tags.get("operator", "")
            cap = extract_capacity(tags)
            country = tags.get("addr:country", "")  # often blank

            w.writerow({
                "plant_id": f"osm:{c.osm_type}/{c.osm_id}",
                "name": name,
                "lat": c.lat,
                "lon": c.lon,
                "country": country,
                "capacity_m3_day": cap,
                "osm_type": c.osm_type,
                "osm_id": c.osm_id,
                "operator": operator,
                "score": c.score,
                "reasons": "; ".join(c.reasons),
                "tags_json": json.dumps(tags, ensure_ascii=False),
            })

    # Optional debug JSON (includes rejected too)
    if args.debug_json:
        dbg = Path(args.debug_json)
        dbg.parent.mkdir(parents=True, exist_ok=True)
        debug_payload = {
            "aoi_bbox": {"south": south, "west": west, "north": north, "east": east},
            "min_score": args.min_score,
            "counts": {
                "q1_elements": len((data1.get("elements") or [])),
                "q2_elements": len((data2.get("elements") or [])),
                "merged_unique": len(merged),
                "in_aoi": len(candidates),
                "kept": len(kept),
            },
            "kept": [
                {
                    "osm_type": c.osm_type,
                    "osm_id": c.osm_id,
                    "lat": c.lat,
                    "lon": c.lon,
                    "score": c.score,
                    "reasons": c.reasons,
                    "tags": c.tags,
                } for c in kept
            ],
            "rejected": [
                {
                    "osm_type": c.osm_type,
                    "osm_id": c.osm_id,
                    "lat": c.lat,
                    "lon": c.lon,
                    "score": c.score,
                    "reasons": c.reasons,
                    "tags": c.tags,
                } for c in candidates if c.score < args.min_score
            ],
        }
        dbg.write_text(json.dumps(debug_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print summary
    print(f"✅ wrote plants CSV: {outp} (rows={len(kept)})")
    if args.debug_json:
        print(f"🧪 wrote debug JSON: {args.debug_json}")
    print(f"AOI bbox used: south={south:.4f}, west={west:.4f}, north={north:.4f}, east={east:.4f}")
    print(f"Overpass endpoints tried: {len(endpoints)}")


if __name__ == "__main__":
    main()
