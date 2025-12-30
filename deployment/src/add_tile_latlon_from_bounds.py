#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv
from pathlib import Path

import rasterio
from pystac_client import Client
import planetary_computer as pc
from pyproj import CRS, Transformer

S2_COLLECTION = "sentinel-2-l2a"
WGS84 = CRS.from_epsg(4326)

def fetch_item(stac: Client, scene_id: str):
    coll = stac.search(collections=[S2_COLLECTION], ids=[scene_id]).item_collection()
    items = list(coll)
    if not items:
        return None
    return pc.sign(items[0])

def get_scene_crs(item) -> CRS | None:
    # Open any georeferenced asset to read CRS (B03 is fine)
    for key in ("B03", "B04", "B02", "visual"):
        if key in item.assets:
            href = item.assets[key].href
            try:
                with rasterio.open(href) as ds:
                    if ds.crs:
                        return CRS.from_wkt(ds.crs.to_wkt())
            except Exception:
                continue
    return None

def main():
    ap = argparse.ArgumentParser(description="Add tile_lon/tile_lat (EPSG:4326) to index.csv using bounds + per-scene CRS")
    ap.add_argument("--in_csv", required=True, help="Path to index.csv (has xmin,ymin,xmax,ymax)")
    ap.add_argument("--out_csv", required=True, help="Output CSV with tile_lon/tile_lat filled")
    ap.add_argument("--scene_col", default="scene_id")
    ap.add_argument("--tile_col", default="tile")
    ap.add_argument("--xmin", default="xmin")
    ap.add_argument("--ymin", default="ymin")
    ap.add_argument("--xmax", default="xmax")
    ap.add_argument("--ymax", default="ymax")
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    rows = []
    with in_path.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)

    if not rows:
        raise SystemExit("Empty input CSV")

    stac = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", ignore_conformance=True)

    item_cache = {}
    crs_cache = {}
    tfm_cache = {}

    # ensure output fields exist
    out_fields = list(rows[0].keys())
    if "tile_lon" not in out_fields:
        out_fields.append("tile_lon")
    if "tile_lat" not in out_fields:
        out_fields.append("tile_lat")

    filled = 0
    skipped = 0

    for r in rows:
        scene_id = r.get(args.scene_col, "").strip()
        if not scene_id:
            skipped += 1
            continue

        # centroid in scene CRS units
        try:
            xmin = float(r[args.xmin]); ymin = float(r[args.ymin])
            xmax = float(r[args.xmax]); ymax = float(r[args.ymax])
        except Exception:
            skipped += 1
            continue

        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)

        # get item + CRS + transformer
        if scene_id not in item_cache:
            item_cache[scene_id] = fetch_item(stac, scene_id)
        item = item_cache[scene_id]
        if item is None:
            skipped += 1
            continue

        if scene_id not in crs_cache:
            crs_cache[scene_id] = get_scene_crs(item)
        crs_scene = crs_cache[scene_id]
        if crs_scene is None:
            skipped += 1
            continue

        if scene_id not in tfm_cache:
            tfm_cache[scene_id] = Transformer.from_crs(crs_scene, WGS84, always_xy=True)

        tfm = tfm_cache[scene_id]
        lon, lat = tfm.transform(cx, cy)

        r["tile_lon"] = f"{lon:.8f}"
        r["tile_lat"] = f"{lat:.8f}"
        filled += 1

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(rows)

    print(f"✅ wrote: {out_path}")
    print(f"filled={filled} skipped={skipped} total={len(rows)}")

if __name__ == "__main__":
    main()
