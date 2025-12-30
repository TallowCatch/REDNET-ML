#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import requests
from urllib.parse import quote

PC_ITEM_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/{}"


def centroid_from_bbox(bbox):
    # bbox = [minLon, minLat, maxLon, maxLat]
    if not bbox or len(bbox) != 4:
        return None
    minx, miny, maxx, maxy = bbox
    lon = (minx + maxx) / 2.0
    lat = (miny + maxy) / 2.0
    return lat, lon


def derive_scene_root(scene_id: str) -> str:
    parts = str(scene_id).split("_")
    return "_".join(parts[:5]) if len(parts) >= 5 else str(scene_id)


def fetch_item(scene_id: str, timeout=60):
    url = PC_ITEM_URL.format(quote(scene_id, safe=""))
    r = requests.get(url, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer_csv", required=True, help="tile-level inference csv with scene_id")
    ap.add_argument("--out_csv", required=True, help="scene centroids output csv")
    args = ap.parse_args()

    df = pd.read_csv(args.infer_csv)
    if "scene_id" not in df.columns:
        raise SystemExit("❌ infer_csv must contain scene_id")

    scenes = sorted(df["scene_id"].astype(str).unique())

    out_rows = []
    missing = 0

    for sid in scenes:
        item = fetch_item(sid)
        if not item:
            missing += 1
            out_rows.append((sid, derive_scene_root(sid), None, None))
            continue

        cen = centroid_from_bbox(item.get("bbox"))
        if not cen:
            missing += 1
            out_rows.append((sid, derive_scene_root(sid), None, None))
            continue

        lat, lon = cen
        out_rows.append((sid, derive_scene_root(sid), float(lat), float(lon)))

    out = pd.DataFrame(out_rows, columns=["scene_id", "scene_root", "scene_lat", "scene_lon"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f"✅ wrote {args.out_csv} scenes={len(out)} missing={missing}")
    if missing == len(out):
        print("⚠️ All missing: either your scene_id is not a real PC item id, or PC endpoint is blocked.")
    elif missing > 0:
        print("⚠️ Some missing: those scene_ids likely don’t exist in PC as-is (format mismatch).")


if __name__ == "__main__":
    main()
