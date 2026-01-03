#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import quote
import time

import pandas as pd
import requests

PC_ITEM_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/{}"


def fetch_item(scene_id: str, timeout: int = 60) -> dict | None:
    url = PC_ITEM_URL.format(quote(scene_id, safe=""))
    r = requests.get(url, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def expand_paths(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in patterns:
        pp = Path(p)
        if pp.exists():
            out.append(pp)
        else:
            out.extend(sorted(Path().glob(p)))

    # de-dup
    seen = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen and p.exists():
            uniq.append(p)
            seen.add(rp)
    return uniq


def bbox_centroid_wgs84(item: dict) -> tuple[float, float] | None:
    """
    Planetary Computer STAC items provide:
      bbox = [minLon, minLat, maxLon, maxLat]  (WGS84)
    We compute midpoint. This is stable + already in WGS84.
    """
    bbox = item.get("bbox")
    if not bbox or len(bbox) != 4:
        return None

    minx, miny, maxx, maxy = bbox
    lon = (float(minx) + float(maxx)) / 2.0
    lat = (float(miny) + float(maxy)) / 2.0

    # sanity check
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        # If somehow swapped, try swap
        if (-180 <= lat <= 180 and -90 <= lon <= 90):
            lon, lat = lat, lon
        else:
            return None

    return lat, lon


def main():
    ap = argparse.ArgumentParser(
        description="Get WGS84 scene centroids from Planetary Computer STAC (correct lat/lon). "
                    "Optionally collapse to ONE lat/lon pair for a plant/AOI."
    )
    ap.add_argument(
        "--csv",
        required=True,
        nargs="+",
        help="One or more CSV files/globs containing a scene_id column (quote globs).",
    )
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--scene_col", default="scene_id", help="Column name for scene ids (default: scene_id)")
    ap.add_argument(
        "--mode",
        choices=["per_scene", "unique_coords", "one_pair"],
        default="one_pair",
        help=(
            "per_scene = output one row per scene_id; "
            "unique_coords = dedupe identical (lat,lon) pairs; "
            "one_pair = output ONE representative pair (median lat/lon)."
        ),
    )
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")
    ap.add_argument("--sleep", type=float, default=0.0, help="Optional sleep between requests (seconds)")
    args = ap.parse_args()

    paths = expand_paths(args.csv)
    if not paths:
        raise SystemExit("❌ No CSV files found. Tip: quote globs like 'deployment/outputs/**/chip_indices.csv'")

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["__source_csv"] = str(p)
        frames.append(df)
    idx = pd.concat(frames, ignore_index=True)

    if args.scene_col not in idx.columns:
        raise SystemExit(f"❌ scene_col='{args.scene_col}' not found. Available: {list(idx.columns)}")

    # unique scene_ids only (fast, avoids repeated identical results)
    scene_ids = (
        idx[args.scene_col]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    if not scene_ids:
        raise SystemExit("❌ No scene_ids found after dropping NaNs.")

    cache: dict[str, tuple[float, float] | None] = {}
    rows = []
    missing_item = 0
    missing_bbox = 0

    for sid in scene_ids:
        if sid in cache:
            latlon = cache[sid]
        else:
            item = fetch_item(sid, timeout=args.timeout)
            if not item:
                cache[sid] = None
                missing_item += 1
                latlon = None
            else:
                latlon = bbox_centroid_wgs84(item)
                if latlon is None:
                    missing_bbox += 1
                cache[sid] = latlon

        if args.sleep > 0:
            time.sleep(args.sleep)

        if latlon is None:
            continue

        lat, lon = latlon
        rows.append({"scene_id": sid, "tile_lat": float(lat), "tile_lon": float(lon)})

    if not rows:
        raise SystemExit(
            "❌ No valid centroids produced.\n"
            f"missing_item={missing_item} missing_bbox={missing_bbox}"
        )

    out = pd.DataFrame(rows)

    if args.mode == "per_scene":
        # keep as-is: one row per scene_id
        pass

    elif args.mode == "unique_coords":
        # drop duplicate coordinate pairs
        out = out.drop_duplicates(subset=["tile_lat", "tile_lon"]).reset_index(drop=True)

    elif args.mode == "one_pair":
        # ONE stable coordinate pair for the plant/AOI
        # (median is robust if you ever have tiny jitter across scenes)
        lat = float(out["tile_lat"].median())
        lon = float(out["tile_lon"].median())
        out = pd.DataFrame([{"tile_lat": lat, "tile_lon": lon, "n_scenes_used": int(len(out))}])

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f"✅ wrote {args.out_csv} rows={len(out)}")
    print(f"   scenes_in={len(scene_ids)} scenes_with_coords={len(rows)}")
    print(f"   missing_item={missing_item} missing_bbox_or_invalid={missing_bbox}")


if __name__ == "__main__":
    main()
