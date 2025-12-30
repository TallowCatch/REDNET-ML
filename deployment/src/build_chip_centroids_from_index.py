#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from pyproj import Transformer

PC_ITEM_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/{}"


def fetch_item(scene_id: str, timeout=60) -> dict | None:
    url = PC_ITEM_URL.format(quote(scene_id, safe=""))
    r = requests.get(url, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def expand_paths(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in patterns:
        # If shell expanded already, it's a literal path and exists.
        pp = Path(p)
        if pp.exists():
            out.append(pp)
            continue

        # Otherwise, expand ourselves (supports ** if user passes it).
        matches = sorted(Path().glob(p))
        out.extend(matches)

    # de-dup while preserving order
    seen = set()
    uniq = []
    for p in out:
        rp = p.resolve()
        if rp not in seen and p.exists():
            uniq.append(p)
            seen.add(rp)
    return uniq


def main():
    ap = argparse.ArgumentParser(
        description="Build per-tile WGS84 centroids from one or more index.csv files."
    )
    ap.add_argument(
        "--index_csv",
        required=True,
        nargs="+",
        help="One or more index.csv files or globs (quote globs): 'data/aerial_*/index.csv'",
    )
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    args = ap.parse_args()

    paths = expand_paths(args.index_csv)
    if not paths:
        raise SystemExit("❌ No index_csv files found. Tip: quote globs like 'data/aerial_*/index.csv'")

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["__source_index_csv"] = str(p)
        frames.append(df)

    idx = pd.concat(frames, ignore_index=True)

    required = {"tile", "scene_id", "xmin", "ymin", "xmax", "ymax"}
    missing_cols = required - set(idx.columns)
    if missing_cols:
        raise SystemExit(f"❌ Missing columns in concatenated index: {sorted(missing_cols)}")

    # numeric bounds
    for c in ["xmin", "ymin", "xmax", "ymax"]:
        idx[c] = pd.to_numeric(idx[c], errors="coerce")

    # Cache scene_id -> transformer
    transformer_cache: dict[str, Transformer | None] = {}

    out_rows = []
    missing = 0
    missing_epsg = 0
    missing_item = 0

    for row in idx.itertuples(index=False):
        tile = str(row.tile)
        sid = str(row.scene_id)

        if pd.isna(row.xmin) or pd.isna(row.xmax) or pd.isna(row.ymin) or pd.isna(row.ymax):
            missing += 1
            out_rows.append((tile, sid, None, None, getattr(row, "__source_index_csv", None)))
            continue

        cx = (float(row.xmin) + float(row.xmax)) / 2.0
        cy = (float(row.ymin) + float(row.ymax)) / 2.0

        if sid not in transformer_cache:
            item = fetch_item(sid)
            if not item:
                transformer_cache[sid] = None
                missing_item += 1
            else:
                epsg = item.get("properties", {}).get("proj:epsg")
                if not epsg:
                    transformer_cache[sid] = None
                    missing_epsg += 1
                else:
                    transformer_cache[sid] = Transformer.from_crs(
                        f"EPSG:{epsg}", "EPSG:4326", always_xy=True
                    )

        tfm = transformer_cache[sid]
        if tfm is None:
            missing += 1
            out_rows.append((tile, sid, None, None, getattr(row, "__source_index_csv", None)))
            continue

        lon, lat = tfm.transform(cx, cy)
        out_rows.append((tile, sid, float(lat), float(lon), getattr(row, "__source_index_csv", None)))

    out = pd.DataFrame(out_rows, columns=["tile", "scene_id", "tile_lat", "tile_lon", "source_index_csv"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f"✅ wrote {args.out_csv} rows={len(out)} missing={missing}")
    print(f"   missing_item={missing_item} missing_proj_epsg={missing_epsg}")


if __name__ == "__main__":
    main()
