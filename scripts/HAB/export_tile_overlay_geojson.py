#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd


from pyproj import Transformer, CRS

crs = CRS.from_epsg(32640)
print(crs.area_of_use)

# CHANGE THIS to your actual CRS
SRC_CRS = "EPSG:32640"   # example only
DST_CRS = "EPSG:4326"

transformer = Transformer.from_crs(SRC_CRS, DST_CRS, always_xy=True)

def reproject_bbox(xmin, ymin, xmax, ymax):
    lon1, lat1 = transformer.transform(xmin, ymin)
    lon2, lat2 = transformer.transform(xmax, ymax)
    return lon1, lat1, lon2, lat2


def _coerce_num(x):
    try:
        return float(x)
    except Exception:
        return None
    
def strip_chip_suffix(s: str) -> str:
    # S2C_..._0000 → S2C_...
    return s.rsplit("_", 1)[0]



def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def bbox_to_polygon(xmin: float, ymin: float, xmax: float, ymax: float):
    # GeoJSON polygon ring (closed)
    return [[
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
        [xmin, ymin],
    ]]


def load_index_csv(index_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(index_csv)

    required = ["scene_id", "xmin", "ymin", "xmax", "ymax"]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"❌ index.csv missing required column '{c}'. cols={list(df.columns)}")

    df = df.copy()

    # Reconstruct chip_id EXACTLY like tiles_png filenames:
    # <scene_id>_<0000-based counter per scene>
    df["_chip_idx"] = df.groupby("scene_id").cumcount()
    df["chip_id"] = df.apply(
        lambda r: f"{r['scene_id']}_{int(r['_chip_idx']):04d}", axis=1
    )
    df.drop(columns=["_chip_idx"], inplace=True)

    for c in ["xmin", "ymin", "xmax", "ymax"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["xmin", "ymin", "xmax", "ymax"])

    keep = ["chip_id", "xmin", "ymin", "xmax", "ymax"]
    for c in ["scene_id", "datetime", "tile"]:
        if c in df.columns:
            keep.append(c)

    return df[keep]




def load_scores_csv(scores_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(scores_csv)
    if "chip_id" not in df.columns:
        # allow scene_id fallback
        if "scene_id" in df.columns:
            df = df.rename(columns={"scene_id": "chip_id"})
        else:
            raise SystemExit(f"❌ scores csv missing chip_id. cols={list(df.columns)}")

    df["chip_id"] = df["chip_id"].astype(str).map(lambda s: Path(s).stem)

    # coerce numeric score cols where present
    for c in ["hab_prob", "p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def write_geojson(df: pd.DataFrame, out_path: Path, *, id_fields: List[str], score_fields: List[str]) -> None:
    feats = []
    for _, r in df.iterrows():
        xmin, ymin, xmax, ymax = r["xmin"], r["ymin"], r["xmax"], r["ymax"]
        if not np.isfinite([xmin, ymin, xmax, ymax]).all():
            continue

        props: Dict[str, Any] = {}
        for k in id_fields:
            if k in df.columns:
                props[k] = r[k]

        for k in score_fields:
            if k in df.columns:
                v = r[k]
                props[k] = None if (v is None or (isinstance(v, float) and not np.isfinite(v))) else float(v)

        lon1, lat1, lon2, lat2 = reproject_bbox(xmin, ymin, xmax, ymax)
        geom = {
            "type": "Polygon",
            "coordinates": bbox_to_polygon(lon1, lat1, lon2, lat2),
        }

        feats.append({"type": "Feature", "geometry": geom, "properties": props})

    out = {"type": "FeatureCollection", "features": feats}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    print(f"✅ wrote {out_path} ({len(feats)} tiles)")


def main():
    ap = argparse.ArgumentParser("Export tile overlay GeoJSON (chips/index.csv + scores/inference csv)")
    ap.add_argument("--plant_root", required=True, help="e.g. deployment/outputs/by_plant/osm_way_386838289")
    ap.add_argument("--month", default=None, help="YYYY-MM (if omitted, exports all months found)")
    ap.add_argument("--index_rel", default="chips/index.csv", help="relative path to index.csv inside month folder")
    ap.add_argument("--scores_rel", default="detector_scores.csv",
                    help="relative path to detector_scores.csv inside month folder (or inference.csv)")
    ap.add_argument("--also_merge_inference", action="store_true",
                    help="if set, merge inference.csv too (hab_prob etc) when scores_rel is detector_scores.csv")
    ap.add_argument("--out_dir", required=True, help="where to write geojson outputs")
    args = ap.parse_args()

    plant_root = Path(args.plant_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    month_dirs = [plant_root / args.month] if args.month else sorted([p for p in plant_root.iterdir() if p.is_dir()])

    for md in month_dirs:
        if not md.is_dir():
            continue
        month = md.name

        index_csv = md / args.index_rel
        scores_csv = md / args.scores_rel

        if not index_csv.exists():
            print(f"⏭️  skip {month}: missing {index_csv}")
            continue
        if not scores_csv.exists():
            print(f"⏭️  skip {month}: missing {scores_csv}")
            continue

        idx = load_index_csv(index_csv)
        scores = load_scores_csv(scores_csv)

        merged = idx.merge(scores, on="chip_id", how="left")

        # Optionally merge inference.csv as well (common for hab_prob + fused prob)
        if args.also_merge_inference:
            inf_all = plant_root / "inference_all_months.csv"
            if inf_all.exists():
                inf_df = pd.read_csv(inf_all)

                # normalize inference ID
                if "scene_id" in inf_df.columns:
                    inf_df["join_id"] = inf_df["scene_id"].astype(str)
                elif "chip_id" in inf_df.columns:
                    inf_df["join_id"] = inf_df["chip_id"].astype(str).map(strip_chip_suffix)
                else:
                    raise SystemExit("❌ inference file missing scene_id / chip_id")

                # normalize overlay IDs
                merged["join_id"] = merged["chip_id"].astype(str).map(strip_chip_suffix)

                # filter month if available
                if "month_key" in inf_df.columns:
                    inf_df = inf_df[inf_df["month_key"] == month]

                merged = merged.merge(
                    inf_df,
                    on="join_id",
                    how="left",
                    suffixes=("", "_inf")
                )

            
        score_fields = []
        for c in ["hab_prob", "p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med"]:
            if c in merged.columns:
                score_fields.append(c)

        # If you have a fused prob column, include it too automatically
        for c in ["p_fused", "prob", "hab_prob_raw"]:
            if c in merged.columns:
                score_fields.append(c)

        out_path = out_dir / f"{month}_tile_overlay.geojson"
        write_geojson(
            merged,
            out_path,
            id_fields=["chip_id", "month_key", "datetime", "tile", "scene_id"],
            score_fields=score_fields,
        )


if __name__ == "__main__":
    main()
