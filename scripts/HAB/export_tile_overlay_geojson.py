#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


def strip_chip_suffix(s: str) -> str:
    # S2C_..._0000 -> S2C_...
    return s.rsplit("_", 1)[0]


def bbox_to_polygon(xmin: float, ymin: float, xmax: float, ymax: float):
    return [[
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
        [xmin, ymin],
    ]]


def safe_json_val(v):
    # convert numpy scalars + NaN -> None
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    # keep strings, timestamps, etc.
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    # pandas NaT
    if v is pd.NaT:
        return None
    return v


def _infer_epsg_from_scene(scene_id: str) -> Optional[str]:
    """
    Sentinel-2 scene_id includes tile like T40QFM.
    UTM zone = 40, latitude band Q => Northern hemisphere => EPSG:32640.
    """
    if not isinstance(scene_id, str):
        return None
    # find substring like _T40QFM_
    parts = scene_id.split("_")
    tile = None
    for p in parts:
        if p.startswith("T") and len(p) >= 6 and p[1:3].isdigit():
            tile = p
            break
    if tile is None:
        return None

    zone = int(tile[1:3])
    lat_band = tile[3]  # e.g., 'Q'
    is_north = lat_band >= "N"  # Sentinel latitude bands: N..X are north
    epsg = 32600 + zone if is_north else 32700 + zone
    return f"EPSG:{epsg}"


def load_index_csv(index_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(index_csv)

    required = ["scene_id", "xmin", "ymin", "xmax", "ymax"]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"❌ index.csv missing required column '{c}'. cols={list(df.columns)}")

    df = df.copy()

    # chip_id must match tiles_png filenames: <scene_id>_<0000>
    df["_chip_idx"] = df.groupby("scene_id").cumcount()
    df["chip_id"] = df.apply(lambda r: f"{r['scene_id']}_{int(r['_chip_idx']):04d}", axis=1)
    df.drop(columns=["_chip_idx"], inplace=True)

    for c in ["xmin", "ymin", "xmax", "ymax"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["xmin", "ymin", "xmax", "ymax"])

    # tile filename if you want it
    df["tile"] = df["chip_id"].astype(str) + ".jpg"

    keep = ["chip_id", "scene_id", "tile", "xmin", "ymin", "xmax", "ymax"]
    if "datetime" in df.columns:
        keep.append("datetime")

    return df[keep]


def load_scores_csv(scores_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(scores_csv)
    if "chip_id" not in df.columns:
        if "scene_id" in df.columns:
            df = df.rename(columns={"scene_id": "chip_id"})
        else:
            raise SystemExit(f"❌ scores csv missing chip_id. cols={list(df.columns)}")

    df["chip_id"] = df["chip_id"].astype(str).map(lambda s: Path(s).stem)

    for c in ["hab_prob", "p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med", "p_fused"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def maybe_fix_scaled_coords(df: pd.DataFrame, xmul: float, ymul: float) -> pd.DataFrame:
    """
    Your case needs: x *= -10, y *= 10 (to get plausible UTM meters).
    Keep it parameterized so you can disable later (xmul=1, ymul=1).
    """
    df = df.copy()
    for c in ["xmin", "xmax"]:
        df[c] = df[c].astype(float) * float(xmul)
    for c in ["ymin", "ymax"]:
        df[c] = df[c].astype(float) * float(ymul)
    return df


def write_geojson(
    df: pd.DataFrame,
    out_path: Path,
    transformer: Transformer,
    *,
    id_fields: List[str],
    score_fields: List[str],
) -> None:
    feats = []
    for _, r in df.iterrows():
        xmin, ymin, xmax, ymax = r["xmin"], r["ymin"], r["xmax"], r["ymax"]
        if not np.isfinite([xmin, ymin, xmax, ymax]).all():
            continue

        # reproject bbox corners to lon/lat
        lon1, lat1 = transformer.transform(float(xmin), float(ymin))
        lon2, lat2 = transformer.transform(float(xmax), float(ymax))

        props: Dict[str, Any] = {}
        for k in id_fields:
            if k in df.columns:
                props[k] = safe_json_val(r[k])

        for k in score_fields:
            if k in df.columns:
                props[k] = safe_json_val(r[k])

        geom = {"type": "Polygon", "coordinates": bbox_to_polygon(lon1, lat1, lon2, lat2)}
        feats.append({"type": "Feature", "geometry": geom, "properties": props})

    out = {"type": "FeatureCollection", "features": feats}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # force valid JSON (no NaN allowed)
    out_path.write_text(json.dumps(out, allow_nan=False))
    print(f"✅ wrote {out_path} ({len(feats)} tiles)")


def main():
    ap = argparse.ArgumentParser("Export tile overlay GeoJSON (index + detector scores + optional inference)")
    ap.add_argument("--plant_root", required=True)
    ap.add_argument("--month", required=True, help="YYYY-MM")
    ap.add_argument("--index_rel", default="chips/index.csv")
    ap.add_argument("--scores_rel", default="detector_scores.csv")
    ap.add_argument("--also_merge_inference", action="store_true")
    ap.add_argument("--inference_rel", default="inference.csv",
                    help="monthly inference file inside month dir (fallback: plant_root/inference_all_months.csv)")
    ap.add_argument("--out_dir", required=True)

    # CRS / coordinate fix
    ap.add_argument("--src_crs", default="auto", help="auto or EPSG:32640 etc")
    ap.add_argument("--dst_crs", default="EPSG:4326")
    ap.add_argument("--xmul", type=float, default=-10.0, help="multiply xmin/xmax by this (your case: -10)")
    ap.add_argument("--ymul", type=float, default=10.0, help="multiply ymin/ymax by this (your case: 10)")

    args = ap.parse_args()

    plant_root = Path(args.plant_root)
    md = plant_root / args.month
    if not md.is_dir():
        raise SystemExit(f"❌ month dir not found: {md}")

    index_csv = md / args.index_rel
    scores_csv = md / args.scores_rel
    if not index_csv.exists():
        raise SystemExit(f"❌ missing {index_csv}")
    if not scores_csv.exists():
        raise SystemExit(f"❌ missing {scores_csv}")

    idx = load_index_csv(index_csv)
    scores = load_scores_csv(scores_csv)

    merged = idx.merge(scores, on="chip_id", how="left")

    # Always stamp month_key so you never get NaN
    merged["month_key"] = args.month

    # scene_id safety
    if "scene_id" not in merged.columns or merged["scene_id"].isna().all():
        merged["scene_id"] = merged["chip_id"].astype(str).map(strip_chip_suffix)

    # Merge inference scene-level -> chips
    if args.also_merge_inference:
        inf_month = md / args.inference_rel
        inf_all = plant_root / "inference_all_months.csv"
        inf_path = inf_month if inf_month.exists() else (inf_all if inf_all.exists() else None)

        if inf_path is not None:
            inf_df = pd.read_csv(inf_path)

            # normalize join key
            if "scene_id" in inf_df.columns:
                inf_df["scene_id"] = inf_df["scene_id"].astype(str)
            elif "chip_id" in inf_df.columns:
                inf_df["scene_id"] = inf_df["chip_id"].astype(str).map(strip_chip_suffix)
            else:
                raise SystemExit(f"❌ inference file missing scene_id/chip_id: {inf_path}")

            # month filter if present (be tolerant)
            if "month_key" in inf_df.columns:
                inf_df["month_key"] = inf_df["month_key"].astype(str)
                inf_df = inf_df[inf_df["month_key"] == args.month]

            # keep only useful cols to avoid exploding
            keep_cols = ["scene_id"]
            for c in ["hab_prob", "p_fused", "prob"]:
                if c in inf_df.columns:
                    inf_df[c] = pd.to_numeric(inf_df[c], errors="coerce")
                    keep_cols.append(c)
            inf_df = inf_df[keep_cols].drop_duplicates(subset=["scene_id"])

            merged = merged.merge(inf_df, on="scene_id", how="left")

    # Fix your scaled/sign-flipped coords BEFORE reprojection
    merged = maybe_fix_scaled_coords(merged, args.xmul, args.ymul)

    # Decide CRS
    if args.src_crs == "auto":
        # infer from first scene_id
        src = _infer_epsg_from_scene(str(merged["scene_id"].iloc[0]))
        src = src or "EPSG:32640"
    else:
        src = args.src_crs

    transformer = Transformer.from_crs(CRS.from_string(src), CRS.from_string(args.dst_crs), always_xy=True)

    score_fields = []
    for c in ["hab_prob", "p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med", "p_fused"]:
        if c in merged.columns:
            score_fields.append(c)

    out_dir = Path(args.out_dir)
    out_path = out_dir / f"{args.month}_tile_overlay.geojson"
    write_geojson(
        merged,
        out_path,
        transformer,
        id_fields=["chip_id", "month_key", "datetime", "tile", "scene_id"],
        score_fields=score_fields,
    )


if __name__ == "__main__":
    main()
