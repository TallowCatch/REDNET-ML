from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_json(p: Path) -> dict:
    return json.loads(p.read_text())


def derive_scene_root_from_scene_id(scene_id: str) -> str:
    """
    Sentinel-like IDs often look like:
      S2A_MSIL2A_YYYYMMDDThhmmss_Rxxx_Txxxxx_YYYYMMDDThhmmss
    The "scene root" for grouping is usually the first 5 parts:
      S2A_MSIL2A_YYYY..._Rxxx_Txxxxx
    """
    parts = str(scene_id).split("_")
    return "_".join(parts[:5]) if len(parts) >= 5 else str(scene_id)


def main():
    ap = argparse.ArgumentParser(description="Operational scene alert aggregation from tile-level inference CSV.")
    ap.add_argument("--tile_pred_csv", required=True, help="CSV output of batch_infer_csv.py (must include prob)")
    ap.add_argument("--scene_col", default="scene_id", help="Column to group by (default: scene_id)")
    ap.add_argument("--make_scene_root", action="store_true",
                    help="If set, create scene_root from scene_col and group by scene_root instead.")
    ap.add_argument("--threshold_from", required=True, help="Path to thresholds.json (or a versioned thresholds file)")
    ap.add_argument("--agg", default="max", choices=["max", "mean"], help="How to compute scene_prob from tiles")
    ap.add_argument("--out_csv", required=True, help="Where to write scene-level alerts CSV")
    ap.add_argument("--top_k", type=int, default=20, help="Print top-K scenes by scene_prob")
    args = ap.parse_args()

    df = pd.read_csv(Path(args.tile_pred_csv))
    if "prob" not in df.columns:
        raise SystemExit("❌ tile_pred_csv must contain a 'prob' column (run batch_infer_csv.py first).")

    scene_col = args.scene_col
    if scene_col not in df.columns:
        raise SystemExit(f"❌ scene_col '{scene_col}' not found in tile_pred_csv.")

    thrj = load_json(Path(args.threshold_from))
    tile_thr = float(thrj.get("default_threshold", 0.5))

    # optional: derive scene_root
    group_col = scene_col
    if args.make_scene_root:
        df["scene_root"] = df[scene_col].map(derive_scene_root_from_scene_id)
        group_col = "scene_root"

    # aggregation
    if args.agg == "max":
        scene_prob = df.groupby(group_col)["prob"].max()
    else:
        scene_prob = df.groupby(group_col)["prob"].mean()

    max_tile = df.groupby(group_col)["prob"].max()
    mean_tile = df.groupby(group_col)["prob"].mean()
    n_tiles = df.groupby(group_col).size()
    n_over = df.groupby(group_col).apply(lambda g: int((g["prob"] >= tile_thr).sum()))

    # best tile evidence
    def best_tile_row(g: pd.DataFrame) -> pd.Series:
        i = int(g["prob"].values.argmax())
        row = g.iloc[i]
        return pd.Series({
            "best_tile": row.get("tile", ""),
            "best_tile_prob": float(row["prob"]),
        })

    best = df.groupby(group_col).apply(best_tile_row)

    out = pd.DataFrame({
        "scene_id": scene_prob.index,
        "scene_prob": scene_prob.values,
        "max_tile_prob": max_tile.values,
        "mean_tile_prob": mean_tile.values,
        "n_tiles": n_tiles.values,
        "n_tiles_over_thr": n_over.values,
        "threshold_tile_used": tile_thr,
        "agg": args.agg,
    })

    out = out.merge(best, left_on="scene_id", right_index=True, how="left")

    out["scene_alert"] = (out["n_tiles_over_thr"] > 0).astype(int)

    out = out.sort_values("scene_prob", ascending=False).reset_index(drop=True)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("✅ wrote:", out_path)
    print(f"rows(scenes)={len(out)} | alerts={int(out['scene_alert'].sum())} | tile_threshold={tile_thr:.6f}")

    print("\nTop scenes:")
    show = out.head(args.top_k)[["scene_id", "scene_prob", "n_tiles", "n_tiles_over_thr", "best_tile", "best_tile_prob", "scene_alert"]]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
