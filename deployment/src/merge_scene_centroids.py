#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd


def derive_scene_root(scene_id: str) -> str:
    parts = str(scene_id).split("_")
    return "_".join(parts[:5]) if len(parts) >= 5 else str(scene_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_alerts_csv", required=True)
    ap.add_argument("--scene_centroids_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    alerts = pd.read_csv(args.scene_alerts_csv)
    cents = pd.read_csv(args.scene_centroids_csv)

    if "scene_id" not in alerts.columns:
        raise SystemExit("❌ scene_alerts_csv must contain scene_id (this is your scene_root when --make_scene_root is used).")

    # alerts["scene_id"] is actually the scene_root if you used --make_scene_root
    alerts = alerts.copy()
    alerts["scene_root"] = alerts["scene_id"].astype(str)

    cents = cents.copy()

    # ensure cents has scene_root column
    if "scene_root" not in cents.columns:
        if "scene_id" not in cents.columns:
            raise SystemExit("❌ scene_centroids_csv must contain scene_id or scene_root.")
        cents["scene_root"] = cents["scene_id"].astype(str).map(derive_scene_root)

    # aggregate centroid per root (some roots may have 2 items like you saw)
    cents_root = (
        cents.groupby("scene_root", as_index=False)
        .agg(scene_lat=("scene_lat", "mean"), scene_lon=("scene_lon", "mean"))
    )

    out = alerts.merge(cents_root, on="scene_root", how="left")

    missing = int(out["scene_lat"].isna().sum())
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f"✅ wrote {args.out_csv} rows={len(out)} missing_scene_latlon={missing}")
    if missing > 0:
        print("⚠️ If missing > 0 here, it means those scene_roots had no matching centroids in scene_centroids.csv.")


if __name__ == "__main__":
    main()
