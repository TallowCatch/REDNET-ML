#!/usr/bin/env python3
import argparse, glob, os
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="rednet-risk-viewer/public/data/plant_*_hab.csv")
    ap.add_argument("--out_csv", default="runs/datasets/plants_all_merged.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched: {args.glob}")

    rows = []
    for f in files:
        df = pd.read_csv(f)
        df["plant_file"] = os.path.basename(f)
        rows.append(df)

    merged = pd.concat(rows, ignore_index=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)

    pos = int((pd.to_numeric(merged.get("hab_label", 0), errors="coerce") > 0.5).sum()) if "hab_label" in merged.columns else 0
    print(f"✓ merged {len(files)} plant CSVs -> {args.out_csv} (rows={len(merged)}, base_pos={pos})")

if __name__ == "__main__":
    main()
