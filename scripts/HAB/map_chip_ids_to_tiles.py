#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--det_csv", required=True)
ap.add_argument("--split_csv", required=True, help="CSV or txt file listing tile filenames in order")
ap.add_argument("--out_csv", required=True)
args = ap.parse_args()

# load detector csv
det = pd.read_csv(args.det_csv)

# load split file (list of image paths)
tiles = pd.read_csv(args.split_csv, header=None).iloc[:,0].astype(str).tolist()

# create id→tile mapping (1-indexed, since chip_id starts at 1)
id2tile = {i+1: Path(t).stem for i, t in enumerate(tiles)}

# add mapped tile column
det["tile"] = det["chip_id"].map(id2tile)

# clean order
det = det[["tile"] + [c for c in det.columns if c != "tile"]]

Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
det.to_csv(args.out_csv, index=False)
print(f"✅ wrote mapped CSV with tile names: {args.out_csv} (rows={len(det)})")
