#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd, numpy as np, joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True,
                    help="Your dataset table (e.g. runs/datasets/hab_train_nonleaky.csv)")
    ap.add_argument("--model", required=True,
                    help="joblib from train_hab_baseline_cv.py (model.joblib)")
    ap.add_argument("--id_col", default="tile",
                    help="column that uniquely identifies a chip (tile filename or similar)")
    ap.add_argument("--group_by", default="scene_id")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.labels_csv)
    pack = joblib.load(args.model)
    pipe = pack["pipe"]
    feats = pack["features"]

    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = df.dropna(subset=feats + ["hab_label"]).copy()

    p = pipe.predict_proba(keep[feats].values)[:,1]
    out = keep[[args.id_col, args.group_by, "hab_label"]].copy()
    out["p_tab"] = p

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"✓ wrote {args.out_csv} (rows={len(out)})")

if __name__ == "__main__":
    main()
