#!/usr/bin/env python3
import argparse, joblib, pickle
from pathlib import Path
import pandas as pd, numpy as np

def month_ord(s):
    try:
        y, m = str(s).split("-")[:2]
        return int(y) * 12 + int(m)
    except Exception:
        return -10**9

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--model", required=True, help="joblib from train_hab_baseline_cv.py (must contain 'pipe' and 'features')")
    ap.add_argument("--id_col", default="tile")
    ap.add_argument("--group_by", default="scene_id")
    ap.add_argument("--month_key", default="month_key", help="column with YYYY-MM month (add beforehand if missing)")
    ap.add_argument("--cv_time_folds", type=int, default=5, help="time-based OOF folds")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    # Load dataset
    df = pd.read_csv(args.labels_csv).copy()
    if args.month_key not in df.columns:
        df[args.month_key] = df[args.group_by].astype(str).str.extract(r"(20\d{2})[-_\.]?(\d{2})")[0] + "-" + \
                             df[args.group_by].astype(str).str.extract(r"(20\d{2})[-_\.]?(\d{2})")[1]
    df["_mord"] = df[args.month_key].map(month_ord)

    # Load model and prepare
    pack = joblib.load(args.model)
    pipe = pack["pipe"]
    feats = pack["features"]

    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = df.dropna(subset=feats + ["hab_label"]).copy().reset_index(drop=True)

    # Build chronological folds
    uniq = np.array(sorted([m for m in keep["_mord"].unique() if m >= 0]))
    if len(uniq) < args.cv_time_folds:
        blocks = [uniq]  # fallback
    else:
        blocks = np.array_split(uniq, args.cv_time_folds)

    p_oof = np.full(len(keep), np.nan, dtype=float)

    for i, block in enumerate(blocks, 1):
        te_mask = keep["_mord"].isin(block)
        tr_mask = keep["_mord"] < block.min()
        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            continue

        X_tr = keep.loc[tr_mask, feats].values
        y_tr = keep.loc[tr_mask, "hab_label"].astype(int).values
        X_te = keep.loc[te_mask, feats].values

        # Clone pipeline safely
        pipe_i = pickle.loads(pickle.dumps(pipe))
        pipe_i.fit(X_tr, y_tr)
        p_oof[te_mask.values] = pipe_i.predict_proba(X_te)[:, 1]

    # Handle leftovers (earliest months)
    miss = np.isnan(p_oof)
    if miss.any():
        last_block_end = blocks[0].min() if blocks else keep["_mord"].max()
        tr_mask = keep["_mord"] > last_block_end
        if tr_mask.sum() > 0:
            pipe_f = pickle.loads(pickle.dumps(pipe))
            pipe_f.fit(keep.loc[tr_mask, feats].values, keep.loc[tr_mask, "hab_label"].astype(int).values)
            p_oof[miss] = pipe_f.predict_proba(keep.loc[miss, feats].values)[:, 1]
        else:
            pipe.fit(keep[feats].values, keep["hab_label"].astype(int).values)
            p_oof[miss] = pipe.predict_proba(keep[feats].values)[:, 1][miss]

    # Save
    out = keep[[args.id_col, args.group_by, "hab_label"]].copy()
    out["p_tab"] = p_oof
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"✓ wrote OOF p_tab to {args.out_csv} (rows={len(out)}, nan={np.isnan(p_oof).sum()})")

if __name__ == "__main__":
    main()
