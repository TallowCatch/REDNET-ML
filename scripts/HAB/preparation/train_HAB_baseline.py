#!/usr/bin/env python3
import argparse, glob, json, os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATS = ["fai_mean","rednir_mean","ndwi_mean","chlor_a","kd490","flh"]

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def load_labels(pattern_or_path: str) -> pd.DataFrame:
    paths = []
    if any(ch in pattern_or_path for ch in "*?[]"):
        paths = sorted(glob.glob(pattern_or_path))
        if not paths:
            raise SystemExit(f"No files matched: {pattern_or_path}")
    else:
        paths = [pattern_or_path]
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df["__src__"] = p
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def pick_existing_feats(df, wanted):
    cols = [c for c in wanted if c in df.columns]
    if not cols:
        raise SystemExit("None of the requested features are present in the data.")
    return cols

def make_group_split(df, group_col, test_size=0.25, random_state=42):
    """
    Stratified split by groups:
      - Make a group table (one row per group) with group label = max(hab_label) in that group
      - StratifiedShuffleSplit on the group table
      - Map selected groups back to rows
    """
    if group_col not in df.columns:
        raise SystemExit(f"--group_by '{group_col}' column not in data.")

    grp_tbl = (df.groupby(group_col)["hab_label"]
                 .max()   # a scene is "HAB" if it has ANY positive chip
                 .astype(int)
                 .reset_index(name="group_label"))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_train, idx_test = next(sss.split(grp_tbl[[ "group_label" ]], grp_tbl["group_label"]))
    train_groups = set(grp_tbl.iloc[idx_train][group_col].tolist())
    test_groups  = set(grp_tbl.iloc[idx_test][group_col].tolist())

    # map to row masks
    tr_mask = df[group_col].isin(train_groups)
    te_mask = df[group_col].isin(test_groups)

    # hard safety: no overlap
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(f"Grouping split failed, overlap groups found: {list(overlap)[:5]} ...")

    return tr_mask, te_mask, grp_tbl

def summarize_split(df, group_col, tr_mask, te_mask):
    def stats(mask, name):
        sub = df.loc[mask]
        grp = sub.groupby(group_col)["hab_label"].max()
        return {
            "rows": int(mask.sum()),
            "groups": int(grp.shape[0]),
            "groups_pos": int((grp>0).sum()),
            "groups_neg": int((grp==0).sum()),
            "rows_pos": int(sub["hab_label"].sum()),
            "rows_neg": int((1 - sub["hab_label"]).sum()),
        }
    return {"train": stats(tr_mask, "train"), "test": stats(te_mask, "test")}

def best_threshold_by_f1(y_true, y_prob):
    p, r, thr = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns thresholds length = len(p)-1
    f1 = (2*p*r)/(p+r+1e-9)
    k = np.nanargmax(f1)
    # clamp index for thresholds array
    k_thr = max(0, min(k-1, len(thr)-1))
    return float(thr[k_thr]), float(f1[k])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True,
                    help="Path or glob to *_hab.csv (after labeling).")
    ap.add_argument("--outdir", default="runs/hab_baseline_scene_split")
    ap.add_argument("--features", nargs="*", default=DEFAULT_FEATS,
                    help=f"Feature list; defaults: {DEFAULT_FEATS}")
    ap.add_argument("--group_by", default="scene_id",
                    help="Column to keep grouped during split (default: scene_id).")
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_labels(args.labels_csv)

    # basic hygiene
    if "hab_label" not in df.columns:
        raise SystemExit("No 'hab_label' column found. Did you run make_hab_labels.py?")

    # (optional) normalize FLH vs nFLH names
    if "flh" not in df.columns and "nflh" in df.columns:
        df = df.rename(columns={"nflh":"flh"})
    if "kd490" not in df.columns and "Kd_490" in df.columns:
        df = df.rename(columns={"Kd_490":"kd490"})

    feats = pick_existing_feats(df, args.features)
    for c in feats + ["hab_label"]:
        df[c] = to_num(df[c])

    df = df.dropna(subset=feats + ["hab_label"]).copy()
    df["hab_label"] = df["hab_label"].astype(int)

    # ---- grouped stratified split (by scene, no leakage) ----
    tr_mask, te_mask, grp_tbl = make_group_split(
        df, args.group_by, test_size=args.test_size, random_state=args.random_state
    )

    # sanity logs
    split_info = summarize_split(df, args.group_by, tr_mask, te_mask)
    print("Split summary (grouped by", args.group_by, "):")
    print(json.dumps(split_info, indent=2))

    Xtr, ytr = df.loc[tr_mask, feats].values, df.loc[tr_mask, "hab_label"].values
    Xte, yte = df.loc[te_mask, feats].values, df.loc[te_mask, "hab_label"].values

    # ---- model: scaler + LR (balanced) ----
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"))
    ])
    pipe.fit(Xtr, ytr)

    # ---- eval ----
    prob = pipe.predict_proba(Xte)[:, 1]
    auprc = average_precision_score(yte, prob)
    auroc = roc_auc_score(yte, prob)
    thr, best_f1 = best_threshold_by_f1(yte, prob)
    pred = (prob >= thr).astype(int)

    cm = confusion_matrix(yte, pred)  # [[TN, FP], [FN, TP]]
    print(f"\nAUPRC: {auprc:.3f}  AUROC: {auroc:.3f}  thr* (F1): {thr:.3f}  F1*: {best_f1:.3f}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:", cm.tolist())
    print("\nClassification report:\n", 
          pd.DataFrame(classification_report(yte, pred, output_dict=True)).T.round(3).to_string())

    # ---- feature weights (log-odds) ----
    clf = pipe.named_steps["clf"]
    coefs = clf.coef_.ravel()
    coef_table = sorted(zip(feats, coefs), key=lambda x: abs(x[1]), reverse=True)
    print("\nFeature weights (log-odds):")
    for name, w in coef_table:
        print(f"  {name:<12} {w:+.3f}")

    # ---- save model + metrics ----
    model_path = outdir / "logreg_baseline.joblib"
    joblib.dump({"pipe": pipe, "features": feats, "group_by": args.group_by}, model_path)
    metrics = {
        "auprc": float(auprc),
        "auroc": float(auroc),
        "threshold_f1": float(thr),
        "best_f1": float(best_f1),
        "confusion_matrix": cm.tolist(),
        "split_summary": split_info,
        "features_sorted": [{"name": n, "weight": float(w)} for n, w in coef_table]
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ saved model -> {model_path}")
    print(f"✓ saved metrics -> {outdir/'metrics.json'}")

if __name__ == "__main__":
    main()
