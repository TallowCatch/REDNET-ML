#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

def try_metrics_json(p: Path):
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None

    # common keys people use
    for k in [
        "threshold", "best_threshold", "selected_threshold",
        "decision_threshold", "threshold_selected",
        "threshold_prec_at_recall", "prec_at_recall_threshold",
    ]:
        if k in d and isinstance(d[k], (int, float)):
            return float(d[k])

    # sometimes nested
    for k in ["eval", "metrics", "thresholding", "selection"]:
        if k in d and isinstance(d[k], dict):
            sub = d[k]
            for kk in sub:
                if "threshold" in kk and isinstance(sub[kk], (int, float)):
                    return float(sub[kk])

    return None

def threshold_prec_at_recall(y_true, y_score, min_recall: float):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    # sort by score desc
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)
    pos = max(1, int((y_true == 1).sum()))
    recall = tp / pos
    precision = tp / np.maximum(1, (tp + fp))

    ok = np.where(recall >= min_recall)[0]
    if len(ok) == 0:
        # if impossible, return highest-score threshold
        return float(y_score[0])

    # choose threshold that maximizes precision under recall>=min_recall
    best_i = ok[np.argmax(precision[ok])]
    return float(y_score[best_i])

def try_predictions_csv(p: Path, min_recall: float):
    df = pd.read_csv(p)

    # try common column names
    y_cols = ["y_true", "label", "target", "hab_label", "hab_label_final", "hab_label_final2"]
    s_cols = ["y_prob", "prob", "score", "hab_prob", "p", "pred_prob"]

    ycol = next((c for c in y_cols if c in df.columns), None)
    scol = next((c for c in s_cols if c in df.columns), None)
    if ycol is None or scol is None:
        return None

    # if it includes train/test flag, use test only
    for split_col in ["split", "is_test", "fold_split"]:
        if split_col in df.columns:
            # handle common encodings
            if df[split_col].dtype == object:
                test_df = df[df[split_col].astype(str).str.contains("test", case=False, na=False)]
            else:
                test_df = df[df[split_col] == 1]
            if len(test_df) >= 10:
                df = test_df
            break

    # drop nans
    df = df[[ycol, scol]].dropna()
    if len(df) == 0:
        return None

    return threshold_prec_at_recall(df[ycol].values, df[scol].values, min_recall=min_recall)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv_dir", required=True, help="e.g. runs/fusion/fusion_alllabels_cv5")
    ap.add_argument("--min_recall", type=float, default=0.60)
    args = ap.parse_args()

    cv_dir = Path(args.cv_dir)

    # 1) try metrics json first (fast + authoritative)
    jsons = sorted(cv_dir.glob("metrics_cv*.json"))
    for j in jsons:
        t = try_metrics_json(j)
        if t is not None:
            print(t)
            return

    # 2) fallback: compute from predictions csv
    preds = sorted(cv_dir.glob("predictions_cv*.csv"))
    for p in preds:
        t = try_predictions_csv(p, min_recall=args.min_recall)
        if t is not None:
            print(t)
            return

    raise SystemExit("Could not find threshold in metrics_cv*.json or infer from predictions_cv*.csv")

if __name__ == "__main__":
    main()
