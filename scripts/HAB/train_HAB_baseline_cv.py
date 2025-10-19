#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, roc_curve, classification_report,
    confusion_matrix
)
import joblib
import matplotlib.pyplot as plt


# ------------------------- plotting helpers -------------------------
def plot_pr(recall, precision, auprc, baseline, outpng):
    plt.figure(figsize=(5,4))
    plt.plot(recall, precision)
    # prevalence baseline
    plt.hlines(baseline, 0, 1, linestyles="--", alpha=0.5)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR (AUPRC={auprc:.3f}, baseline={baseline:.3f})")
    plt.tight_layout(); plt.savefig(outpng, dpi=160); plt.close()

def plot_roc(fpr, tpr, auroc, outpng):
    plt.figure(figsize=(5,4))
    if len(fpr) and len(tpr):
        plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], linestyle="--", alpha=0.5)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    title = f"ROC (AUROC={auroc:.3f})" if np.isfinite(auroc) else "ROC (AUROC=nan)"
    plt.title(title)
    plt.tight_layout(); plt.savefig(outpng, dpi=160); plt.close()


# ------------------------- metrics (safe) -------------------------
def safe_pr_metrics(y_true, scores):
    """Return (auprc, precision, recall, thresholds) safely."""
    try:
        auprc = float(average_precision_score(y_true, scores))
    except Exception:
        auprc = float("nan")
    try:
        prec, rec, thr = precision_recall_curve(y_true, scores)
    except Exception:
        prec, rec, thr = np.array([0.]), np.array([0.]), np.array([])
    return auprc, prec, rec, thr

def safe_roc_metrics(y_true, scores):
    try:
        auroc = float(roc_auc_score(y_true, scores))
        fpr, tpr, thr = roc_curve(y_true, scores)
    except Exception:
        auroc = float("nan")
        fpr, tpr, thr = np.array([]), np.array([]), np.array([])
    return auroc, fpr, tpr, thr


# ------------------------- grouped split with min positives -------------------------
def grouped_split_with_min_pos(X, y, G, test_size, seed,
                               min_pos_train=2, min_pos_test=2, max_tries=500):
    """
    Keep trying GroupShuffleSplit with incrementing seeds until
    both train and test have at least the requested number of positives.
    """
    for t in range(max_tries):
        rs = int(seed + t)
        gss = GroupShuffleSplit(test_size=test_size, random_state=rs)
        tr, te = next(gss.split(X, y, groups=G))
        if (y[tr].sum() >= min_pos_train) and (y[te].sum() >= min_pos_test):
            return tr, te, rs, t + 1
    # Fallback to a single split if we never meet constraints
    tr, te = next(GroupShuffleSplit(test_size=test_size, random_state=seed).split(X, y, groups=G))
    return tr, te, seed, 1


# ------------------------- (optional) lightweight C tuning -------------------------
def tune_logreg_C(X, y, groups, cand_C=(0.1, 0.5, 1.0, 2.0, 5.0), folds=5):
    gkf = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    best_C, best_score = 1.0, -np.inf
    for C in cand_C:
        scores = []
        for tr, va in gkf.split(X, y, groups=groups):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=C))
            ])
            pipe.fit(X[tr], y[tr])
            p = pipe.predict_proba(X[va])[:, 1]
            try:
                s = average_precision_score(y[va], p)
            except Exception:
                s = -np.inf
            scores.append(s)
        m = np.nanmean(scores)
        if m > best_score:
            best_score, best_C = m, C
    return best_C, best_score


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--outdir", default="runs/hab_baseline_no_leak")
    ap.add_argument("--features", nargs="+",
                    default=["fai_mean","rednir_mean","ndwi_mean","month_sin","month_cos"])
    ap.add_argument("--group_by", default="scene_id", help="column for grouping")
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--cv_folds", type=int, default=5)
    # new safety knobs (do not change your A/B/C behaviour unless you want to)
    ap.add_argument("--min_pos_train", type=int, default=2)
    ap.add_argument("--min_pos_test", type=int, default=2)
    ap.add_argument("--tune_C", action="store_true", help="light CV to pick LogisticRegression C")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ----- load & validate -----
    df = pd.read_csv(args.labels_csv)
    for c in args.features + [args.group_by, "hab_label"]:
        if c not in df.columns:
            raise SystemExit(f"Column missing: {c}")

    # numeric features + drop NA rows
    for c in args.features:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=args.features + ["hab_label"]).copy()

    y  = df["hab_label"].astype(int).values
    X  = df[args.features].values
    G  = df[args.group_by].astype(str).values

    # prevalence (PR baseline)
    prevalence = float(np.mean(y)) if len(y) else 0.0

    # ----- robust grouped split with min positives -----
    tr_idx, te_idx, used_seed, tries = grouped_split_with_min_pos(
        X, y, G, test_size=args.test_size, seed=args.random_state,
        min_pos_train=args.min_pos_train, min_pos_test=args.min_pos_test, max_tries=500
    )
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]
    Gtr, Gte = G[tr_idx], G[te_idx]

    print(f"[Split] train rows={len(ytr)} (pos={int(ytr.sum())}) | "
          f"test rows={len(yte)} (pos={int(yte.sum())}) | seed={used_seed} (tries={tries})")

    # ----- (optional) C tuning on train -----
    chosen_C = 1.0
    cv_summary = []
    if args.tune_C:
        chosen_C, cv_best = tune_logreg_C(Xtr, ytr, Gtr, folds=args.cv_folds)
        cv_summary.append({"best_C": chosen_C, "cv_mean_auprc": float(cv_best)})

    # ----- final model -----
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=chosen_C))
    ])
    pipe.fit(Xtr, ytr)

    # ----- test metrics -----
    p_te = pipe.predict_proba(Xte)[:, 1]

    auprc, prec, rec, thr_pr = safe_pr_metrics(yte, p_te)
    auroc, fpr, tpr, thr_roc = safe_roc_metrics(yte, p_te)

    # pick threshold for max F1 (guard against empty thr list)
    if len(thr_pr):
        f1s = (2 * prec * rec / (prec + rec + 1e-12))[:-1]
        best_i = int(np.argmax(f1s))
        best_thr = float(thr_pr[best_i])
    else:
        best_thr = 0.5

    yhat = (p_te >= best_thr).astype(int)
    cm = confusion_matrix(yte, yhat).tolist()
    rep = classification_report(yte, yhat, digits=3)

    print(f"AUPRC: {auprc:.3f}  AUROC: {auroc:.3f}  thr*: {best_thr:.3f}  (baseline={prevalence:.3f})")
    print("Confusion matrix [[TN,FP],[FN,TP]]:", cm)
    print(rep)

    # ----- GroupKFold CV on train (diagnostic only, unchanged) -----
    cv_scores = []
    gkf = GroupKFold(n_splits=min(args.cv_folds, len(np.unique(Gtr))))
    for tr, va in gkf.split(Xtr, ytr, groups=Gtr):
        pipe_cv = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=chosen_C))
        ])
        pipe_cv.fit(Xtr[tr], ytr[tr])
        p_va = pipe_cv.predict_proba(Xtr[va])[:, 1]
        a_pr, _, _, _ = safe_pr_metrics(ytr[va], p_va)
        a_roc, _, _, _ = safe_roc_metrics(ytr[va], p_va)
        cv_scores.append({"auprc": float(a_pr), "auroc": float(a_roc)})

    # ----- save artifacts -----
    joblib.dump({"pipe": pipe, "features": args.features}, outdir / "model.joblib")

    metrics = {
        "auprc": float(auprc),
        "auroc": float(auroc),
        "threshold": float(best_thr),
        "confusion_matrix": cm,
        "test_pos": int(yte.sum()),
        "test_neg": int((1 - yte).sum()),
        "prevalence": prevalence,
        "cv": cv_scores,
        "cv_tuning": cv_summary,
        "features": args.features,
        "group_by": args.group_by,
        "split_seed": int(used_seed),
        "tries_to_meet_min_pos": int(tries)
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    pr_png  = outdir / "pr_curve.png"
    roc_png = outdir / "roc_curve.png"
    plot_pr(rec, prec, auprc, prevalence, pr_png)
    plot_roc(fpr, tpr, auroc, roc_png)
    print(f"✓ saved {pr_png} and {roc_png}")
    print(f"✓ model -> {outdir/'model.joblib'}")
    print(f"✓ metrics -> {outdir/'metrics.json'}")


if __name__ == "__main__":
    main()
