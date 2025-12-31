#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
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
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------- plotting ----------
def plot_pr(recall, precision, auprc, baseline, outpng):
    plt.figure(figsize=(5,4))
    if len(recall) and len(precision):
        plt.plot(recall, precision)
    plt.hlines(baseline, 0, 1, linestyles="--", alpha=0.5)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    b = "nan" if baseline is None else f"{baseline:.3f}"
    a = "nan" if not np.isfinite(auprc) else f"{auprc:.3f}"
    plt.title(f"PR (AUPRC={a}, baseline={b})")
    plt.tight_layout(); plt.savefig(outpng, dpi=160); plt.close()

def plot_roc(fpr, tpr, auroc, outpng):
    plt.figure(figsize=(5,4))
    if len(fpr) and len(tpr):
        plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], linestyle="--", alpha=0.5)
    title = f"ROC (AUROC={auroc:.3f})" if np.isfinite(auroc) else "ROC (AUROC=nan)"
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title)
    plt.tight_layout(); plt.savefig(outpng, dpi=160); plt.close()

# ---------- safe metrics ----------
def safe_pr_metrics(y_true, scores):
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

# ---------- month helpers ----------
MONTH_RE = re.compile(r"(20\d{2})[-_.]?(\d{2})")
def derive_month_key_from_scene(scene_id: str) -> str | None:
    m = MONTH_RE.search(str(scene_id))
    if not m: return None
    y, mo = m.group(1), m.group(2)
    try:
        yi, mi = int(y), int(mo)
        if 1 <= mi <= 12: return f"{yi:04d}-{mi:02d}"
    except Exception:
        pass
    return None

def month_ord(s: str | None) -> int:
    if not isinstance(s, str): return -10**9
    try:
        y, m = s.split("-")[:2]
        return int(y) * 12 + int(m)
    except Exception:
        return -10**9

# ---------- chronological split ----------
def chrono_split(df, month_col, group_col, test_frac_months=0.25,
                 min_pos_train=2, min_pos_test=2):
    uniq = sorted([m for m in df[month_col].unique() if isinstance(m, str)])
    if not uniq:
        return None

    import math, numpy as np
    total_months = len(uniq)
    k = max(1, int(math.ceil(test_frac_months * total_months)))

    for widen in range(0, total_months):
        test_months = uniq[-(k + widen):]
        te_mask = df[month_col].isin(test_months)
        tr_mask = ~te_mask

        # Remove any overlapping groups both ways
        te_groups = set(df.loc[te_mask, group_col].astype(str))
        tr_groups = set(df.loc[tr_mask, group_col].astype(str))
        both = te_groups & tr_groups
        if both:
            # drop overlaps from TRAIN (strictest choice for generalization)
            tr_mask = tr_mask & ~df[group_col].astype(str).isin(both)

        ytr = df.loc[tr_mask, "hab_label"].astype(int)
        yte = df.loc[te_mask, "hab_label"].astype(int)

        if len(ytr) == 0 or len(yte) == 0:
            continue
        if ytr.sum() >= min_pos_train and yte.sum() >= min_pos_test:
            tr_idx = np.where(tr_mask.values)[0]
            te_idx = np.where(te_mask.values)[0]

            # --- Debug prints to prove the split is clean
            te_groups = set(df.iloc[te_idx][group_col].astype(str))
            tr_groups = set(df.iloc[tr_idx][group_col].astype(str))
            inter = te_groups & tr_groups
            print(f"[chrono] months_test={test_months}")
            print(f"[chrono] groups: train={len(tr_groups)} test={len(te_groups)} overlap={len(inter)}")
            assert len(inter) == 0, "Train/Test share groups!"

            return tr_idx, te_idx, test_months

    # Fallback
    test_months = uniq[-2:] if len(uniq) >= 2 else uniq
    te_mask = df[month_col].isin(test_months)
    tr_mask = ~te_mask
    tr_idx = np.where(tr_mask.values)[0]
    te_idx = np.where(te_mask.values)[0]
    print(f"[warn] fallback chrono split (test months={test_months})")
    return tr_idx, te_idx, test_months



# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--outdir", default="runs/hab_baseline_timecv")
    # You can pass features explicitly OR clone them from an existing joblib:
    ap.add_argument("--features", nargs="+", default=None,
                    help="Explicit feature list (ignored if --clone_features_from provided)")
    ap.add_argument("--clone_features_from", default=None,
                    help="Path to an old model.joblib to reuse its features list (A/B/C)")
    ap.add_argument("--group_by", default="scene_id")
    ap.add_argument("--id_col", default="tile")
    ap.add_argument("--month_key", default="month_key",
                    help="Column with YYYY-MM; inferred from scene_id if missing")
    ap.add_argument("--test_size", type=float, default=0.25,
                    help="Approx fraction of months held out as TEST (chronological)")
    ap.add_argument("--cv_folds", type=int, default=5, help="inner GroupKFold on TRAIN (diagnostic)")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--min_pos_train", type=int, default=2)
    ap.add_argument("--min_pos_test", type=int, default=2)
    ap.add_argument("--tune_C", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ----- features -----
    if args.clone_features_from:
        pack_old = joblib.load(args.clone_features_from)
        feats = list(pack_old["features"])
        print(f"[features] cloned from {args.clone_features_from}: {feats}")
    else:
        if not args.features:
            raise SystemExit("Provide --features or --clone_features_from <old_model.joblib>")
        feats = list(args.features)
        print(f"[features] using explicit list: {feats}")

    # ----- load & prepare -----
    df = pd.read_csv(args.labels_csv).copy()
    # derive month_key if missing
    if args.month_key not in df.columns:
        mk = df[args.group_by].astype(str).map(derive_month_key_from_scene)
        if mk.isna().any():
            raise SystemExit(f"Could not derive {args.month_key} from {args.group_by} for all rows.")
        df[args.month_key] = mk
    df["_mord"] = df[args.month_key].map(month_ord)

    need_cols = feats + [args.group_by, args.month_key, "hab_label"]
    for c in need_cols:
        if c not in df.columns:
            raise SystemExit(f"Column missing: {c}")

    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feats + ["hab_label"]).copy()

    y  = df["hab_label"].astype(int).values
    X  = df[feats].values
    G  = df[args.group_by].astype(str).values
    prevalence = float(np.mean(y)) if len(y) else 0.0

    # ----- chronological split (fallback to grouped shuffle if necessary) -----
    cs = chrono_split(df, args.month_key, args.group_by,
                      test_frac_months=args.test_size,
                      min_pos_train=args.min_pos_train,
                      min_pos_test=args.min_pos_test)
    if cs is None:
        print("[warn] chrono split failed to meet min positives; falling back to GroupShuffleSplit.")
        tr, te = next(GroupShuffleSplit(test_size=args.test_size, random_state=args.random_state).split(X, y, groups=G))
        test_months = []
    else:
        tr, te, test_months = cs

    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    Gtr, Gte = G[tr], G[te]
    print(f"[Split] TRAIN n={len(ytr)} (pos={int(ytr.sum())}) | TEST n={len(yte)} (pos={int(yte.sum())}) "
          f"| months_test={test_months if test_months else 'GSS fallback'}")

    # ----- (optional) very light C tuning on TRAIN -----
    chosen_C = 1.0
    if args.tune_C:
        gkf = GroupKFold(n_splits=min(args.cv_folds, len(np.unique(Gtr))))
        best_C, best_score = 1.0, -np.inf
        for C in (0.25, 0.5, 1.0, 2.0, 4.0):
            scores = []
            for tri, vai in gkf.split(Xtr, ytr, groups=Gtr):
                pipe_try = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=C))
                ])
                pipe_try.fit(Xtr[tri], ytr[tri])
                p_va = pipe_try.predict_proba(Xtr[vai])[:, 1]
                try:
                    scores.append(average_precision_score(ytr[vai], p_va))
                except Exception:
                    scores.append(np.nan)
            m = np.nanmean(scores)
            if m > best_score:
                best_score, chosen_C = m, C
        print(f"[tune] chosen C={chosen_C} (cv mean AUPRC={best_score:.3f})")

    # ----- final fit -----
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=chosen_C))
    ])
    pipe.fit(Xtr, ytr)

    # ----- test metrics -----
    p_te = pipe.predict_proba(Xte)[:, 1]
    auprc, prec, rec, thr_pr = safe_pr_metrics(yte, p_te)
    auroc, fpr, tpr, _ = safe_roc_metrics(yte, p_te)

    if len(thr_pr):
        f1s = (2 * prec * rec / (prec + rec + 1e-12))[:-1]
        best_i = int(np.argmax(f1s))
        best_thr = float(thr_pr[best_i])
    else:
        best_thr = 0.5

    yhat = (p_te >= best_thr).astype(int)
    cm = confusion_matrix(yte, yhat).tolist()
    rep = classification_report(yte, yhat, digits=3, zero_division=0)

    print(f"AUPRC: {auprc:.3f}  AUROC: {auroc:.3f}  thr*: {best_thr:.3f}  (baseline={prevalence:.3f})")
    print("Confusion matrix [[TN,FP],[FN,TP]]:", cm)
    print(rep)

    # ----- diagnostic GroupKFold on TRAIN -----
    cv_scores = []
    gkf = GroupKFold(n_splits=min(args.cv_folds, len(np.unique(Gtr))))
    for tri, vai in gkf.split(Xtr, ytr, groups=Gtr):
        pipe_cv = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=chosen_C))
        ])
        pipe_cv.fit(Xtr[tri], ytr[tri])
        p_va = pipe_cv.predict_proba(Xtr[vai])[:, 1]
        a_pr, _, _, _ = safe_pr_metrics(ytr[vai], p_va)
        a_roc, _, _, _ = safe_roc_metrics(ytr[vai], p_va)
        cv_scores.append({"auprc": float(a_pr), "auroc": float(a_roc)})

    # ----- save -----
    joblib.dump({"pipe": pipe, "features": feats}, outdir / "model.joblib")

    metrics = {
        "auprc": float(auprc),
        "auroc": float(auroc),
        "threshold": float(best_thr),
        "confusion_matrix": cm,
        "test_pos": int(yte.sum()),
        "test_neg": int((1 - yte).sum()),
        "prevalence": prevalence,
        "cv": cv_scores,
        "features": feats,
        "group_by": args.group_by,
        "month_key": args.month_key,
        "months_test": test_months,
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_pr(rec, prec, auprc, prevalence, outdir / "pr_curve.png")
    plot_roc(fpr, tpr, auroc, outdir / "roc_curve.png")
    print(f"✓ model -> {outdir/'model.joblib'}")
    print(f"✓ metrics -> {outdir/'metrics.json'}")
    print(f"✓ saved PR/ROC curves")

if __name__ == "__main__":
    main()
