#!/usr/bin/env python3
# scripts/fusion/make_report_artifacts_v2.py

"""
Generate paper-ready artifacts (tables + plots) for REDNET-ML fusion datasets.

Outputs (PNG) into --outdir:
  1) table_ranges_overall.png         (1 row per factor: Range + Mean±SD)
     OR table_ranges_by_group_wide.png (1 row per factor, grouped columns; optional)
  2) confusion_matrix.png             (paper-style colored matrix like your example)
  3) pr_curve.png                     (Precision-Recall)
  4) roc_curve.png                    (ROC)
  5) calibration_curve.png            (Reliability diagram)
  6) rf_trees_curve.png               (AUPRC vs n_estimators)
  7) feature_importance.png           (CatBoost or RF)
  8) shap_summary.png                 (optional, if shap installed)
  9) metrics.json                     (all metrics saved)

It does CV with StratifiedGroupKFold (if available) to avoid scene leakage.
"""

import argparse, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedShuffleSplit

# Prefer StratifiedGroupKFold if present
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except Exception:
    HAS_SGKF = False

from sklearn.ensemble import RandomForestClassifier


def _try_import_catboost():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except Exception:
        return None


# ----------------------------- util -----------------------------

def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

def _format_mean_sd(mu, sd):
    if not np.isfinite(mu):
        return "nan"
    if not np.isfinite(sd):
        sd = 0.0
    return f"{mu:.3f} ± {sd:.3f}"

def _format_range(vmin, vmax):
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        return "nan"
    return f"{vmin:.3f}–{vmax:.3f}"

def _make_table_image(df_table: pd.DataFrame, outpng: Path, title: str):
    # Wider + taller depending on cols/rows so it doesn’t become a skinny “list”
    nrows = len(df_table)
    ncols = len(df_table.columns)
    fig_w = max(10.0, 1.6 * ncols)
    fig_h = max(3.2, 0.55 * (nrows + 2))

    plt.figure(figsize=(fig_w, fig_h))
    plt.axis("off")
    plt.title(title, fontsize=14, pad=14)

    tbl = plt.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.35)

    # Light header emphasis
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_linewidth(1.0)
        else:
            cell.set_linewidth(0.6)

    _savefig(outpng)

def _safe_cols(df, cols):
    return [c for c in cols if c in df.columns]

def _to_num(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _pick_threshold_prec_at_recall(y_tr, p_tr, min_recall=0.60):
    prec, rec, thr = precision_recall_curve(y_tr, p_tr)
    ok = np.where(rec[:-1] >= float(min_recall))[0]
    if len(ok) == 0:
        return 0.5, "fallback(0.5)"
    best = ok[np.argmax(prec[:-1][ok])]
    return float(thr[best]), f"prec@recall>={min_recall:.2f}"


# ----------------------------- plots -----------------------------

def _plot_pr(y, p, outpng: Path):
    prec, rec, _ = precision_recall_curve(y, p)
    auprc = average_precision_score(y, p) if y.sum() > 0 else 0.0
    base = float(np.mean(y)) if len(y) else 0.0

    plt.figure(figsize=(6.2, 4.8))
    plt.plot(rec, prec, lw=2)
    plt.hlines(base, 0, 1, linestyles="--", alpha=0.6)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall (AUPRC={auprc:.3f}, baseline={base:.3f})")
    plt.grid(True, linestyle="--", alpha=0.35)
    _savefig(outpng)

def _plot_roc(y, p, outpng: Path):
    if len(np.unique(y)) < 2:
        fpr, tpr = np.array([0, 1]), np.array([0, 1])
        auroc = float("nan")
    else:
        fpr, tpr, _ = roc_curve(y, p)
        auroc = roc_auc_score(y, p)

    plt.figure(figsize=(6.2, 4.8))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], "--", alpha=0.6)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC (AUROC={auroc:.3f})")
    plt.grid(True, linestyle="--", alpha=0.35)
    _savefig(outpng)

def _plot_calibration(y, p, outpng: Path, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    plt.figure(figsize=(6.2, 4.8))
    plt.plot(mean_pred, frac_pos, marker="o", lw=2)
    plt.plot([0, 1], [0, 1], "--", alpha=0.6)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curve (Reliability)")
    plt.grid(True, linestyle="--", alpha=0.35)
    _savefig(outpng)

def _plot_confusion_matrix_paperstyle(
    TN, FP, FN, TP,
    outpng: Path,
    neg_name="Non Bloom",
    pos_name="Bloom",
    title="Confusion matrix",
):
    """
    Paper-style 2x2 heatmap:
      rows = Actual [neg, pos]
      cols = Pred   [neg, pos]
    Cells show big counts + row-normalized rates (TN/FP on neg row, FN/TP on pos row).
    """
    # Row-normalized rates
    neg_total = TN + FP
    pos_total = TP + FN
    tn_rate = (TN / neg_total) if neg_total > 0 else 0.0
    fp_rate = (FP / neg_total) if neg_total > 0 else 0.0
    fn_rate = (FN / pos_total) if pos_total > 0 else 0.0
    tp_rate = (TP / pos_total) if pos_total > 0 else 0.0

    # Manual pastel cell colors (correct=greenish, error=reddish)
    # layout: [[TN, FP],
    #          [FN, TP]]
    cell_rgb = np.array([
        [[0.86, 0.93, 0.86], [0.98, 0.86, 0.86]],
        [[0.98, 0.86, 0.86], [0.86, 0.93, 0.86]],
    ])

    plt.figure(figsize=(8.6, 6.4))
    ax = plt.gca()
    ax.imshow(cell_rgb, aspect="equal")

    # Axis labels like your example
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([neg_name, pos_name], fontsize=12)
    ax.set_yticklabels([neg_name, pos_name], fontsize=12)
    ax.set_xlabel("Predicted Class", fontsize=13, labelpad=16)
    ax.set_ylabel("Actual Class", fontsize=13, labelpad=16)
    ax.set_title(title, fontsize=14, pad=14)

    # Grid lines for crisp 2x2 blocks
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate counts + rates
    # (0,0)=TN, (0,1)=FP, (1,0)=FN, (1,1)=TP
    annotations = [
        (0, 0, f"{TN}", f"TN Rate = {tn_rate*100:.1f}%"),
        (0, 1, f"{FP}", f"FP Rate = {fp_rate*100:.1f}%"),
        (1, 0, f"{FN}", f"FN Rate = {fn_rate*100:.1f}%"),
        (1, 1, f"{TP}", f"TP Rate = {tp_rate*100:.1f}%"),
    ]
    for r, c, big, small in annotations:
        ax.text(c, r, big, ha="center", va="center", fontsize=28, fontweight="bold")
        ax.text(c, r + 0.28, small, ha="center", va="center", fontsize=11)

    # Clean frame
    for spine in ax.spines.values():
        spine.set_visible(False)

    _savefig(outpng)


# ----------------------------- table builders -----------------------------

def _build_table_overall(df: pd.DataFrame, table_cols):
    # Columns where 0 is typically a placeholder / missing-fill and should not define the min
    ZERO_EXCLUDE = {"sst", "chlor_a", "kd490", "rednir_mean"}

    rows = []
    for col in table_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.dropna()

        # exclude zeros only for selected columns
        if col in ZERO_EXCLUDE:
            s = s[s != 0]

        vals = s.values
        if len(vals) == 0:
            continue

        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        rows.append({
            "Parameter": col,
            "Range": _format_range(np.min(vals), np.max(vals)),
            "Mean ± SD": _format_mean_sd(mu, sd),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out[["Parameter", "Range", "Mean ± SD"]]
    return out





# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Your fusion dataset CSV")
    ap.add_argument("--outdir", default="runs/fusion/report_artifacts_v2")
    ap.add_argument("--label_col", default="hab_label_final2")
    ap.add_argument("--id_col", default="tile")
    ap.add_argument("--group_by", default="scene_id")

    ap.add_argument("--model", choices=["catboost", "rf"], default="catboost")
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--min_recall", type=float, default=0.60)

    # Table factors (script filters to what exists)
    ap.add_argument("--table_cols", nargs="*", default=[
    "chlor_a", "kd490", "nflh", "sst", "fai_mean", "ndwi_mean", "rednir_mean"
    ])

    

    # Confusion-matrix class names
    ap.add_argument("--neg_name", default="Non Bloom")
    ap.add_argument("--pos_name", default="Bloom")

    # Candidate feature columns (script auto-filters)
    ap.add_argument("--feature_cols", nargs="*", default=[
        # in-tab detectors
        "p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med", "p_tab",
        # env base
        "fai_mean", "rednir_mean", "ndwi_mean", "kd490", "chlor_a", "nflh", "sst",
        "month_sin", "month_cos", "ndwi_std", "rednir_std",
        # derived env
        "sst_anom", "sst_anom_z", "log_kd490", "log_chlor_a", "log_nflh",
        "ratio_chl_kd", "chl_times_nflh", "ratio_nflh_kd",
        # interactions (if present)
        "sst_anom_x_chlor_a", "sst_anom_x_nflh", "sst_anom_x_fai_mean", "sst_anom_x_kd490",
        "sst_anom_x_month_sin", "sst_anom_x_month_cos",
    ])

    # RF sweep for “performance vs trees”
    ap.add_argument("--rf_trees", nargs="*", type=int, default=[25, 50, 100, 200, 400])

    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.label_col not in df.columns:
        raise SystemExit(f"Missing label_col '{args.label_col}' in CSV")

    has_groups = args.group_by in df.columns

    # clean label
    df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce").fillna(0).astype(int)
    df[args.label_col] = (df[args.label_col] > 0).astype(int)

    # ---------------- Table (1 row per factor) ----------------
    table_cols = _safe_cols(df, args.table_cols)

    if len(table_cols) == 0:
        print("[warn] Skipping Table-1 image (no table_cols found in CSV).")
    else:
        table_df = _build_table_overall(df, table_cols)
        if not table_df.empty:
            _make_table_image(
                table_df,
                outdir / "table_ranges_overall.png",
                title="Table: Range of data used",
            )


    # ---------------- CV evaluation ----------------
    feats = _safe_cols(df, args.feature_cols)
    if len(feats) == 0:
        raise SystemExit("No usable feature columns found. Pass --feature_cols with columns that exist in your CSV.")

    work = _to_num(df.copy(), feats)
    work[feats] = work[feats].fillna(0.0)

    X = work[feats].values
    y = work[args.label_col].values.astype(int)

    if has_groups:
        groups = work[args.group_by].astype(str).values
    else:
        groups = np.array(["nogroup"] * len(work))

    # choose splitter
    if args.cv_folds > 1 and HAS_SGKF and has_groups:
        splitter = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(np.zeros(len(y)), y, groups=groups))
    else:
        sss = StratifiedShuffleSplit(n_splits=args.cv_folds, test_size=0.25, random_state=args.seed)
        splits = list(sss.split(np.zeros(len(y)), y))
        print("[warn] Using StratifiedShuffleSplit fallback (no StratifiedGroupKFold / no groups).")

    CatBoost = _try_import_catboost()
    if args.model == "catboost" and CatBoost is None:
        print("[warn] CatBoost not available; falling back to RandomForest.")
        args.model = "rf"

    fold_metrics = []
    all_test_p = []
    all_test_y = []

    last_importance = None
    last_model_type = None

    for fold_id, (tr_idx, te_idx) in enumerate(splits, 1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        # internal early-stop split for CatBoost (from TRAIN only)
        cal_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed + fold_id)
        fit_idx, val_idx = next(cal_split.split(np.zeros(len(y_tr)), y_tr))
        X_fit, y_fit = X_tr[fit_idx], y_tr[fit_idx]
        X_val, y_val = X_tr[val_idx], y_tr[val_idx]

        if args.model == "catboost":
            model = CatBoost(
                iterations=1200,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=6.0,
                loss_function="Logloss",
                eval_metric="Logloss",
                custom_metric="AUC",
                random_seed=args.seed + fold_id,
                auto_class_weights="Balanced",
                od_type="Iter",
                od_wait=80,
                verbose=False,
            )
            model.fit(X_fit, y_fit, eval_set=(X_val, y_val), use_best_model=True)
            p_tr = model.predict_proba(X_tr)[:, 1]
            p_te = model.predict_proba(X_te)[:, 1]
            last_importance = model.get_feature_importance()
            last_model_type = "catboost"
        else:
            model = RandomForestClassifier(
                n_estimators=300,
                random_state=args.seed + fold_id,
                class_weight="balanced_subsample",
                n_jobs=-1,
                max_depth=None,
                min_samples_leaf=1,
            )
            model.fit(X_tr, y_tr)
            p_tr = model.predict_proba(X_tr)[:, 1]
            p_te = model.predict_proba(X_te)[:, 1]
            last_importance = model.feature_importances_
            last_model_type = "rf"

        thr, how = _pick_threshold_prec_at_recall(y_tr, p_tr, min_recall=args.min_recall)
        yhat = (p_te >= thr).astype(int)

        cm = confusion_matrix(y_te, yhat, labels=[0, 1])
        TN, FP = int(cm[0, 0]), int(cm[0, 1])
        FN, TP = int(cm[1, 0]), int(cm[1, 1])

        auprc = average_precision_score(y_te, p_te) if y_te.sum() > 0 else 0.0
        auroc = roc_auc_score(y_te, p_te) if len(np.unique(y_te)) == 2 else float("nan")

        fold_metrics.append({
            "fold": fold_id,
            "threshold": float(thr),
            "threshold_policy": how,
            "test_pos": int(y_te.sum()),
            "test_total": int(len(y_te)),
            "AUPRC": float(auprc),
            "AUROC": float(auroc) if np.isfinite(auroc) else None,
            "TN": TN, "FP": FP, "FN": FN, "TP": TP,
        })

        all_test_p.append(p_te)
        all_test_y.append(y_te)

        print(f"[fold {fold_id}] AUPRC={auprc:.3f} AUROC={(auroc if np.isfinite(auroc) else float('nan')):.3f} thr={thr:.3f} ({how})")

    all_test_p = np.concatenate(all_test_p)
    all_test_y = np.concatenate(all_test_y)

    overall_auprc = average_precision_score(all_test_y, all_test_p) if all_test_y.sum() > 0 else 0.0
    overall_auroc = roc_auc_score(all_test_y, all_test_p) if len(np.unique(all_test_y)) == 2 else float("nan")

    pooled_thr, pooled_how = _pick_threshold_prec_at_recall(all_test_y, all_test_p, min_recall=args.min_recall)
    pooled_yhat = (all_test_p >= pooled_thr).astype(int)
    pooled_cm = confusion_matrix(all_test_y, pooled_yhat, labels=[0, 1])
    pooled_TN, pooled_FP = int(pooled_cm[0, 0]), int(pooled_cm[0, 1])
    pooled_FN, pooled_TP = int(pooled_cm[1, 0]), int(pooled_cm[1, 1])

    # ---------------- plots ----------------
    _plot_confusion_matrix_paperstyle(
        pooled_TN, pooled_FP, pooled_FN, pooled_TP,
        outpng=outdir / "confusion_matrix.png",
        neg_name=args.neg_name,
        pos_name=args.pos_name,
        title=f"Confusion Matrix (pooled CV) | thr={pooled_thr:.3f} ({pooled_how})",
    )
    _plot_pr(all_test_y, all_test_p, outdir / "pr_curve.png")
    _plot_roc(all_test_y, all_test_p, outdir / "roc_curve.png")
    _plot_calibration(all_test_y, all_test_p, outdir / "calibration_curve.png", n_bins=10)

    # Feature importance plot
    if last_importance is not None:
        imp = np.asarray(last_importance, dtype=float)
        idx = np.argsort(imp)[::-1][:25]
        plt.figure(figsize=(8.2, 5.6))
        plt.barh([feats[i] for i in idx][::-1], imp[idx][::-1])
        plt.title(f"Top feature importances ({last_model_type})")
        plt.xlabel("Importance")
        plt.grid(True, axis="x", linestyle="--", alpha=0.35)
        _savefig(outdir / "feature_importance.png")

    # RF “performance vs trees” curve (single split, paper-friendly)
    rf_points = []
    for n_trees in args.rf_trees:
        rf = RandomForestClassifier(
            n_estimators=int(n_trees),
            random_state=args.seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
        tr_idx, te_idx = next(sss.split(np.zeros(len(y)), y))
        rf.fit(X[tr_idx], y[tr_idx])
        p = rf.predict_proba(X[te_idx])[:, 1]
        auprc = average_precision_score(y[te_idx], p) if y[te_idx].sum() > 0 else 0.0
        rf_points.append((int(n_trees), float(auprc)))

    xs = [a for a, _ in rf_points]
    ys = [b for _, b in rf_points]
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(xs, ys, marker="o", lw=2)
    plt.xlabel("n_estimators (trees)")
    plt.ylabel("AUPRC")
    plt.title("Random Forest: performance vs trees")
    plt.grid(True, linestyle="--", alpha=0.35)
    _savefig(outdir / "rf_trees_curve.png")

    # Optional SHAP for CatBoost
    try:
        import shap  # type: ignore
        if args.model == "catboost" and CatBoost is not None:
            model_full = CatBoost(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=6.0,
                loss_function="Logloss",
                random_seed=args.seed,
                auto_class_weights="Balanced",
                verbose=False,
            )
            model_full.fit(X, y)
            explainer = shap.TreeExplainer(model_full)
            n = min(1000, len(X))
            Xs = X[:n]
            shap_values = explainer.shap_values(Xs)
            plt.figure()
            shap.summary_plot(shap_values, pd.DataFrame(Xs, columns=feats), show=False, max_display=20)
            _savefig(outdir / "shap_summary.png")
    except Exception:
        pass

    # Save metrics JSON
    out = {
        "csv": str(args.csv),
        "label_col": args.label_col,
        "group_by": args.group_by,
        "model": args.model,
        "cv_folds": args.cv_folds,
        "min_recall": args.min_recall,
        "features_used": feats,
        "pooled": {
            "AUPRC": float(overall_auprc),
            "AUROC": float(overall_auroc) if np.isfinite(overall_auroc) else None,
            "threshold": float(pooled_thr),
            "threshold_policy": pooled_how,
            "TN": pooled_TN, "FP": pooled_FP, "FN": pooled_FN, "TP": pooled_TP,
        },
        "folds": fold_metrics,
        "rf_curve": [{"n_estimators": a, "auprc": b} for a, b in rf_points],
        "table": {
            "table_cols": table_cols,
        }
    }
    (outdir / "metrics.json").write_text(json.dumps(out, indent=2))

    print(f"\n✓ Saved artifacts to: {outdir.resolve()}")
    print("  - table_ranges_overall.png")
    print("  - confusion_matrix.png")
    print("  - pr_curve.png / roc_curve.png / calibration_curve.png")
    print("  - rf_trees_curve.png / feature_importance.png")
    print("  - metrics.json")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
