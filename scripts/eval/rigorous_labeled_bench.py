#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# sklearn baselines + metrics
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    accuracy_score,
    cohen_kappa_score,
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _as_float(x) -> np.ndarray:
    return pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)

def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def safe_roc_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))

def safe_pr_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))

def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray, n_grid: int = 200) -> float:
    qs = np.linspace(0.0, 1.0, n_grid)
    best_thr = 0.5
    best_f1 = -1.0
    for thr in qs:
        pred = (y_score >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr

def eval_threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (y_score >= thr).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )

    acc = accuracy_score(y_true, pred)
    kappa = cohen_kappa_score(y_true, pred)

    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "accuracy": float(acc),
        "kappa": float(kappa),
    }

def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    if n < 10:
        m = metric_fn(y_true, y_score)
        return float(m), float("nan"), float("nan")

    point = metric_fn(y_true, y_score)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(metric_fn(y_true[idx], y_score[idx]))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float(point), float("nan"), float("nan")
    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    return float(point), lo, hi

def rolling_year_splits(df: pd.DataFrame, start_year: int, end_year: int):
    """
    Yields (train_df, test_df, test_year) for year-by-year forward-chaining evaluation:
      train <= (y-1), test == y
    """
    for y in range(start_year, end_year + 1):
        tr = df[df["year_"] <= (y - 1)].copy()
        te = df[df["year_"] == y].copy()
        if len(tr) == 0 or len(te) == 0:
            continue
        yield tr, te, y


# ──────────────────────────────────────────────────────────────────────────────
# McNemar exact test (no statsmodels dependency)
# ──────────────────────────────────────────────────────────────────────────────
def mcnemar_exact(y_true, pred_a, pred_b) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    pred_a = np.asarray(pred_a).astype(int)
    pred_b = np.asarray(pred_b).astype(int)

    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)

    b = int(np.sum(a_correct & (~b_correct)))
    c = int(np.sum((~a_correct) & b_correct))

    n = b + c
    if n == 0:
        return {"b": float(b), "c": float(c), "p_value": 1.0}

    from math import comb
    k = min(b, c)
    p_le = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p_ge = sum(comb(n, i) for i in range(max(b, c), n + 1)) / (2 ** n)
    p = 2.0 * min(p_le, p_ge)
    p = min(p, 1.0)
    return {"b": float(b), "c": float(c), "p_value": float(p)}


# ──────────────────────────────────────────────────────────────────────────────
# DeLong (ROC AUC) test — lightweight implementation
# ──────────────────────────────────────────────────────────────────────────────
def _compute_midrank(x):
    x = np.asarray(x)
    idx = np.argsort(x)
    sorted_x = x[idx]
    n = len(x)
    midranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        mid = 0.5 * (i + j - 1) + 1  # 1-based
        midranks[i:j] = mid
        i = j
    out = np.empty(n, dtype=float)
    out[idx] = midranks
    return out

def delong_roc_test(y_true: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    s1 = np.asarray(s1).astype(float)
    s2 = np.asarray(s2).astype(float)

    m = np.isfinite(s1) & np.isfinite(s2) & np.isfinite(y_true)
    y_true = y_true[m]
    s1 = s1[m]
    s2 = s2[m]

    pos = y_true == 1
    neg = y_true == 0
    if pos.sum() < 5 or neg.sum() < 5:
        return {"auc1": float("nan"), "auc2": float("nan"), "p_value": float("nan")}

    order = np.concatenate([np.where(pos)[0], np.where(neg)[0]])
    y = y_true[order]
    X = np.vstack([s1[order], s2[order]])  # (2, N)

    m_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    V = np.zeros((2, len(y)), dtype=float)
    for r in range(2):
        V[r, :] = _compute_midrank(X[r, :])

    V10 = V[:, :m_pos]
    V01 = V[:, m_pos:]

    aucs = (V10.sum(axis=1) - m_pos * (m_pos + 1) / 2) / (m_pos * n_neg)

    sx = (V10 - V10.mean(axis=1, keepdims=True)) / n_neg
    sy = (V01 - V01.mean(axis=1, keepdims=True)) / m_pos
    S = np.cov(sx, bias=False) + np.cov(sy, bias=False)
    diff = aucs[0] - aucs[1]
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "p_value": float("nan")}

    z = diff / np.sqrt(var)
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "p_value": float(p)}


# ──────────────────────────────────────────────────────────────────────────────
# Data load + split
# ──────────────────────────────────────────────────────────────────────────────
def load_labeled(glob_str: str) -> pd.DataFrame:
    paths = [Path(p) for p in sorted(glob.glob(glob_str))]
    if not paths:
        raise SystemExit(f"❌ No files matched: {glob_str}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df["__source_file"] = p.name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def add_year_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "year" in df.columns and "month_num" in df.columns:
        df["year_"] = pd.to_numeric(df["year"], errors="coerce")
        df["month_"] = pd.to_numeric(df["month_num"], errors="coerce")
        return df
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df["year_"] = dt.dt.year
        df["month_"] = dt.dt.month
        return df
    df["year_"] = np.nan
    df["month_"] = np.nan
    return df

def _choose_group_col(df: pd.DataFrame) -> Optional[str]:
    return _pick_first_existing(df, ["scene_id", "chip_id", "tile"])

def make_group_id(df: pd.DataFrame) -> np.ndarray:
    group_col = _choose_group_col(df)
    if group_col is None:
        return np.arange(len(df))
    if group_col == "chip_id":
        s = df["chip_id"].astype(str).str.rsplit("_", n=1).str[0]
        return s.to_numpy()
    return df[group_col].astype(str).to_numpy()

def time_split(df: pd.DataFrame, train_end_year: int, test_year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = df[df["year_"] <= train_end_year].copy()
    te = df[df["year_"] == test_year].copy()
    return tr, te

def group_split(df: pd.DataFrame, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    groups = make_group_id(df)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(len(df))
    tr_idx, te_idx = next(gss.split(idx, groups=groups))
    return df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Baselines
# ──────────────────────────────────────────────────────────────────────────────
def build_rf_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    seed: int,
    debug_topk: int = 25,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, float]]]:
    """
    Returns (train_prob, test_prob, top_importances) for RF baseline.
    """
    X_tr = train_df[feature_cols]
    X_te = test_df[feature_cols]
    y_tr = train_df[label_col].astype(int).to_numpy()

    imp = SimpleImputer(strategy="median")
    X_tr_i = imp.fit_transform(X_tr)
    X_te_i = imp.transform(X_te)

    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_depth=None,
        min_samples_leaf=2,
    )
    rf.fit(X_tr_i, y_tr)

    # robust predict_proba
    proba_tr = rf.predict_proba(X_tr_i)
    proba_te = rf.predict_proba(X_te_i)
    if proba_tr.shape[1] < 2 or proba_te.shape[1] < 2:
        raise RuntimeError("RF predict_proba returned single column (single-class fit)")

    p_tr = proba_tr[:, 1]
    p_te = proba_te[:, 1]

    # feature importances diagnostic
    imps = getattr(rf, "feature_importances_", None)
    top = []
    if imps is not None and len(imps) == len(feature_cols):
        top = sorted(zip(feature_cols, imps), key=lambda x: x[1], reverse=True)[:debug_topk]
    return p_tr, p_te, top


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment runner
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser("Rigorous labeled benchmark: baselines + seeds + CI + significance + diagnostics")
    ap.add_argument("--labeled_glob", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--label_col", default=None)
    ap.add_argument("--score_cols", nargs="+", default=None)
    ap.add_argument("--index_col", default=None)
    ap.add_argument("--seeds", type=int, default=10)

    ap.add_argument("--train_end_year", type=int, default=2023)
    ap.add_argument("--test_year", type=int, default=2024)
    ap.add_argument("--test_size", type=float, default=0.2)

    ap.add_argument("--trusted_only", action="store_true",
                    help="Evaluate only rows with hab_label_trusted == 1")

    # RF feature control
    ap.add_argument("--rf_mode", choices=["auto", "safe"], default="auto",
                    help="auto=all numeric minus drops; safe=only known sensor/index features")
    ap.add_argument("--rf_drop", nargs="*", default=[],
                    help="Extra columns to DROP from RF features (space-separated)")
    ap.add_argument("--rf_topk", type=int, default=25,
                    help="How many RF importances to print per seed (if RF runs)")

    # new split modes
    ap.add_argument("--split_mode", choices=["time", "group", "rolling_year"], default="time")

    # rolling-year controls
    ap.add_argument("--rolling_start_year", type=int, default=2017) #trying 2017, was 2019
    ap.add_argument("--rolling_end_year", type=int, default=2024)


    ap.add_argument("--thr_policy", choices=["best_f1"], default="best_f1")
    ap.add_argument("--n_boot", type=int, default=1000)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_labeled(args.labeled_glob)
    df = add_year_month(df)

    label_col = args.label_col or _pick_first_existing(
        df, ["hab_label_final", "hab_label_final2", "hab_label", "hab_label_fusion", "label", "y"]
    )
    if label_col is None:
        raise SystemExit(f"❌ Could not find a label column. Columns={list(df.columns)}")

    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = df[label_col].astype(int)

    # trusted_only filter
    if args.trusted_only:
        if "hab_label_trusted" not in df.columns:
            raise SystemExit("❌ --trusted_only set but hab_label_trusted column not found")
        before = len(df)
        df = df[df["hab_label_trusted"] == 1].copy()
        print(f"[info] trusted_only: {before} → {len(df)} rows")

    # score cols
    default_scores = ["hab_prob", "p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med", "p_fused", "prob"]
    score_cols = args.score_cols or [c for c in default_scores if c in df.columns]
    if not score_cols:
        raise SystemExit(f"❌ No score columns found. Expected like {default_scores}.")

    # index col
    idx_col = args.index_col or _pick_first_existing(df, ["fai_mean", "ndci_mean", "ndci", "fai"])
    if idx_col is None:
        print("[warn] No index baseline column found (FAI/NDCI). Index baseline will be skipped.")

    # ── diagnostics: grouping
    group_col = _choose_group_col(df)
    print(f"[diag] group_col chosen = {group_col if group_col else 'None (row index)'}")

    # Build RF feature cols
    id_like = set([c for c in ["tile", "scene_id", "datetime", "chip_id", "month_key",
                               "__source_file", "month", "season"] if c in df.columns])
    label_like = set([c for c in df.columns if "label" in c.lower()])
    target_like = set([c for c in df.columns if c.lower() in {"y", "target", "class"}])

    # HARD suspicious defaults 
    suspicious_defaults = set()

    extra_drop = set(args.rf_drop or [])
    drop_cols = set(score_cols) | {label_col} | id_like | label_like | target_like | suspicious_defaults | extra_drop

    # coerce numeric once for feature detection
    for c in df.columns:
        if c in drop_cols:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="ignore")

    if args.rf_mode == "safe":
        SAFE_FEATURES = [
            # physical / index features
            "ndwi_mean","ndwi_std","fai_mean","fai_std","rednir_mean","rednir_std","valid_px",
            "chlor_a","kd490","nflh","sst","flh",
            "log_kd490","log_chlor_a","log_nflh",
            "ratio_chl_kd","chl_times_nflh","ratio_nflh_kd",
            "month_sin","month_cos",
        ]
        feature_cols = [c for c in SAFE_FEATURES if c in df.columns and c not in drop_cols]
        print(f"[diag] RF mode=safe → {len(feature_cols)} features")
        missing = [c for c in SAFE_FEATURES if c not in df.columns]
        if missing:
            print(f"[diag] SAFE_FEATURES missing in df (ok): {missing[:15]}{'...' if len(missing)>15 else ''}")
    else:
        numeric_cols = [c for c in df.columns
                        if (c not in drop_cols) and pd.api.types.is_numeric_dtype(df[c])]
        feature_cols = numeric_cols
        print(f"[diag] RF mode=auto → {len(feature_cols)} numeric features after drop set")

    # quick diagnostic: constant-ish features
    if feature_cols:
        const_feats = []
        for c in feature_cols:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):
                v = s.to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                if len(v) > 0:
                    if np.nanstd(v) < 1e-12:
                        const_feats.append(c)
        if const_feats:
            print(f"[diag] constant features (std≈0): {const_feats[:15]}{'...' if len(const_feats)>15 else ''}")

    all_seed_rows = []
    ci_payload = {"per_model": {}, "notes": {}}
    delong_rows = []
    mcnemar_rows = []

    all_seed_rows = []
    def run_one_split(tr, te, seed, split_tag):
        if len(te) < 50 or len(tr) < 100:
            print(f"[warn][{split_tag}][seed {seed}] small split (train={len(tr)}, test={len(te)})")

        y_tr = tr[label_col].to_numpy()
        y_te = te[label_col].to_numpy()

        model_probs_tr = {}
        model_probs_te = {}

        # 1) existing score columns
        for c in score_cols:
            if c not in tr.columns or c not in te.columns:
                print(f"[warn][{split_tag}][seed {seed}] score col '{c}' missing")
                continue
            model_probs_tr[c] = _as_float(tr[c])
            model_probs_te[c] = _as_float(te[c])

        # 2) index baseline
        if idx_col is not None and idx_col in tr.columns:
            idx_tr = _as_float(tr[idx_col])
            idx_te = _as_float(te[idx_col])
            med = np.nanmedian(idx_tr)
            mad = np.nanmedian(np.abs(idx_tr - med)) + 1e-6
            z_tr = (idx_tr - med) / (1.4826 * mad)
            z_te = (idx_te - med) / (1.4826 * mad)
            model_probs_tr[f"index::{idx_col}"] = sigmoid(z_tr)
            model_probs_te[f"index::{idx_col}"] = sigmoid(z_te)

        # 3) RF baseline
        if len(feature_cols) >= 5 and len(np.unique(y_tr)) >= 2:
            rf_tr, rf_te, rf_top = build_rf_baseline(
                tr, te, label_col, feature_cols, seed, debug_topk=args.rf_topk
            )

            # diagnostic print (once per split+seed)
            if rf_top:
                print(f"[diag][{split_tag}][seed {seed}] RF top-{len(rf_top)} importances:")
                for name, val in rf_top:
                    print(f"  {name:<30} {val:.4f}")
            model_probs_tr["rf_baseline"] = rf_tr
            model_probs_te["rf_baseline"] = rf_te
        else:
            print(f"[warn][{split_tag}][seed {seed}] skipping RF")

        thresholds = {}

        for name, p_tr in model_probs_tr.items():
            p_te = model_probs_te[name]

            thr = best_f1_threshold(y_tr, p_tr)
            thresholds[name] = thr

            roc = safe_roc_auc(y_te, p_te)
            pr = safe_pr_auc(y_te, p_te)
            thr_metrics = eval_threshold_metrics(y_te, p_te, thr)

            all_seed_rows.append({
                "seed": seed,
                "split": split_tag,
                "model": name,
                "n_train": len(tr),
                "n_test": len(te),
                "pos_rate_test": float(np.mean(y_te)),
                "roc_auc": roc,
                "pr_auc": pr,
                **thr_metrics,
            })

        # significance tests (unchanged)
        main_model = "hab_prob" if "hab_prob" in model_probs_te else list(model_probs_te.keys())[0]
        p_main = model_probs_te[main_model]

        for name, p_te in model_probs_te.items():
            if name == main_model:
                continue

            d = delong_roc_test(y_te, p_main, p_te)
            delong_rows.append({
                "seed": seed,
                "split": split_tag,
                "main_model": main_model,
                "other_model": name,
                "auc_main": d["auc1"],
                "auc_other": d["auc2"],
                "p_value": d["p_value"],
            })

            pred_a = (p_main >= thresholds[main_model]).astype(int)
            pred_b = (p_te >= thresholds[name]).astype(int)
            mc = mcnemar_exact(y_te, pred_a, pred_b)

            mcnemar_rows.append({
                "seed": seed,
                "split": split_tag,
                "main_model": main_model,
                "other_model": name,
                "b_Acorrect_Bwrong": mc["b"],
                "c_Awrong_Bcorrect": mc["c"],
                "p_value": mc["p_value"],
            })

    for seed in range(args.seeds):
        if args.split_mode == "time":
            tr, te = time_split(df, args.train_end_year, args.test_year)
            run_one_split(tr, te, seed, f"time_{args.train_end_year}_vs_{args.test_year}")

        elif args.split_mode == "rolling_year":
            for tr, te, y in rolling_year_splits(
                df,
                args.rolling_start_year,
                args.rolling_end_year
            ):
                run_one_split(tr, te, seed, f"roll_test_{y}")

        else:  # group
            tr, te = group_split(df, args.test_size, seed)
            run_one_split(tr, te, seed, f"group_sceneid_{args.test_size}")


    # Write per-seed metrics
    per_seed = pd.DataFrame(all_seed_rows)
    per_seed.to_csv(outdir / "per_seed_metrics.csv", index=False)

    summary = (
        per_seed
        .groupby(["split", "model"], as_index=False)
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            kappa_mean=("kappa", "mean"),
            kappa_std=("kappa", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
        )
    )

    summary.to_csv(outdir / "summary_mean_std.csv", index=False)

    if delong_rows:
        pd.DataFrame(delong_rows).to_csv(outdir / "delong_vs_main.csv", index=False)
    if mcnemar_rows:
        pd.DataFrame(mcnemar_rows).to_csv(outdir / "mcnemar_vs_main.csv", index=False)

    ci_payload["notes"] = {
        "split_mode": args.split_mode,
        "time_split": {"train_end_year": args.train_end_year, "test_year": args.test_year} if args.split_mode == "time" else None,
        "n_boot": args.n_boot,
        "thr_policy": args.thr_policy,
        "main_model": "hab_prob" if "hab_prob" in per_seed["model"].unique() else str(per_seed["model"].unique()[0]),
        "rf_mode": args.rf_mode,
        "rf_extra_drop": list(extra_drop),
    }
    (outdir / "bootstrap_ci_per_seed.json").write_text(json.dumps(ci_payload, indent=2))

    (outdir / "README.txt").write_text(
        f"""Rigorous labeled benchmark
=========================
Input: {args.labeled_glob}

Label col: {label_col}
Score cols: {score_cols}
Index baseline: {idx_col if idx_col else "none"}

Split mode: {args.split_mode}
  - time: train<= {args.train_end_year}, test== {args.test_year}
  - group: GroupShuffleSplit test_size={args.test_size} per seed

Runs: {args.seeds} seeds
Threshold policy: best F1 on TRAIN only
Bootstrap: {args.n_boot} resamples per seed (ROC-AUC / PR-AUC)

RF baseline:
  - mode: {args.rf_mode}
  - extra drop: {args.rf_drop}

Diagnostics printed to console:
  - group_col chosen
  - per-seed train/test sizes, pos rates
  - per-seed group overlap (must be 0)
  - RF top importances (if RF runs)

Outputs:
- per_seed_metrics.csv
- summary_mean_std.csv
- bootstrap_ci_per_seed.json
- delong_vs_main.csv
- mcnemar_vs_main.csv
"""
    )

    print("✅ Done.")
    print("→ wrote:", outdir)
    print("→ models:", sorted(per_seed["model"].unique().tolist()))


if __name__ == "__main__":
    main()
