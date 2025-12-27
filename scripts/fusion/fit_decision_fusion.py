#!/usr/bin/env python3
# fit_decision_fusion.py
"""
Decision-level fusion for S2 tabular scores + external detectors.

Fixes:
  - Prevents leakage: all env imputation + SST climatology/anomaly + region-month context
    are fit on TRAIN only and applied to TEST per split.

Adds/keeps:
  - sst_anom (region x month-of-year climatology) + z-score (split-safe)
  - model choice: logreg | catboost
  - Option A threshold: Max precision subject to recall >= min_recall
      --threshold_policy prec_at_recall --min_recall 0.60
  - Split-safe probability calibration (sigmoid or isotonic)
  - Explicit interaction features (basic HAB-motivated)
"""

import argparse, json, re, warnings, hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGF = True
except Exception:
    HAS_SGF = False

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, roc_curve,
    classification_report, confusion_matrix
)
from sklearn.isotonic import IsotonicRegression


# ---------- custom scaler: standardize then apply per-feature weights ----------
class WeightedScaler(StandardScaler):
    """StandardScaler followed by per-feature multipliers."""
    def __init__(self, weights=None):
        super().__init__()
        self.weights = np.asarray(weights) if weights is not None else None

    def transform(self, X, copy=None):
        X_scaled = super().transform(X, copy)
        if self.weights is not None:
            X_scaled = X_scaled * self.weights
        return X_scaled


# ---------- calibration helpers ----------
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

class _SigmoidCalibrator:
    """Platt scaling on logit(p) -> y via logistic regression."""
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, p, y):
        x = _logit(p).reshape(-1, 1)
        y = np.asarray(y, dtype=int)
        lr = LogisticRegression(max_iter=1000, solver="lbfgs")
        lr.fit(x, y)
        self.a_ = float(lr.coef_.ravel()[0])
        self.b_ = float(lr.intercept_.ravel()[0])
        return self

    def transform(self, p):
        x = _logit(p)
        z = self.a_ * x + self.b_
        return 1.0 / (1.0 + np.exp(-z))

class _IsotonicCalibrator:
    def __init__(self):
        self.iso_ = IsotonicRegression(out_of_bounds="clip")

    def fit(self, p, y):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        y = np.asarray(y, dtype=float)
        self.iso_.fit(p, y)
        return self

    def transform(self, p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.clip(self.iso_.transform(p), 1e-6, 1 - 1e-6)

def _fit_calibrator(method: str, p_cal, y_cal):
    if method == "none":
        return None
    if method == "sigmoid":
        return _SigmoidCalibrator().fit(p_cal, y_cal)
    if method == "isotonic":
        return _IsotonicCalibrator().fit(p_cal, y_cal)
    raise ValueError(f"Unknown calibrate='{method}'")

def _apply_calibrator(cal, p):
    if cal is None:
        return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.clip(cal.transform(p), 1e-6, 1 - 1e-6)


# ---------------- plotting ----------------
def _pr_envelope(rec, prec):
    if len(rec) == 0:
        return rec, prec
    rec = np.asarray(rec, dtype=float)
    prec = np.asarray(prec, dtype=float)
    df = pd.DataFrame({"rec": rec, "prec": prec})
    df = df.groupby("rec", as_index=False)["prec"].max().sort_values("rec")
    rec_u = df["rec"].values
    prec_u = df["prec"].values
    prec_env = np.maximum.accumulate(prec_u[::-1])[::-1]
    return rec_u, prec_env

def _pr_plot(rec, prec, auprc, base, outpng):
    plt.close("all")
    if len(rec) and rec[0] == 0 and len(prec) and prec[0] == 1:
        rec, prec = rec[1:], prec[1:]
    rec, prec = _pr_envelope(rec, prec)
    plt.figure(figsize=(5.6, 4.4))
    plt.step(rec, prec, where="post", lw=2.0, alpha=0.95, label="Precision–Recall")
    if len(rec) <= 200:
        plt.scatter(rec, prec, s=10)
    if base is not None:
        plt.hlines(base, 0, 1, linestyles="--", alpha=0.6, label="Baseline")
    base_str = f"{base:.3f}" if base is not None else "nan"
    plt.title(f"PR (AUPRC={auprc:.3f}, base={base_str})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.xlim([-0.05, 1.05]); plt.ylim([-0.05, 1.05])
    plt.grid(True, linestyle="--", alpha=0.35); plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout(); plt.savefig(outpng, dpi=200); plt.close()

def _roc_plot(fpr, tpr, auroc, outpng):
    plt.close("all")
    plt.figure(figsize=(5.6, 4.4))
    plt.plot(fpr, tpr, lw=2.0, label="ROC")
    plt.plot([0, 1], [0, 1], "--", alpha=0.6, label="Random")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUROC={auroc:.3f})")
    plt.xlim([-0.05, 1.05]); plt.ylim([-0.05, 1.05])
    plt.grid(True, linestyle="--", alpha=0.35); plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(outpng, dpi=200); plt.close()


# ---------------- helpers -----------------
REGION_RE = re.compile(r"r\d{3}_c\d{3}", re.IGNORECASE)
MGRS_RE = re.compile(r"T\d{2}[A-Z]{3}")
DATE8_RE = re.compile(r"(20\d{2})(\d{2})(\d{2})")
MONTH_RE = re.compile(r"(20\d{2})[._-]?(\d{2})")
RANGE_RE = re.compile(r"(20\d{2})(\d{2})(\d{2})_(20\d{2})(\d{2})(\d{2})")
TAIL4_RE = re.compile(r"_(\d{4})$")

def _clean_columns(df: pd.DataFrame, src: str = "") -> pd.DataFrame:
    df = df.copy()
    cols = [str(c).strip() for c in df.columns]
    dup_mask = pd.Index(cols).duplicated(keep="first")
    if dup_mask.any():
        dropped = [c for i, c in enumerate(cols) if dup_mask[i]]
        print(f"[warn] {src} had duplicated columns {dropped}; keeping first occurrence.")
        df = df.loc[:, ~dup_mask]
        cols = [c for i, c in enumerate(cols) if not dup_mask[i]]
    df.columns = cols
    bad = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if bad:
        df = df.drop(columns=bad)
    return df

def _normalize_ids(s: pd.Series) -> pd.Series:
    return s.astype(str).apply(lambda x: Path(str(x)).name)

def _swap_ext(name: str) -> str:
    p = Path(name)
    if p.suffix.lower() in (".jpg", ".jpeg"):
        return p.with_suffix(".png").name
    if p.suffix.lower() == ".png":
        return p.with_suffix(".jpg").name
    return p.name

def _canonical_scene_key(x: str) -> str:
    stem = Path(str(x)).stem
    return TAIL4_RE.sub("", stem)

def _extract_month_key_from_scene(scene_id: str) -> str | None:
    m = DATE8_RE.search(scene_id)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m2 = MONTH_RE.search(scene_id)
    return f"{m2.group(1)}-{m2.group(2)}" if m2 else None

def _extract_month_key_from_modis(fname: str) -> str | None:
    r = RANGE_RE.search(fname)
    if r:
        return f"{r.group(1)}-{r.group(2)}"
    m = MONTH_RE.search(fname)
    return f"{m.group(1)}-{m.group(2)}" if m else None

def _extract_dates_from_name(s: str):
    s = str(s)
    r = RANGE_RE.search(s)
    if r:
        try:
            d1 = datetime(int(r.group(1)), int(r.group(2)), int(r.group(3)))
            d2 = datetime(int(r.group(4)), int(r.group(5)), int(r.group(6)))
            return d1, d2
        except Exception:
            return None, None
    m = DATE8_RE.search(s)
    if m:
        try:
            d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return d, d
        except Exception:
            return None, None
    return None, None

def _mid_date(d1, d2):
    if d1 is None and d2 is None:
        return None
    if d1 is None:
        return d2
    if d2 is None:
        return d1
    return d1 + (d2 - d1) / 2

def _extract_region_key(s: str) -> str | None:
    m = REGION_RE.search(s)
    if m:
        return m.group(0).lower()
    m2 = MGRS_RE.search(s)
    if m2:
        return m2.group(0).upper()
    return None

def _load_coco_map(coco_json: str):
    if not coco_json:
        return None, None
    with open(coco_json, "r") as f:
        coco = json.load(f)
    id2name = {int(im["id"]): str(im["file_name"]) for im in coco["images"]}
    name2id = {str(v): int(k) for k, v in id2name.items()}
    return id2name, name2id

def _coerce_id(df: pd.DataFrame, id_col: str, src: str) -> pd.DataFrame:
    df = df.copy()
    if id_col not in df.columns:
        if "chip_id" in df.columns:
            df = df.rename(columns={"chip_id": id_col})
        elif "tile" in df.columns:
            df = df.rename(columns={"tile": id_col})
        elif "image" in df.columns:
            df[id_col] = df["image"].apply(lambda s: Path(str(s)).name)
        else:
            raise SystemExit(f"{src} has no id column '{id_col}'. Columns: {list(df.columns)}")
    return df

def _guess_score_col(df: pd.DataFrame, id_col: str) -> str:
    numeric = [c for c in df.columns if c != id_col and pd.api.types.is_numeric_dtype(df[c])]
    cand = [c for c in numeric if str(c).lower().startswith("p_") or "score" in str(c).lower()]
    cand = [c for c in cand if not str(c).lower().endswith("count")]
    if cand:
        return cand[0]
    numeric = [c for c in numeric if not str(c).lower().endswith("count")]
    if numeric:
        return numeric[0]
    for c in df.columns:
        if c != id_col:
            return c
    raise SystemExit("Could not guess score column.")

def _attach_month_region_keys_for_base(base: pd.DataFrame, id_col: str, group_col: str):
    """Ensure month_key, date_key, region_key exist for base table."""
    base = base.copy()

    if "month_key" not in base.columns:
        if "datetime" in base.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dt = pd.to_datetime(base["datetime"], errors="coerce", utc=True)
            base["month_key"] = dt.dt.strftime("%Y-%m")
        else:
            base["month_key"] = base[group_col].astype(str).map(_extract_month_key_from_scene)

    if "datetime" in base.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt = pd.to_datetime(base["datetime"], errors="coerce", utc=True)
        # store naive timestamp
        base["date_key"] = dt.dt.tz_convert(None)
    else:
        d1d2 = base[group_col].astype(str).map(_extract_dates_from_name)
        base["date_key"] = [_mid_date(a, b) for (a, b) in d1d2]

    base["region_key"] = base[group_col].astype(str).map(_extract_region_key)
    mask = base["region_key"].isna()
    if mask.any():
        base.loc[mask, "region_key"] = base.loc[mask, id_col].astype(str).map(_extract_region_key)
    return base

def _attach_month_region_keys_for_det(det_df: pd.DataFrame, id_col: str):
    det = det_df.copy()
    det["month_key"] = det[id_col].astype(str).map(_extract_month_key_from_modis)
    det["region_key"] = det[id_col].astype(str).map(_extract_region_key)
    d1d2 = det[id_col].astype(str).map(_extract_dates_from_name)
    det["date_key"] = [_mid_date(a, b) for (a, b) in d1d2]
    return det

def _agg(df: pd.DataFrame, by_cols, score_name: str, how: str) -> pd.DataFrame:
    if how == "max":
        return df.groupby(by_cols, as_index=False)[score_name].max()
    if how == "mean":
        return df.groupby(by_cols, as_index=False)[score_name].mean()
    if how == "median":
        return df.groupby(by_cols, as_index=False)[score_name].median()
    raise ValueError(f"Unknown agg '{how}'")

def _merge_on_id(base, det_df, id_col, score_name):
    merged = base.merge(det_df[[id_col, score_name]], on=id_col, how="left")
    if merged[score_name].isna().mean() < 1.0:
        cov = 100.0 * merged[score_name].notna().mean()
        print(f"[info] Detector '{score_name}' merged on {id_col}. Coverage={cov:.1f}%")
        return merged, True
    det_tmp = det_df.copy()
    det_tmp[id_col] = det_tmp[id_col].apply(_swap_ext)
    merged2 = base.merge(det_tmp[[id_col, score_name]], on=id_col, how="left")
    if merged2[score_name].isna().mean() < 1.0:
        cov = 100.0 * merged2[score_name].notna().mean()
        print(f"[info] Detector '{score_name}' merged on {id_col} via ext swap. Coverage={cov:.1f}%")
        return merged2, True
    return base, False

def _merge_on_scene(base, det_df_scene_like, group_col, score_name):
    if group_col not in base.columns:
        return base, False
    det_tmp = det_df_scene_like.copy()
    det_tmp = det_tmp.iloc[:, :2].copy()
    det_tmp.columns = ["__file__", score_name]
    det_tmp["__scene_key__"] = det_tmp["__file__"].astype(str).apply(lambda s: _canonical_scene_key(s))
    base2 = base.copy()
    base2["__scene_key__"] = base[group_col].astype(str).apply(lambda s: _canonical_scene_key(s))
    merged = base2.merge(
        det_tmp[["__scene_key__", score_name]], on="__scene_key__", how="left"
    ).drop(columns=["__scene_key__"], errors="ignore")
    if merged[score_name].isna().mean() < 1.0:
        cov = 100.0 * merged[score_name].notna().mean()
        print(f"[info] Detector '{score_name}' merged on scene. Coverage={cov:.1f}%")
        return merged, True
    det_tmp2 = det_df_scene_like.iloc[:, :2].copy()
    det_tmp2.columns = ["__file__", score_name]
    det_tmp2["__file__"] = det_tmp2["__file__"].map(_swap_ext)
    det_tmp2["__scene_key__"] = det_tmp2["__file__"].astype(str).apply(lambda s: _canonical_scene_key(s))
    merged2 = base2.merge(
        det_tmp2[["__scene_key__", score_name]], on="__scene_key__", how="left"
    ).drop(columns=["__scene_key__"], errors="ignore")
    if merged2[score_name].isna().mean() < 1.0:
        cov = 100.0 * merged2[score_name].notna().mean()
        print(f"[info] Detector '{score_name}' merged on scene via ext swap. Coverage={cov:.1f}%")
        return merged2, True
    return base, False

def _nearest_month_map(base_months: pd.Series, det_months: pd.Series, k_months: int = 1) -> dict:
    def _to_ym(s: str) -> datetime | None:
        try:
            y, m = s.split("-")
            return datetime(int(y), int(m), 1)
        except Exception:
            return None

    b = sorted(set([m for m in base_months.dropna().astype(str)]))
    d = sorted(set([m for m in det_months.dropna().astype(str)]))
    bd = {m: _to_ym(m) for m in b}
    dd = {m: _to_ym(m) for m in d}
    out = {}
    for bm in b:
        bdt = bd[bm]
        best, best_delta = None, 999
        for dm, ddt in dd.items():
            delta = abs((ddt.year - bdt.year) * 12 + (ddt.month - bdt.month))
            if delta <= k_months and delta < best_delta:
                best, best_delta = dm, delta
        out[bm] = best
    return out

def _merge_on_month_region(base, det_df, score_name, agg: str, month_backfill: int = 0):
    det_ok = det_df.copy()

    if {"month_key", "region_key", score_name}.issubset(det_ok.columns) and "region_key" in base.columns:
        det_mr = _agg(det_ok.dropna(subset=["month_key", "region_key"]), ["month_key", "region_key"], score_name, agg)
        m = base.merge(det_mr, on=["month_key", "region_key"], how="left")
        if m[score_name].isna().mean() < 1.0:
            cov = 100.0 * m[score_name].notna().mean()
            print(f"[info] Detector '{score_name}' merged on month+region ({agg}). Coverage={cov:.1f}%")
            return m, True

    if {"month_key", score_name}.issubset(det_ok.columns):
        det_m = _agg(det_ok.dropna(subset=["month_key"]), ["month_key"], score_name, agg)
        m2 = base.merge(det_m, on="month_key", how="left")
        if m2[score_name].isna().mean() < 1.0:
            cov = 100.0 * m2[score_name].notna().mean()
            print(f"[info] Detector '{score_name}' merged on month-only ({agg}). Coverage={cov:.1f}%")
            return m2, True

        if month_backfill and len(det_m):
            nn_map = _nearest_month_map(base["month_key"], det_m["month_key"], k_months=month_backfill)
            base_remap = base.copy()
            base_remap["__det_month__"] = base_remap["month_key"].map(nn_map)
            det_m2 = det_m.rename(columns={"month_key": "__det_month__"})
            m3 = base_remap.merge(det_m2, on="__det_month__", how="left").drop(
                columns="__det_month__", errors="ignore"
            )
            if m3[score_name].isna().mean() < 1.0:
                cov = 100.0 * m3[score_name].notna().mean()
                print(
                    f"[info] Detector '{score_name}' merged via nearest-month ±{month_backfill} ({agg}). "
                    f"Coverage={cov:.1f}%"
                )
                return m3, True

    return base, False

def _merge_on_nearest_date(base, det_df, score_name, max_day_gap: int = 12, use_region: bool = True):
    if "date_key" not in base.columns or "date_key" not in det_df.columns:
        return base, False
    b = base.dropna(subset=["date_key"]).sort_values("date_key")
    d = det_df.dropna(subset=["date_key"]).sort_values("date_key")
    if b.empty or d.empty:
        return base, False

    if use_region and "region_key" in b.columns and "region_key" in d.columns:
        parts = []
        for rk, g in b.groupby("region_key"):
            dd = d[d["region_key"] == rk]
            if dd.empty:
                parts.append(g.assign(**{score_name: np.nan}))
                continue
            mg = pd.merge_asof(
                g,
                dd[["date_key", score_name]].sort_values("date_key"),
                on="date_key",
                direction="nearest",
                tolerance=pd.Timedelta(days=max_day_gap),
            )
            parts.append(mg)
        merged = pd.concat(parts, axis=0).sort_index()
    else:
        merged = pd.merge_asof(
            b,
            d[["date_key", score_name]].sort_values("date_key"),
            on="date_key",
            direction="nearest",
            tolerance=pd.Timedelta(days=max_day_gap),
        )
    out = base.copy()
    out[score_name] = merged.reindex(base.index)[score_name]
    if out[score_name].isna().mean() < 1.0:
        cov = 100.0 * out[score_name].notna().mean()
        print(f"[info] Detector '{score_name}' merged on nearest-date ±{max_day_gap}d. Coverage={cov:.1f}%")
        return out, True
    return base, False


# --------------- thresholding ---------------
def _pick_threshold_from_policy(
    y,
    p,
    prec,
    rec,
    thr,
    policy="f1",
    target_precision=None,
    target_recall=None,
    target_fpr=None,
    top_frac=None,
    expected_pos_rate=None,
    min_recall=None,
):
    thr = np.asarray(thr)

    if policy == "prec_at_recall":
        if min_recall is None:
            min_recall = 0.60
        ok = np.where(rec[:-1] >= float(min_recall))[0]
        if len(ok):
            best = ok[np.argmax(prec[:-1][ok])]
            return float(thr[best]), f"prec_at_recall>=({min_recall:.3f})"
        # fallback: max recall -> lowest threshold
        if len(thr):
            return float(np.min(thr)), f"fallback_max_recall(min_recall_unmet={min_recall:.3f})"
        return 0.0, f"fallback_max_recall(min_recall_unmet={min_recall:.3f})"

    if policy == "precision" and target_precision is not None:
        cand = np.where(prec[:-1] >= target_precision)[0]
        if len(cand):
            return float(thr[cand[0]]), "precision"

    if policy == "recall" and target_recall is not None:
        cand = np.where(rec[:-1] >= target_recall)[0]
        if len(cand):
            return float(thr[cand[-1]]), "recall"

    if policy == "fpr" and target_fpr is not None:
        neg = p[y == 0]
        if len(neg):
            q = np.clip(1.0 - float(target_fpr), 0.0, 1.0)
            return float(np.quantile(neg, q)), "fpr_neg_quantile"

    if policy == "topfrac" and top_frac is not None:
        q = np.clip(1.0 - float(top_frac), 0.0, 1.0)
        return float(np.quantile(p, q)), "topfrac"

    if policy == "expected_pos" and expected_pos_rate is not None:
        q = np.clip(1.0 - float(expected_pos_rate), 0.0, 1.0)
        return float(np.quantile(p, q)), f"expected_pos(rate={expected_pos_rate:.3f})"

    f1s = (2 * prec * rec / (prec + rec + 1e-12))[:-1]
    i = int(np.argmax(f1s)) if len(f1s) else 0
    return (float(thr[i]) if len(thr) else 0.5), "f1"

def _safe_metrics(y_true, p):
    base_rate = float(y_true.mean()) if y_true.size else None
    auprc = average_precision_score(y_true, p) if y_true.sum() > 0 else 0.0
    auroc = roc_auc_score(y_true, p) if len(np.unique(y_true)) == 2 else float("nan")
    prec, rec, thr = precision_recall_curve(y_true, p)
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, p)
    else:
        fpr, tpr = np.array([0, 1]), np.array([0, 1])
    return base_rate, auprc, auroc, prec, rec, thr, fpr, tpr

def _eval_and_save(outdir: Path, tag: str, feats, y_true, p, thr_star, ids_df):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    base_rate, auprc, auroc, prec, rec, thr, fpr, tpr = _safe_metrics(y_true, p)
    yhat = (p >= thr_star).astype(int)
    cm = confusion_matrix(y_true, yhat).tolist()
    rep = classification_report(y_true, yhat, digits=3, zero_division=0)
    uniq = len(np.unique(np.round(p, 6)))
    print(f"[diag:{tag}] n={len(y_true)} pos={int(y_true.sum())} base={base_rate:.3f} unique_scores≈{uniq}")

    preds = ids_df.copy()
    preds["p_fused"] = p
    preds["yhat"] = yhat
    preds["thr_star"] = thr_star
    preds.to_csv(outdir / f"predictions_{tag}.csv", index=False)

    _pr_plot(rec, prec, auprc, base_rate, outdir / f"pr_fusion_{tag}.png")
    _roc_plot(fpr, tpr, auroc if not np.isnan(auroc) else 0.0, outdir / f"roc_fusion_{tag}.png")

    (outdir / f"metrics_{tag}.json").write_text(
        json.dumps(
            {
                "feats": feats,
                "auprc": float(auprc),
                "auroc": float(auroc) if not np.isnan(auroc) else None,
                "thr_star": float(thr_star),
                "base_rate": base_rate,
                "cm": cm,
                "n": int(len(y_true)),
                "pos": int(y_true.sum()),
                "unique_scores": uniq,
            },
            indent=2,
        )
    )

    print(
        f"[{tag}] AUPRC={auprc:.3f} AUROC={(auroc if not np.isnan(auroc) else float('nan')):.3f} "
        f"thr*={thr_star:.3f} (baseline={base_rate:.3f})"
    )
    print(f"[{tag}] Confusion matrix [[TN,FP],[FN,TP]]: {cm}")
    print(rep)
    return auprc, auroc


# --------- split-safe env feature engineering (NO LEAKAGE) ----------
def _ensure_month_num(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df["month_num"] = dt.dt.month.astype("Int64")
    return df

def _fit_env_transforms_train(df_train: pd.DataFrame, env_base_cols, interactions: str, no_env_context: bool):
    """
    Fit TRAIN-only statistics for:
      - env imputation medians per (region_key, month_key)
      - SST climatology mean/std per (region_key, month_num)
      - region-month context mean/std per (region_key, month_key) for all env/derived/interaction cols

    Returns dict with lookup tables and global fallbacks.
    """
    tr = df_train.copy()
    tr = _ensure_month_num(tr)

    # --- imputation tables (region_key, month_key) medians ---
    impute_tables = {}
    global_medians = {}
    for col in env_base_cols:
        s = pd.to_numeric(tr[col], errors="coerce")
        global_medians[col] = float(s.median()) if np.isfinite(s.median()) else 0.0
        # median per (region_key, month_key)
        gmed = (
            pd.concat([tr[["region_key", "month_key"]], s.rename(col)], axis=1)
            .groupby(["region_key", "month_key"], as_index=False)[col]
            .median()
        )
        impute_tables[col] = gmed

    # --- SST climatology mean/std per (region_key, month_num) ---
    tr_sst = tr.copy()
    tr_sst["sst"] = pd.to_numeric(tr_sst["sst"], errors="coerce")

    sst_rm = (
        tr_sst.dropna(subset=["sst", "region_key", "month_num"])
        .groupby(["region_key", "month_num"])["sst"]
        .agg(sst_clim_mean="mean", sst_clim_std="std")
        .reset_index()
    )

    # region-only fallback
    sst_r = (
        tr_sst.dropna(subset=["sst", "region_key"])
        .groupby(["region_key"])["sst"]
        .agg(sst_r_mean="mean", sst_r_std="std")
        .reset_index()
    )


    # global fallback
    sst_global_mean = float(tr_sst["sst"].median()) if np.isfinite(tr_sst["sst"].median()) else 0.0
    sst_global_std = float(tr_sst["sst"].std()) if np.isfinite(tr_sst["sst"].std()) else 1.0
    if sst_global_std <= 0:
        sst_global_std = 1.0

    # --- derived env columns list (names only) ---
    derived_env_cols = ["month_num", "sst_clim_rm", "sst_anom", "sst_anom_z",
                        "log_kd490", "log_chlor_a", "log_nflh",
                        "ratio_chl_kd", "chl_times_nflh", "ratio_nflh_kd"]

    # --- interaction columns list ---
    interaction_cols = []
    if interactions == "basic":
        inter_specs = [
            ("sst_anom", "chlor_a"),
            ("sst_anom", "nflh"),
            ("sst_anom", "fai_mean"),
            ("sst_anom", "kd490"),
            ("sst_anom", "month_sin"),
            ("sst_anom", "month_cos"),
        ]
        for a, b in inter_specs:
            interaction_cols.append(f"{a}_x_{b}")

    # --- context mean/std tables per (region_key, month_key), for (env+derived+inter) cols ---
    # We'll compute these AFTER derived/inter are created on TRAIN.
    return {
        "impute_tables": impute_tables,
        "global_medians": global_medians,
        "sst_rm": sst_rm,
        "sst_r": sst_r,
        "sst_global_mean": sst_global_mean,
        "sst_global_std": sst_global_std,
        "derived_env_cols": derived_env_cols,
        "interaction_cols": interaction_cols,
        "no_env_context": bool(no_env_context),
    }

def _apply_env_transforms(df: pd.DataFrame, env_base_cols, stats: dict, interactions: str):
    """
    Apply TRAIN-fitted env transforms to a dataframe (train or test):
      - impute env_base_cols using (region_key, month_key) medians, then global
      - compute split-safe SST climatology/anomaly/z
      - compute derived features (logs, ratios)
      - compute interactions if requested

    Returns df with added columns.
    """
    out = df.copy()
    out = _ensure_month_num(out)

    # --- impute base env cols ---
    for col in env_base_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        tab = stats["impute_tables"][col]
        out = out.merge(tab, on=["region_key", "month_key"], how="left", suffixes=("", "__imp__"))
        imp_col = f"{col}__imp__"
        # if original NaN -> use group median; else keep original
        out[col] = out[col].where(out[col].notna(), out[imp_col])
        out[col] = out[col].fillna(stats["global_medians"][col])
        out = out.drop(columns=[imp_col], errors="ignore")

    # --- SST climatology fit on train ---
    out["sst"] = pd.to_numeric(out["sst"], errors="coerce").fillna(stats["global_medians"].get("sst", 0.0))

    out = out.merge(stats["sst_rm"], on=["region_key", "month_num"], how="left")
    out = out.merge(stats["sst_r"], on=["region_key"], how="left")

    # fill climatology mean/std with region fallback, then global fallback
    out["sst_clim_mean"] = out["sst_clim_mean"].fillna(out["sst_r_mean"]).fillna(stats["sst_global_mean"])
    out["sst_clim_std"] = out["sst_clim_std"].fillna(out["sst_r_std"]).fillna(stats["sst_global_std"])
    out["sst_clim_std"] = out["sst_clim_std"].replace(0, np.nan).fillna(stats["sst_global_std"])

    out["sst_clim_rm"] = out["sst_clim_mean"]
    out["sst_anom"] = out["sst"] - out["sst_clim_mean"]
    out["sst_anom_z"] = (out["sst"] - out["sst_clim_mean"]) / out["sst_clim_std"]

    # drop helper columns (keep sst_clim_rm, anom, z)
    out = out.drop(columns=["sst_clim_mean", "sst_clim_std", "sst_r_mean", "sst_r_std"], errors="ignore")

    # --- logs ---
    out["log_kd490"] = np.log10(pd.to_numeric(out["kd490"], errors="coerce").clip(lower=1e-4))
    out["log_chlor_a"] = np.log10(pd.to_numeric(out["chlor_a"], errors="coerce").clip(lower=1e-4))
    out["log_nflh"] = np.log10(pd.to_numeric(out["nflh"], errors="coerce").clip(lower=1e-4))

    # --- ratios / combos ---
    kd = pd.to_numeric(out["kd490"], errors="coerce").replace(0, np.nan)
    chl = pd.to_numeric(out["chlor_a"], errors="coerce")
    nflh = pd.to_numeric(out["nflh"], errors="coerce")
    out["ratio_chl_kd"] = (chl / kd).replace([np.inf, -np.inf], np.nan)
    out["chl_times_nflh"] = (chl * nflh).replace([np.inf, -np.inf], np.nan)
    out["ratio_nflh_kd"] = (nflh / kd).replace([np.inf, -np.inf], np.nan)

    # --- interactions ---
    if interactions == "basic":
        def _safe_mul(a, b):
            return (pd.to_numeric(out[a], errors="coerce") * pd.to_numeric(out[b], errors="coerce")).replace(
                [np.inf, -np.inf], np.nan
            )
        inter_specs = [
            ("sst_anom", "chlor_a"),
            ("sst_anom", "nflh"),
            ("sst_anom", "fai_mean"),
            ("sst_anom", "kd490"),
            ("sst_anom", "month_sin"),
            ("sst_anom", "month_cos"),
        ]
        for a, b in inter_specs:
            out[f"{a}_x_{b}"] = _safe_mul(a, b)

    return out

def _fit_context_tables_train(df_train_aug: pd.DataFrame, ctx_cols):
    """
    Fit TRAIN-only region-month context mean/std for ctx_cols.
    Returns a dict {col: table_df(region_key, month_key, mean, std)} plus global mean/std fallbacks.
    """
    ctx_tables = {}
    global_mu = {}
    global_sd = {}
    for col in ctx_cols:
        s = pd.to_numeric(df_train_aug[col], errors="coerce")
        global_mu[col] = float(s.mean()) if np.isfinite(s.mean()) else 0.0
        sd = float(s.std()) if np.isfinite(s.std()) else 1.0
        global_sd[col] = sd if sd > 0 else 1.0
        tmp = pd.concat([df_train_aug[["region_key", "month_key"]], s.rename(col)], axis=1)
        tab = (
            tmp.dropna(subset=[col])
            .groupby(["region_key", "month_key"], as_index=False)[col]
            .agg(["mean", "std"])
            .reset_index()
        )
        tab.columns = ["region_key", "month_key", f"{col}_rm_mean", f"{col}_rm_std"]
        ctx_tables[col] = tab
    return {"tables": ctx_tables, "global_mu": global_mu, "global_sd": global_sd}

def _apply_context(df_aug: pd.DataFrame, ctx_cols, ctx_stats: dict):
    """
    Apply TRAIN-only region-month context mean/std to df_aug, producing:
      - {col}_rm_mean, {col}_anom_rm, {col}_z_rm
    """
    out = df_aug.copy()
    for col in ctx_cols:
        tab = ctx_stats["tables"][col]
        out = out.merge(tab, on=["region_key", "month_key"], how="left")
        mu = f"{col}_rm_mean"
        sd = f"{col}_rm_std"
        # fallbacks
        out[mu] = pd.to_numeric(out[mu], errors="coerce").fillna(ctx_stats["global_mu"][col])
        out[sd] = pd.to_numeric(out[sd], errors="coerce").replace(0, np.nan).fillna(ctx_stats["global_sd"][col])
        out[f"{col}_anom_rm"] = pd.to_numeric(out[col], errors="coerce") - out[mu]
        out[f"{col}_z_rm"] = (pd.to_numeric(out[col], errors="coerce") - out[mu]) / out[sd]
        # drop std helper
        out = out.drop(columns=[sd], errors="ignore")
    return out


def _make_fit_cal_split(df_train_full: pd.DataFrame,
                        y_train_full: np.ndarray,
                        g_train: np.ndarray,
                        calib_frac: float,
                        seed: int):
    """
    Robustly split TRAIN into (fit, calib).

    Priority:
      1) GroupShuffleSplit if >=2 unique groups and it yields non-empty splits
      2) StratifiedShuffleSplit on rows (keeps class balance) if possible
      3) Fallback: no calibration split (return all rows for fit, empty for cal)
    """
    n = len(df_train_full)
    if n < 8:
        return np.arange(n), np.array([], dtype=int)

    calib_frac = float(np.clip(calib_frac, 0.05, 0.45))

    uniq_groups = pd.Series(g_train).nunique(dropna=True)
    if uniq_groups >= 2:
        for frac in [calib_frac, 0.15, 0.10, 0.08, 0.05]:
            try:
                gss = GroupShuffleSplit(test_size=frac, random_state=seed)
                tr_fit_idx, tr_cal_idx = next(gss.split(df_train_full, y_train_full, groups=g_train))
                if len(tr_fit_idx) > 0 and len(tr_cal_idx) > 0:
                    y_cal = y_train_full[tr_cal_idx]
                    if len(np.unique(y_cal)) == 2:
                        return tr_fit_idx, tr_cal_idx
            except Exception:
                pass

    if len(np.unique(y_train_full)) == 2:
        for frac in [calib_frac, 0.15, 0.10, 0.08, 0.05]:
            try:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
                tr_fit_idx, tr_cal_idx = next(sss.split(np.zeros(n), y_train_full))
                if len(tr_fit_idx) > 0 and len(tr_cal_idx) > 0:
                    y_cal = y_train_full[tr_cal_idx]
                    if len(np.unique(y_cal)) == 2:
                        return tr_fit_idx, tr_cal_idx
            except Exception:
                pass

    return np.arange(n), np.array([], dtype=int)


# --------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabular_csv", required=True)
    ap.add_argument("--det", nargs="*", default=[])
    ap.add_argument("--outdir", default="runs/fusion/decision_fusion")
    ap.add_argument("--id_col", default="tile")
    ap.add_argument("--group_by", default="scene_id")
    ap.add_argument("--min_pos_per_split", type=int, default=2)
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--max_tries", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv_folds", type=int, default=0)
    ap.add_argument("--cv_time_folds", type=int, default=0)
    ap.add_argument("--coco_json", default="")
    ap.add_argument("--require_overlap", action="store_true")

    # model
    ap.add_argument("--model", choices=["logreg", "catboost"], default="logreg")

    # CatBoost params (safe defaults)
    ap.add_argument("--cb_iters", type=int, default=800)
    ap.add_argument("--cb_depth", type=int, default=6)
    ap.add_argument("--cb_lr", type=float, default=0.05)
    ap.add_argument("--cb_l2", type=float, default=6.0)
    ap.add_argument("--cb_early_stop", type=int, default=80)

    # thresholding
    ap.add_argument(
        "--threshold_policy",
        choices=["f1", "precision", "recall", "fpr", "topfrac", "expected_pos", "prec_at_recall"],
        default="f1",
    )
    ap.add_argument("--target_precision", type=float, default=None)
    ap.add_argument("--target_recall", type=float, default=None)
    ap.add_argument("--target_fpr", type=float, default=None)
    ap.add_argument("--top_frac", type=float, default=None)
    ap.add_argument("--expected_pos_rate", type=float, default=None)
    ap.add_argument("--min_recall", type=float, default=0.60, help="For threshold_policy=prec_at_recall")

    ap.add_argument(
        "--threshold_scope",
        choices=["train", "test_unsupervised"],
        default="train",
    )

    # calibration (split-safe)
    ap.add_argument("--calibrate", choices=["none", "sigmoid", "isotonic"], default="sigmoid")
    ap.add_argument("--calib_frac", type=float, default=0.20, help="Fraction of TRAIN used for calibration fit")
    ap.add_argument("--min_calib_n", type=int, default=40, help="Minimum calibration set size to enable calibration")
    ap.add_argument("--min_calib_pos", type=int, default=8, help="Minimum positives in calibration set")
    ap.add_argument("--min_calib_neg", type=int, default=8, help="Minimum negatives in calibration set")

    # merge/agg
    ap.add_argument("--det_agg", choices=["max", "mean", "median"], default="mean")
    ap.add_argument("--month_backfill", type=int, default=1)
    ap.add_argument("--max_day_gap", type=int, default=12)

    # features
    ap.add_argument("--drop_p_tab", action="store_true")
    ap.add_argument("--intersection_only", action="store_true")
    ap.add_argument("--no_missing_flags", action="store_true")

    # normalization
    ap.add_argument("--normalize_scores", action="store_true")
    ap.add_argument("--normalize_detectors_only", action="store_true")

    # env context
    ap.add_argument("--no_env_context", action="store_true")

    # interactions
    ap.add_argument("--interactions", choices=["none", "basic"], default="basic")

    # logreg weighting knobs
    ap.add_argument("--p_tab_weight", type=float, default=1.8, help="Only affects logreg mode")
    ap.add_argument("--missing_weight", type=float, default=0.5, help="Only affects logreg mode")
    ap.add_argument("--logreg_C", type=float, default=1.0, help="Only affects logreg mode")

    # debug
    ap.add_argument("--shuffle_labels", action="store_true")

    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- load base ---
    id2name, _ = _load_coco_map(args.coco_json)
    base = pd.read_csv(args.tabular_csv)
    base = _clean_columns(base, args.tabular_csv)
    base = _coerce_id(base, args.id_col, args.tabular_csv)
    base[args.id_col] = _normalize_ids(base[args.id_col])

    need = {
        args.id_col, args.group_by, "hab_label", "datetime", "month_key",
        "fai_mean", "rednir_mean", "ndwi_mean", "kd490", "chlor_a", "nflh", "sst",
        "month_sin", "month_cos", "ndwi_std", "rednir_std",
        "p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med",
    }
    if not args.drop_p_tab:
        need.add("p_tab")
    missing = need - set(base.columns)
    if missing:
        raise SystemExit(f"{args.tabular_csv} missing columns: {sorted(missing)}")

    # Keep RAW env columns only here. Derived/context features are created per split (no leakage).
    keep_cols = [
        args.id_col, args.group_by, "datetime", "month_key",
        "fai_mean", "rednir_mean", "ndwi_mean", "kd490", "chlor_a", "nflh", "sst",
        "month_sin", "month_cos", "ndwi_std", "rednir_std",
        "hab_label", "p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med",
    ]
    if not args.drop_p_tab and "p_tab" in base.columns:
        keep_cols.append("p_tab")
    base = base[[c for c in keep_cols if c in base.columns]].copy()

    base[args.group_by] = base[args.group_by].astype(str).map(_canonical_scene_key)
    base = _attach_month_region_keys_for_base(base, args.id_col, args.group_by)
    base["group_for_cv"] = base[args.group_by]

    # env base columns
    env_base_cols = [
        "fai_mean","rednir_mean","ndwi_mean","kd490","chlor_a","nflh","sst",
        "month_sin","month_cos","ndwi_std","rednir_std",
    ]

    # --- merge detectors from separate CSVs ---
    feats_base = []
    det_names = []

    if not args.drop_p_tab and "p_tab" in base.columns:
        feats_base.append("p_tab")

    in_tab_det_cols = ["p_frcnn_r50_med","p_frcnn_mb_med","p_ssd_mb_med"]
    for c in in_tab_det_cols:
        feats_base.append(c); det_names.append(c)
    print(f"[info] Using in-tabular detector columns as features: {det_names}")

    # external detectors
    for spec in args.det:
        if "=" not in spec:
            raise SystemExit("Use name=path.csv for --det")
        name, path = spec.split("=", 1)
        df = pd.read_csv(path)
        df = _clean_columns(df, path)
        df = _coerce_id(df, args.id_col, path)

        if id2name is not None:
            ser = pd.to_numeric(df[args.id_col], errors="coerce").astype("Int64")
            if ser.notna().mean() >= 0.90:
                before = len(df)
                df[args.id_col] = ser.map(
                    lambda k: id2name.get(int(k)) if pd.notna(k) and int(k) in id2name else None
                )
                mapped = df[args.id_col].notna().sum()
                print(f"[info] {path}: COCO mapped {mapped}/{before} ids to filenames.")

        df[args.id_col] = _normalize_ids(df[args.id_col])
        score_col = _guess_score_col(df, args.id_col)
        if score_col != name:
            df = df.rename(columns={score_col: name})
        df = _attach_month_region_keys_for_det(df, args.id_col)

        merged, ok = _merge_on_id(base, df[[args.id_col, name]], args.id_col, name)
        if not ok:
            det_scene_df = df[[args.id_col, name]].copy()
            det_scene_df.columns = ["__file__", name]
            merged, ok = _merge_on_scene(base, det_scene_df, args.group_by, name)
        if not ok:
            merged, ok = _merge_on_month_region(
                base, df[[args.id_col, "month_key", "region_key", name]], name, args.det_agg,
                month_backfill=args.month_backfill
            )
        if not ok:
            merged, ok = _merge_on_nearest_date(
                base, df[[args.id_col, "date_key", "region_key", name]],
                name, max_day_gap=args.max_day_gap, use_region=True
            )
        if not ok:
            print(f"[warn] Detector '{name}' 0% coverage after all strategies.")
            if args.require_overlap:
                raise SystemExit(f"[fatal] '{name}' contributed 0 matches.")

        base = merged
        if name not in base.columns:
            base[name] = np.nan
        feats_base.append(name); det_names.append(name)

    # leak check presence/absence (still useful)
    for c in det_names:
        cov_pos = base.loc[base.hab_label == 1, c].notna().mean()
        cov_neg = base.loc[base.hab_label == 0, c].notna().mean()
        print(f"[leak-check] {c}: coverage pos={cov_pos:.3f}, neg={cov_neg:.3f}")

    # intersection-only
    if args.intersection_only and det_names:
        mask_all = base[det_names].notna().all(axis=1)
        kept = int(mask_all.sum())
        total = int(len(base))
        if kept == 0:
            raise SystemExit("[intersection_only] No rows remain.")
        print(f"[info] intersection_only: keeping {kept}/{total} rows ({kept/total*100:.1f}%).")
        base = base.loc[mask_all].reset_index(drop=True)

    # Debug dump BEFORE split-safe env expansion (raw merged)
    base.to_csv(outdir / "merged_features_debug_raw.csv", index=False)

    X_all = base[feats_base + env_base_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    y_all = base["hab_label"].astype(int).values
    groups = base["group_for_cv"].astype(str).values

    if args.shuffle_labels:
        rng_dbg = np.random.RandomState(12345)
        perm = rng_dbg.permutation(len(base))
        base["hab_label"] = base["hab_label"].values[perm]
        y_all = base["hab_label"].astype(int).values
        print("[debug] Labels have been RANDOMLY SHUFFLED in the DataFrame for leakage check.")

    dataset_tag = Path(args.tabular_csv).stem
    dataset_hash = int(hashlib.sha1(dataset_tag.encode()).hexdigest(), 16) % (10**6)
    final_seed = int(args.seed) + dataset_hash % 100000
    print(f"[debug] Using dataset-dependent seed: {final_seed} (from '{dataset_tag}')")

    # normalization helper (train-fit)
    def _normalize_inplace(train_df: pd.DataFrame, test_df: pd.DataFrame, cols, protect_cols=None):
        protect_cols = set(protect_cols or [])
        cols = [c for c in cols if c not in protect_cols]
        if not cols:
            return []
        mins = train_df[cols].min().replace(np.inf, 0).replace(-np.inf, 0)
        maxs = train_df[cols].max().replace(np.inf, 1).replace(-np.inf, 1)
        spans = (maxs - mins).replace(0, 1.0)
        train_df[cols] = (train_df[cols] - mins) / spans
        test_df[cols] = (test_df[cols] - mins) / spans
        return cols

    def _q(arr, q):
        arr = np.asarray(arr, dtype=float)
        return float(np.quantile(arr, q)) if arr.size else float("nan")

    # fit/predict per split
    def _fit_predict(train_idx, test_idx, tag):
        df_train_full = base.iloc[train_idx].copy()
        df_test = base.iloc[test_idx].copy()

        # --- split-safe env feature engineering (fit on TRAIN, apply to TRAIN+TEST) ---
        env_stats = _fit_env_transforms_train(
            df_train=df_train_full,
            env_base_cols=env_base_cols,
            interactions=args.interactions,
            no_env_context=args.no_env_context,
        )

        df_train_aug = _apply_env_transforms(df_train_full, env_base_cols, env_stats, interactions=args.interactions)
        df_test_aug = _apply_env_transforms(df_test, env_base_cols, env_stats, interactions=args.interactions)

        # context features (train-only tables)
        derived_env_cols = env_stats["derived_env_cols"]
        interaction_cols = env_stats["interaction_cols"]
        env_cols_all = [c for c in (env_base_cols + derived_env_cols + interaction_cols) if c in df_train_aug.columns]

        context_env_cols = []
        if not args.no_env_context:
            ctx_fit = _fit_context_tables_train(df_train_aug, env_cols_all)
            df_train_aug = _apply_context(df_train_aug, env_cols_all, ctx_fit)
            df_test_aug = _apply_context(df_test_aug, env_cols_all, ctx_fit)
            # track names
            for col in env_cols_all:
                context_env_cols.extend([f"{col}_rm_mean", f"{col}_anom_rm", f"{col}_z_rm"])
            print(f"[env-context] added split-safe region-month context for: {env_cols_all}")

        env_feats = env_cols_all + context_env_cols

        # --- build final feature columns for this fold ---
        feat_cols = feats_base + env_feats

        # missing flags: apply AFTER env augmentation
        if args.no_missing_flags:
            for col in feat_cols:
                df_train_aug[col] = pd.to_numeric(df_train_aug[col], errors="coerce").fillna(0.0)
                df_test_aug[col] = pd.to_numeric(df_test_aug[col], errors="coerce").fillna(0.0)
        else:
            miss_tr = df_train_aug[feat_cols].isna().astype(float)
            miss_te = df_test_aug[feat_cols].isna().astype(float)
            miss_cols = [f"{c}_missing" for c in feat_cols]
            miss_tr.columns = miss_cols
            miss_te.columns = miss_cols

            df_train_aug[feat_cols] = df_train_aug[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            df_test_aug[feat_cols] = df_test_aug[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

            df_train_aug = pd.concat([df_train_aug, miss_tr], axis=1)
            df_test_aug = pd.concat([df_test_aug, miss_te], axis=1)

            feat_cols = feat_cols + miss_cols

        # per-split normalization (train-fit)
        if args.normalize_scores:
            if args.normalize_detectors_only:
                protect = ["p_tab"] if ("p_tab" in feat_cols and not args.drop_p_tab) else []
                norm_candidates = det_names + env_feats
                det_env_cols = [c for c in feat_cols if c in norm_candidates and not c.endswith("_missing")]
                used = _normalize_inplace(df_train_aug, df_test_aug, det_env_cols, protect_cols=protect)
                print(f"[norm] normalized (detectors+env only): {used}")
            else:
                cols = [c for c in feat_cols if not c.endswith("_missing")]
                used = _normalize_inplace(df_train_aug, df_test_aug, cols)
                print(f"[norm] normalized (all non-missing-feature cols): {used}")

        # ---- split TRAIN into fit + calib (robust) ----
        g_train = df_train_aug["group_for_cv"].astype(str).values
        y_train_full = df_train_aug["hab_label"].astype(int).values

        tr_fit_idx, tr_cal_idx = _make_fit_cal_split(
            df_train_full=df_train_aug,
            y_train_full=y_train_full,
            g_train=g_train,
            calib_frac=float(args.calib_frac),
            seed=final_seed + 17,
        )

        df_fit = df_train_aug.iloc[tr_fit_idx].copy()
        df_cal = df_train_aug.iloc[tr_cal_idx].copy()

        if len(tr_cal_idx) == 0:
            print("[cal] Calibration split unavailable for this fold (too few groups/classes). Calibration will be disabled.")

        X_fit = df_fit[feat_cols].values
        y_fit = df_fit["hab_label"].astype(int).values
        X_cal = df_cal[feat_cols].values
        y_cal = df_cal["hab_label"].astype(int).values

        X_train_full = df_train_aug[feat_cols].values
        y_train_full = df_train_aug["hab_label"].astype(int).values
        X_test = df_test_aug[feat_cols].values
        y_test = df_test_aug["hab_label"].astype(int).values

        # ---- build model ----
        if args.model == "catboost":
            try:
                from catboost import CatBoostClassifier
            except Exception as e:
                raise SystemExit(
                    "CatBoost import failed. If you see 'numpy.dtype size changed', you have NumPy 2.x "
                    "with a CatBoost build compiled against NumPy 1.x.\n"
                    "Fix:\n"
                    "  python -m pip uninstall -y catboost\n"
                    "  python -m pip install -U 'numpy<2' --force-reinstall\n"
                    "  python -m pip install --no-cache-dir catboost\n"
                    f"\nOriginal error: {e}"
                )

            cb = CatBoostClassifier(
                iterations=args.cb_iters,
                depth=args.cb_depth,
                learning_rate=args.cb_lr,
                l2_leaf_reg=args.cb_l2,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=final_seed,
                verbose=False,
                auto_class_weights="Balanced",
                od_type="Iter",
                od_wait=args.cb_early_stop,
            )

            if len(y_cal) > 0:
                cb.fit(X_fit, y_fit, eval_set=(X_cal, y_cal), use_best_model=True)
            else:
                cb.fit(X_fit, y_fit)

            model = cb

        else:
            # logreg with weighted scaling
            feat_weights = []
            for f in feat_cols:
                if f == "p_tab":
                    feat_weights.append(float(args.p_tab_weight))
                elif f.endswith("_missing"):
                    feat_weights.append(float(args.missing_weight))
                else:
                    feat_weights.append(1.0)
            print(f"[boost] logreg weights: p_tab={args.p_tab_weight}, *_missing={args.missing_weight}, others=1.")

            model = Pipeline(
                [
                    ("scaler", WeightedScaler(weights=feat_weights)),
                    ("clf", LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=final_seed,
                        C=float(args.logreg_C),
                    )),
                ]
            )
            model.fit(X_fit, y_fit)

        # ---- calibration (fit on CAL set) ----
        cal_ok = (
            args.calibrate != "none"
            and len(y_cal) >= int(args.min_calib_n)
            and (y_cal == 1).sum() >= int(args.min_calib_pos)
            and (y_cal == 0).sum() >= int(args.min_calib_neg)
        )

        if cal_ok:
            p_cal_raw = model.predict_proba(X_cal)[:, 1]
            calibrator = _fit_calibrator(args.calibrate, p_cal_raw, y_cal)
            print(f"[cal] fitted {args.calibrate} calibrator on {len(y_cal)} samples "
                  f"(pos={(y_cal==1).sum()}, neg={(y_cal==0).sum()})")
        else:
            if args.calibrate == "none":
                print("[cal] calibration disabled (flag)")
            else:
                print(f"[cal] calibration disabled (insufficient cal data): "
                      f"n={len(y_cal)} pos={(y_cal==1).sum()} neg={(y_cal==0).sum()} "
                      f"(need n>={args.min_calib_n}, pos>={args.min_calib_pos}, neg>={args.min_calib_neg})")
            calibrator = None

        # ---- predict (calibrated) ----
        p_tr_raw = model.predict_proba(X_train_full)[:, 1]
        p_te_raw = model.predict_proba(X_test)[:, 1]

        p_tr = _apply_calibrator(calibrator, p_tr_raw)
        p_te = _apply_calibrator(calibrator, p_te_raw)

        # q50 diagnostics before/after
        pos_tr_raw, neg_tr_raw = p_tr_raw[y_train_full == 1], p_tr_raw[y_train_full == 0]
        pos_tr, neg_tr = p_tr[y_train_full == 1], p_tr[y_train_full == 0]
        print(
            f"[cal-q50:{tag}] raw  train pos q50={_q(pos_tr_raw,0.5):.3f} neg q50={_q(neg_tr_raw,0.5):.3f} | "
            f"cal  train pos q50={_q(pos_tr,0.5):.3f} neg q50={_q(neg_tr,0.5):.3f}"
        )

        # threshold on TRAIN (calibrated)
        prec_tr, rec_tr, thr_tr = precision_recall_curve(y_train_full, p_tr)
        thr_star, how = _pick_threshold_from_policy(
            y_train_full,
            p_tr,
            prec_tr,
            rec_tr,
            thr_tr,
            policy=args.threshold_policy,
            target_precision=args.target_precision,
            target_recall=args.target_recall,
            target_fpr=args.target_fpr,
            top_frac=args.top_frac,
            expected_pos_rate=args.expected_pos_rate,
            min_recall=args.min_recall,
        )
        print(f"[thr] Selected on TRAIN via '{how}': {thr_star:.3f}")

        # optional unsupervised TEST-scope threshold override
        if args.threshold_scope == "test_unsupervised":
            if args.threshold_policy == "expected_pos" and args.expected_pos_rate is not None:
                q = max(0.0, min(1.0, 1.0 - float(args.expected_pos_rate)))
                thr_star = float(np.quantile(p_te, q)) if p_te.size else thr_star
                print(f"[thr] TEST override expected_pos -> q={q:.3f} thr={thr_star:.3f}")
            elif args.threshold_policy == "topfrac" and args.top_frac is not None:
                q = max(0.0, min(1.0, 1.0 - float(args.top_frac)))
                thr_star = float(np.quantile(p_te, q)) if p_te.size else thr_star
                print(f"[thr] TEST override topfrac -> q={q:.3f} thr={thr_star:.3f}")
            elif args.threshold_policy == "fpr" and args.target_fpr is not None:
                q = max(0.0, min(1.0, 1.0 - float(args.target_fpr)))
                thr_star = float(np.quantile(p_te, q)) if p_te.size else thr_star
                print(f"[thr] TEST override fpr≈quantile -> q={q:.3f} thr={thr_star:.3f}")

        # diagnostics: score quantiles
        pos_te, neg_te = p_te[y_test == 1], p_te[y_test == 0]
        print(
            f"[score-q:{tag}] train pos q50={_q(pos_tr,0.5):.3f} q90={_q(pos_tr,0.9):.3f} | "
            f"train neg q90={_q(neg_tr,0.9):.3f}  ||  "
            f"test  pos q50={_q(pos_te,0.5):.3f} q90={_q(pos_te,0.9):.3f} | "
            f"test  neg q90={_q(neg_te,0.9):.3f}"
        )

        # Save a merged debug view for this fold (optional but useful)
        df_train_aug.to_csv(outdir / f"merged_features_debug_{tag}_train.csv", index=False)
        df_test_aug.to_csv(outdir / f"merged_features_debug_{tag}_test.csv", index=False)
        print(f"[debug] wrote fold feature dumps for {tag}")

        ids_te = df_test_aug[[args.id_col, args.group_by, "hab_label", "month_key", "region_key"]].copy()
        auprc, auroc = _eval_and_save(outdir, tag, feat_cols, y_test, p_te, thr_star, ids_te)

        # (Optional) train eval too
        ids_tr = df_train_aug[[args.id_col, args.group_by, "hab_label", "month_key", "region_key"]].copy()
        _eval_and_save(outdir, f"{tag}_train", feat_cols, y_train_full, p_tr, thr_star, ids_tr)

        joblib.dump(
            {
                "model": model,
                "calibrator": calibrator,
                "features": feat_cols,
                "args": vars(args),
            },
            outdir / f"fusion_model_{tag}.joblib",
        )
        return auprc, auroc

    summaries = []

    def _month_to_ordinal(s: str | None) -> int:
        if not isinstance(s, str) or "-" not in s:
            return -10**9
        try:
            y, m = s.split("-")
            return int(y) * 12 + int(m)
        except Exception:
            return -10**9

    if args.cv_time_folds and args.cv_time_folds > 1:
        print(f"[cv-time] Chronological CV with {args.cv_time_folds} folds (min_pos_per_fold={args.min_pos_per_split})")
        base["_month_ord"] = base["month_key"].apply(_month_to_ordinal)
        uniq_months = np.array(sorted([m for m in base["_month_ord"].unique() if m >= 0]))
        if len(uniq_months) < args.cv_time_folds:
            raise SystemExit(f"Not enough unique months ({len(uniq_months)}) for cv_time_folds={args.cv_time_folds}.")

        month_blocks = []
        target_blocks = int(args.cv_time_folds)
        pos_by_month = base.groupby("_month_ord")["hab_label"].sum().to_dict()

        cur = []
        cur_pos = 0
        for m in uniq_months:
            cur.append(m)
            cur_pos += int(pos_by_month.get(m, 0))
            if cur_pos >= args.min_pos_per_split and len(month_blocks) < target_blocks - 1:
                month_blocks.append(np.array(cur))
                cur = []
                cur_pos = 0
        if cur:
            month_blocks.append(np.array(cur))

        print(f"[cv-time] Built {len(month_blocks)} pos-aware blocks (requested={target_blocks}).")
        for fold_id, block in enumerate(month_blocks, 1):
            test_mask = base["_month_ord"].isin(block)
            earliest_test = int(block.min())
            train_mask = base["_month_ord"] < earliest_test
            tr = np.where(train_mask)[0]
            te = np.where(test_mask)[0]

            if len(tr) == 0:
                print(f"[cv-time] Skipping fold {fold_id} (no earlier months to train).")
                continue
            if (base.loc[tr, "hab_label"].sum() < args.min_pos_per_split or
                base.loc[te, "hab_label"].sum() < args.min_pos_per_split):
                print(f"[cv-time] Skipping fold {fold_id} (insufficient positives).")
                continue

            print(f"[CV{fold_id}] (time) train={len(tr)} (pos={int(base.loc[tr,'hab_label'].sum())}) | "
                  f"test={len(te)} (pos={int(base.loc[te,'hab_label'].sum())})")
            auprc, auroc = _fit_predict(tr, te, f"cv{fold_id}")
            summaries.append({"fold": fold_id, "auprc": auprc, "auroc": auroc,
                              "test_pos": int(base.loc[te, "hab_label"].sum()),
                              "test_total": int(len(te))})

        if not summaries:
            raise SystemExit("[cv-time] Could not form any valid time-based folds.")

        df_sum = pd.DataFrame(summaries)
        df_sum.loc["mean"] = {
            "fold": "mean",
            "auprc": df_sum["auprc"].mean(),
            "auroc": df_sum["auroc"].mean(),
            "test_pos": df_sum["test_pos"].sum(),
            "test_total": df_sum["test_total"].sum(),
        }
        df_sum.to_csv(outdir / "summary_cv.csv", index=False)
        print("[cv] Averages:", df_sum.loc["mean"].to_dict())

    elif args.cv_folds and args.cv_folds > 1:
        if not HAS_SGF:
            raise SystemExit("StratifiedGroupKFold not available. Use --cv_time_folds instead.")
        print(f"[cv] StratifiedGroupKFold with {args.cv_folds} folds")
        sgkf = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=final_seed)
        for fold_id, (tr, te) in enumerate(sgkf.split(np.zeros(len(base)), y_all, groups), 1):
            if y_all[tr].sum() < args.min_pos_per_split or y_all[te].sum() < args.min_pos_per_split:
                print(f"[cv] Skipping fold {fold_id} (insufficient positives).")
                continue
            auprc, auroc = _fit_predict(tr, te, f"cv{fold_id}")
            summaries.append({"fold": fold_id, "auprc": auprc, "auroc": auroc,
                              "test_pos": int(y_all[te].sum()), "test_total": int(len(te))})
        if not summaries:
            raise SystemExit("[cv] No valid folds.")
        df_sum = pd.DataFrame(summaries)
        df_sum.loc["mean"] = {
            "fold": "mean",
            "auprc": df_sum["auprc"].mean(),
            "auroc": df_sum["auroc"].mean(),
            "test_pos": df_sum["test_pos"].sum(),
            "test_total": df_sum["test_total"].sum(),
        }
        df_sum.to_csv(outdir / "summary_cv.csv", index=False)
        print("[cv] Averages:", df_sum.loc["mean"].to_dict())

    else:
        rng = np.random.RandomState(final_seed)
        good = None
        for _ in range(1, args.max_tries + 1):
            gss = GroupShuffleSplit(test_size=args.test_size, random_state=int(rng.randint(0, 10_000)))
            tr, te = next(gss.split(np.zeros(len(base)), y_all, groups=groups))
            if y_all[tr].sum() >= args.min_pos_per_split and y_all[te].sum() >= args.min_pos_per_split:
                good = (tr, te)
                break
        if good is None:
            raise SystemExit("Could not find a split with positives in both partitions.")
        tr, te = good
        print(f"[Split] train={len(tr)} (pos={int(y_all[tr].sum())}) | test={len(te)} (pos={int(y_all[te].sum())})")
        auprc, auroc = _fit_predict(tr, te, "holdout")
        pd.DataFrame([{
            "model": f"fusion({args.model})",
            "feats": "+".join(["BASE(" + ",".join(feats_base) + ")", "ENV(split-safe)"]),
            "auprc": auprc,
            "auroc": auroc,
            "test_pos": int(y_all[te].sum()),
            "test_total": int(len(te)),
        }]).to_csv(outdir / "summary.csv", index=False)

    print(f"[debug] Saving outputs to: {outdir.resolve()}")
    print("✓ saved models, metrics, PR/ROC plots, predictions, and merged feature dumps per fold")


if __name__ == "__main__":
    main()
