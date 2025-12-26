#!/usr/bin/env python3
# fit_decision_fusion.py
"""
Decision-level fusion for S2 tabular scores + external detectors.

Assumes the main training table (tabular_csv) has at least:

  tile, scene_id, datetime, month_key,
  fai_mean, rednir_mean, ndwi_mean, kd490, chlor_a, nflh,
  month_sin, month_cos, ndwi_std, rednir_std,
  hab_label,
  p_frcnn_r50_med, p_frcnn_mb_med, p_ssd_mb_med, p_tab

You can also add more detectors via --det name=path.csv

Merge priority for external detector CSVs:
  (1) exact id; (2) scene key; (3) month(+region) agg; (4) nearest date (±k days).

Key features:
  • Optional score normalization learned on TRAIN only:
      --normalize_scores [--normalize_detectors_only]
  • Threshold policies: f1 | precision | recall | fpr | topfrac | expected_pos
      --expected_pos_rate <float in (0,1)>
  • Threshold scope:
      --threshold_scope train | test_unsupervised
     If 'test_unsupervised', threshold is re-picked on TEST by score quantile.
  • Time-based CV: --cv_time_folds K  (train on earlier months, test on month block)
  • Optional region–month env context features (means/anoms/z-scores).
  • Debug flag --shuffle_labels to check for leakage/overfitting.

Outputs (per run):
  outdir/
    - merged_features_debug.csv
    - predictions_*.csv
    - metrics_*.json
    - pr_fusion_*.png, roc_fusion_*.png
    - summary_cv.csv         (for cv_time_folds / cv_folds)
    - summary.csv            (for simple holdout)
"""

import argparse, json, re, warnings, hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import GroupShuffleSplit
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
from sklearn.calibration import CalibratedClassifierCV


# ---------- custom scaler: standardize then apply per-feature weights ----------
class WeightedScaler(StandardScaler):
    """StandardScaler followed by per-feature multipliers.

    This lets us give extra weight to p_tab (or downweight other features)
    while keeping everything otherwise identical to StandardScaler.
    """
    def __init__(self, weights=None):
        super().__init__()
        self.weights = np.asarray(weights) if weights is not None else None

    def transform(self, X, copy=None):
        X_scaled = super().transform(X, copy)
        if self.weights is not None:
            X_scaled = X_scaled * self.weights
        return X_scaled


def _impute_env_features(df: pd.DataFrame, env_cols) -> pd.DataFrame:
    """Fill missing env features per (region_key, month_key) median, then global median."""
    df = df.copy()
    env_cols = [c for c in env_cols if c in df.columns]
    if not env_cols:
        return df
    for col in env_cols:
        if df[col].isna().any():
            df[col] = df.groupby(["region_key", "month_key"])[col].transform(
                lambda s: s.fillna(s.median())
            )
            df[col] = df[col].fillna(df[col].median())
    print(f"[impute] filled missing environmental columns: {env_cols}")
    return df


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

    # month_key: use existing if present, otherwise infer
    if "month_key" not in base.columns:
        if "datetime" in base.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dt = pd.to_datetime(base["datetime"], errors="coerce", utc=True)
            base["month_key"] = dt.dt.strftime("%Y-%m")
        else:
            base["month_key"] = base[group_col].astype(str).map(_extract_month_key_from_scene)

    # date_key
    if "datetime" in base.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt = pd.to_datetime(base["datetime"], errors="coerce", utc=True)
        base["date_key"] = dt.dt.tz_convert(None) if hasattr(dt.dt, "tz_localize") else dt
    else:
        d1d2 = base[group_col].astype(str).map(_extract_dates_from_name)
        base["date_key"] = [_mid_date(a, b) for (a, b) in d1d2]

    # region_key
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

    # month+region
    if {"month_key", "region_key", score_name}.issubset(det_ok.columns) and "region_key" in base.columns:
        det_mr = _agg(det_ok.dropna(subset=["month_key", "region_key"]), ["month_key", "region_key"], score_name, agg)
        m = base.merge(det_mr, on=["month_key", "region_key"], how="left")
        if m[score_name].isna().mean() < 1.0:
            cov = 100.0 * m[score_name].notna().mean()
            print(f"[info] Detector '{score_name}' merged on month+region ({agg}). Coverage={cov:.1f}%")
            return m, True

    # month-only
    if {"month_key", score_name}.issubset(det_ok.columns):
        det_m = _agg(det_ok.dropna(subset=["month_key"]), ["month_key"], score_name, agg)
        m2 = base.merge(det_m, on="month_key", how="left")
        if m2[score_name].isna().mean() < 1.0:
            cov = 100.0 * m2[score_name].notna().mean()
            print(f"[info] Detector '{score_name}' merged on month-only ({agg}). Coverage={cov:.1f}%")
            return m2, True

        # nearest-month backfill
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
):
    thr = np.asarray(thr)

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

    # fallback: max-F1 on train
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


# --------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tabular_csv",
        required=True,
        help="CSV with fusion table (tile,scene_id,datetime,month_key,env cols, hab_label, p_*).",
    )
    ap.add_argument(
        "--det",
        nargs="*",
        default=[],
        help="named detector CSVs: name=path.csv (must contain id_col and a score column)",
    )
    ap.add_argument("--outdir", default="runs/fusion/decision_fusion")
    ap.add_argument("--id_col", default="tile")
    ap.add_argument("--group_by", default="scene_id")
    ap.add_argument("--min_pos_per_split", type=int, default=2)
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--max_tries", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv_folds", type=int, default=0)
    ap.add_argument("--cv_time_folds", type=int, default=0)
    ap.add_argument("--coco_json", default="", help="COCO JSON (chip_id → file_name)")
    ap.add_argument("--require_overlap", action="store_true")

    # thresholding
    ap.add_argument(
        "--threshold_policy",
        choices=["f1", "precision", "recall", "fpr", "topfrac", "expected_pos"],
        default="f1",
    )
    ap.add_argument("--target_precision", type=float, default=None)
    ap.add_argument("--target_recall", type=float, default=None)
    ap.add_argument("--target_fpr", type=float, default=None)
    ap.add_argument("--top_frac", type=float, default=None)
    ap.add_argument(
        "--expected_pos_rate",
        type=float,
        default=None,
        help="Expected positive rate used by threshold_policy=expected_pos.",
    )
    ap.add_argument(
        "--threshold_scope",
        choices=["train", "test_unsupervised"],
        default="train",
        help="Finalize threshold on TRAIN (default) or on TEST by score-quantile (unsupervised).",
    )

    # calibration
    ap.add_argument("--calibrate", choices=["none", "isotonic", "platt"], default="none")

    # merge/agg
    ap.add_argument("--det_agg", choices=["max", "mean", "median"], default="mean")
    ap.add_argument("--month_backfill", type=int, default=1)
    ap.add_argument("--max_day_gap", type=int, default=12)

    # features
    ap.add_argument("--drop_p_tab", action="store_true")
    ap.add_argument("--intersection_only", action="store_true")
    ap.add_argument(
        "--no_missing_flags",
        action="store_true",
        help="If set, just fill NaNs with 0.0. Otherwise add *_missing flags.",
    )

    # normalization
    ap.add_argument(
        "--normalize_scores",
        action="store_true",
        help="Apply train-fitted min-max normalization to scores.",
    )
    ap.add_argument(
        "--normalize_detectors_only",
        action="store_true",
        help="If set, normalize only detector+env columns; keep p_tab as-is.",
    )

    # env context
    ap.add_argument(
        "--no_env_context",
        action="store_true",
        help="Disable region-month env context features (_rm_mean/_anom_rm/_z_rm).",
    )

    # debug
    ap.add_argument(
        "--shuffle_labels",
        action="store_true",
        help="DEBUG: randomly permute labels to sanity-check for leakage/overfitting.",
    )

    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- load base ---
    id2name, _ = _load_coco_map(args.coco_json)
    base = pd.read_csv(args.tabular_csv)
    base = _clean_columns(base, args.tabular_csv)
    base = _coerce_id(base, args.id_col, args.tabular_csv)
    base[args.id_col] = _normalize_ids(base[args.id_col])

    # required cols
    need = {
        args.id_col,
        args.group_by,
        "hab_label",
        "datetime",
        "month_key",
        "fai_mean",
        "rednir_mean",
        "ndwi_mean",
        "kd490",
        "chlor_a",
        "nflh",
        "sst",
        "month_sin",
        "month_cos",
        "ndwi_std",
        "rednir_std",
        "p_frcnn_r50_med",
        "p_frcnn_mb_med",
        "p_ssd_mb_med",
    }
    if not args.drop_p_tab:
        need.add("p_tab")
    missing = need - set(base.columns)
    if missing:
        raise SystemExit(f"{args.tabular_csv} missing columns: {sorted(missing)}")

    # keep only the known fusion columns + label + ids/time
    keep_cols = [
        args.id_col,
        args.group_by,
        "datetime",
        "month_key",
        "fai_mean",
        "rednir_mean",
        "ndwi_mean",
        "kd490",
        "chlor_a",
        "nflh",
        "sst",
        "month_sin",
        "month_cos",
        "ndwi_std",
        "rednir_std",
        "hab_label",
        "p_frcnn_r50_med",
        "p_frcnn_mb_med",
        "p_ssd_mb_med",
    ]
    if not args.drop_p_tab and "p_tab" in base.columns:
        keep_cols.append("p_tab")

    keep_cols = [c for c in keep_cols if c in base.columns]
    base = base[keep_cols].copy()

    # basic canonicalization + region/month/date keys
    base[args.group_by] = base[args.group_by].astype(str).map(_canonical_scene_key)
    base = _attach_month_region_keys_for_base(base, args.id_col, args.group_by)
    base["group_for_cv"] = base["region_key"]
    groups = base["group_for_cv"].astype(str).values

    # env base columns
    env_base_cols = [
        "fai_mean",
        "rednir_mean",
        "ndwi_mean",
        "kd490",
        "chlor_a",
        "nflh",
        "sst",
        "month_sin",
        "month_cos",
        "ndwi_std",
        "rednir_std",
    ]

    # impute env features
    base = _impute_env_features(base, env_base_cols)

    # --- simple derived env features (logs + ratios) ---
    derived_env_cols = []

    # log transforms
    for col in ["kd490", "chlor_a", "nflh"]:
        if col in base.columns:
            new_col = f"log_{col}"
            base[new_col] = np.log10(base[col].clip(lower=1e-4))
            derived_env_cols.append(new_col)

    # ratios/combinations
    if {"chlor_a", "kd490"}.issubset(base.columns):
        base["ratio_chl_kd"] = base["chlor_a"] / base["kd490"].replace(0, np.nan)
        derived_env_cols.append("ratio_chl_kd")
    if {"chlor_a", "nflh"}.issubset(base.columns):
        base["chl_times_nflh"] = base["chlor_a"] * base["nflh"]
        derived_env_cols.append("chl_times_nflh")
    if {"nflh", "kd490"}.issubset(base.columns):
        base["ratio_nflh_kd"] = base["nflh"] / base["kd490"].replace(0, np.nan)
        derived_env_cols.append("ratio_nflh_kd")

    # region-month env context
    context_env_cols = []
    if not args.no_env_context:
        ctx_cols = [c for c in env_base_cols + derived_env_cols if c in base.columns]
        for col in ctx_cols:
            grp = base.groupby(["region_key", "month_key"])[col]
            mean_rm = grp.transform("mean")
            std_rm = grp.transform("std").replace(0, np.nan)
            mcol = f"{col}_rm_mean"
            acol = f"{col}_anom_rm"
            zcol = f"{col}_z_rm"
            base[mcol] = mean_rm
            base[acol] = base[col] - mean_rm
            base[zcol] = (base[col] - mean_rm) / std_rm
            context_env_cols.extend([mcol, acol, zcol])
        print(f"[env-context] added region-month context cols for: {ctx_cols}")

    # --- merge detectors from separate CSVs ---
    feats = []
    det_names = []

    if not args.drop_p_tab and "p_tab" in base.columns:
        feats.append("p_tab")

    # detectors already in table
    in_tab_det_cols = [
        "p_frcnn_r50_med",
        "p_frcnn_mb_med",
        "p_ssd_mb_med",
    ]
    for c in in_tab_det_cols:
        if c in base.columns:
            feats.append(c)
            det_names.append(c)

    if det_names:
        print(f"[info] Using in-tabular detector columns as features: {det_names}")

    # external detectors
    for spec in args.det:
        if "=" not in spec:
            raise SystemExit("Use name=path.csv for --det (e.g., frcnn_r50=runs/fusion/p_frcnn_r50.csv)")
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
        feats.append(name)
        det_names.append(name)

    # leak check presence/absence
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

    # optional: drop p_tab
    if args.drop_p_tab and "p_tab" in feats:
        feats = [f for f in feats if f != "p_tab"]
        print("[info] Dropping p_tab; using detectors+env only:", feats)

    # ---- add env features to feature list explicitly ----
    env_feats = [c for c in env_base_cols + derived_env_cols + context_env_cols if c in base.columns]
    for ef in env_feats:
        if ef not in feats:
            feats.append(ef)

    # save raw merged for debugging
    base.to_csv(outdir / "merged_features_debug.csv", index=False)
    print(f"[info] Final feature columns ({len(feats)}): {feats}")

    # missing handling
    if args.no_missing_flags:
        for col in feats:
            base[col] = base[col].astype(float).fillna(0.0)
    else:
        miss_df = base[feats].isna().astype(float)
        miss_df.columns = [f"{c}_missing" for c in feats]
        base[feats] = base[feats].astype(float).fillna(0.0)
        base = pd.concat([base, miss_df], axis=1)
        feats = feats + list(miss_df.columns)

    X_all = base[feats].values
    y_all = base["hab_label"].astype(int).values
    groups = base["group_for_cv"].astype(str).values

    # debug: optionally shuffle labels to test for leakage/overfitting
    if args.shuffle_labels:
        rng_dbg = np.random.RandomState(12345)
        perm = rng_dbg.permutation(len(base))
        base["hab_label"] = base["hab_label"].values[perm]
        y_all = base["hab_label"].astype(int).values  # update view
        print("[debug] Labels have been RANDOMLY SHUFFLED in the DataFrame for leakage check.")


    # dataset-dependent seed
    dataset_tag = Path(args.tabular_csv).stem
    dataset_hash = int(hashlib.sha1(dataset_tag.encode()).hexdigest(), 16) % (10**6)
    final_seed = int(args.seed) + dataset_hash % 100000
    print(f"[debug] Using dataset-dependent seed: {final_seed} (from '{dataset_tag}')")

    # ------------- normalization helper (train-fit) -------------
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

    # ------------- fit/predict per split -------------
    def _fit_predict(train_idx, test_idx, tag):
        df_train = base.iloc[train_idx].copy()
        df_test = base.iloc[test_idx].copy()

        feat_cols = feats.copy()

        # per-split normalization (train-fit)
        if args.normalize_scores:
            if args.normalize_detectors_only:
                protect = ["p_tab"] if ("p_tab" in feat_cols and not args.drop_p_tab) else []
                # detectors + env, excluding missing flags
                norm_candidates = det_names + env_feats
                det_env_cols = [c for c in feat_cols
                                if c in norm_candidates and not c.endswith("_missing")]
                used = _normalize_inplace(df_train, df_test, det_env_cols, protect_cols=protect)
                print(f"[norm] normalized (detectors+env only): {used}")
            else:
                cols = [c for c in feat_cols if not c.endswith("_missing")]
                used = _normalize_inplace(df_train, df_test, cols)
                print(f"[norm] normalized (all non-missing-feature cols): {used}")

        X_train = df_train[feat_cols].values
        y_train = df_train["hab_label"].astype(int).values
        X_test = df_test[feat_cols].values
        y_test = df_test["hab_label"].astype(int).values

        # ---- feature weights (bias p_tab, downweight missing flags a bit) ----
        feat_weights = []
        for f in feat_cols:
            if f == "p_tab":
                feat_weights.append(3.0)   # main knob to make p_tab dominate more
            elif f.endswith("_missing"):
                feat_weights.append(0.5)
            else:
                feat_weights.append(1.0)
        print(f"[boost] applied feature weights (p_tab=3, *_missing=0.5, others=1).")

        base_clf = Pipeline(
            [
                ("scaler", WeightedScaler(weights=feat_weights)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=final_seed,
                    ),
                ),
            ]
        )

        if args.calibrate != "none":
            method = "isotonic" if args.calibrate == "isotonic" else "sigmoid"
            try:
                clf = CalibratedClassifierCV(base_estimator=base_clf, method=method, cv=3)
            except TypeError:
                clf = CalibratedClassifierCV(estimator=base_clf, method=method, cv=3)
        else:
            clf = base_clf

        clf.fit(X_train, y_train)

        # TRAIN scores (for train-scope threshold)
        p_tr = clf.predict_proba(X_train)[:, 1]
        p_tr = np.clip(p_tr, 1e-6, 1 - 1e-6)
        prec_tr, rec_tr, thr_tr = precision_recall_curve(y_train, p_tr)
        thr_star, how = _pick_threshold_from_policy(
            y_train,
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
        )
        print(f"[thr] Selected on TRAIN via '{how}': {thr_star:.3f}")

        # TEST scores
        p_te = clf.predict_proba(X_test)[:, 1]
        p_te = np.clip(p_te, 1e-6, 1 - 1e-6)

        # optional unsupervised TEST-scope threshold override
        if args.threshold_scope == "test_unsupervised":
            if args.threshold_policy == "expected_pos" and args.expected_pos_rate is not None:
                q = max(0.0, min(1.0, 1.0 - float(args.expected_pos_rate)))
                thr_test = float(np.quantile(p_te, q)) if p_te.size else thr_star
                print(
                    f"[thr] Overriding thr* on TEST via expected_pos(rate={args.expected_pos_rate:.3f}) "
                    f"=> test-quantile={q:.3f} thr_test={thr_test:.3f}"
                )
                thr_star = thr_test
            elif args.threshold_policy == "topfrac" and args.top_frac is not None:
                q = max(0.0, min(1.0, 1.0 - float(args.top_frac)))
                thr_test = float(np.quantile(p_te, q)) if p_te.size else thr_star
                print(
                    f"[thr] Overriding thr* on TEST via topfrac(frac={args.top_frac:.3f}) "
                    f"=> test-quantile={q:.3f} thr_test={thr_test:.3f}"
                )
                thr_star = thr_test
            elif args.threshold_policy == "fpr" and args.target_fpr is not None:
                q = max(0.0, min(1.0, 1.0 - float(args.target_fpr)))
                thr_test = float(np.quantile(p_te, q)) if p_te.size else thr_star
                print(
                    f"[thr] Overriding thr* on TEST via fpr≈quantile(target_fpr={args.target_fpr:.3f}) "
                    f"=> test-quantile={q:.3f} thr_test={thr_test:.3f}"
                )
                thr_star = thr_test
            # precision/recall/f1 need labels; keep train-based

        # diagnostics: score quantiles
        def _q(arr, q):
            return float(np.quantile(arr, q)) if arr.size else float("nan")

        pos_tr, neg_tr = p_tr[y_train == 1], p_tr[y_train == 0]
        pos_te, neg_te = p_te[y_test == 1], p_te[y_test == 0]
        print(
            f"[score-q:{tag}] train pos q50={_q(pos_tr,0.5):.3f} q90={_q(pos_tr,0.9):.3f} | "
            f"train neg q90={_q(neg_tr,0.9):.3f}  ||  "
            f"test  pos q50={_q(pos_te,0.5):.3f} q90={_q(pos_te,0.9):.3f} | "
            f"test  neg q90={_q(neg_te,0.9):.3f}"
        )

        ids_te = df_test[[args.id_col, args.group_by, "hab_label", "month_key", "region_key"]].copy()
        auprc, auroc = _eval_and_save(outdir, tag, feat_cols, y_test, p_te, thr_star, ids_te)

        # train diagnostics (optional)
        ids_tr = df_train[[args.id_col, args.group_by, "hab_label", "month_key", "region_key"]].copy()
        _eval_and_save(outdir, f"{tag}_train", feat_cols, y_train, p_tr, thr_star, ids_tr)

        joblib.dump(
            {"model": clf, "features": feat_cols, "args": vars(args)},
            outdir / f"fusion_model_{tag}.joblib",
        )
        return auprc, auroc

    summaries = []

    # --------- CV strategies ----------
    def _month_to_ordinal(s: str | None) -> int:
        if not isinstance(s, str) or "-" not in s:
            return -10**9
        try:
            y, m = s.split("-")
            return int(y) * 12 + int(m)
        except Exception:
            return -10**9

    if args.cv_time_folds and args.cv_time_folds > 1:
        print(
            f"[cv-time] Chronological CV with {args.cv_time_folds} folds "
            f"(min_pos_per_fold={args.min_pos_per_split})"
        )
        base["_month_ord"] = base["month_key"].apply(_month_to_ordinal)
        uniq_months = np.array(sorted([m for m in base["_month_ord"].unique() if m >= 0]))
        if len(uniq_months) < args.cv_time_folds:
            raise SystemExit(
                f"Not enough unique months ({len(uniq_months)}) for cv_time_folds={args.cv_time_folds}."
            )

        month_blocks = np.array_split(uniq_months, args.cv_time_folds)
        for fold_id, block in enumerate(month_blocks, 1):
            test_mask = base["_month_ord"].isin(block)
            earliest_test = int(block.min())
            train_mask = base["_month_ord"] < earliest_test
            tr = np.where(train_mask)[0]
            te = np.where(test_mask)[0]

            if len(tr) == 0:
                print(f"[cv-time] Skipping fold {fold_id} (no earlier months to train).")
                continue
            if (
                base.loc[tr, "hab_label"].sum() < args.min_pos_per_split
                or base.loc[te, "hab_label"].sum() < args.min_pos_per_split
            ):
                print(f"[cv-time] Skipping fold {fold_id} (insufficient positives).")
                continue

            print(
                f"[CV{fold_id}] (time) train={len(tr)} (pos={int(base.loc[tr,'hab_label'].sum())}) | "
                f"test={len(te)} (pos={int(base.loc[te,'hab_label'].sum())}) | "
                f"months_test={[int(b) for b in block]}"
            )
            auprc, auroc = _fit_predict(tr, te, f"cv{fold_id}")
            summaries.append(
                {
                    "fold": fold_id,
                    "auprc": auprc,
                    "auroc": auroc,
                    "test_pos": int(base.loc[te, "hab_label"].sum()),
                    "test_total": int(len(te)),
                }
            )

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
        print(f"[cv] StratifiedGroupKFold with {args.cv_folds} folds (min_pos_per_fold={args.min_pos_per_split})")
        sgkf = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=final_seed)
        for fold_id, (tr, te) in enumerate(sgkf.split(X_all, y_all, groups), 1):
            if y_all[tr].sum() < args.min_pos_per_split or y_all[te].sum() < args.min_pos_per_split:
                print(f"[cv] Skipping fold {fold_id} (insufficient positives).")
                continue
            print(
                f"[CV{fold_id}] train={len(tr)} (pos={int(y_all[tr].sum())}) | "
                f"test={len(te)} (pos={int(y_all[te].sum())})"
            )
            auprc, auroc = _fit_predict(tr, te, f"cv{fold_id}")
            summaries.append(
                {
                    "fold": fold_id,
                    "auprc": auprc,
                    "auroc": auroc,
                    "test_pos": int(y_all[te].sum()),
                    "test_total": int(len(te)),
                }
            )
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
        # simple group holdout with positives on both sides
        rng = np.random.RandomState(final_seed)
        good = None
        for _ in range(1, args.max_tries + 1):
            gss = GroupShuffleSplit(test_size=args.test_size, random_state=int(rng.randint(0, 10_000)))
            tr, te = next(gss.split(X_all, y_all, groups=groups))
            if y_all[tr].sum() >= args.min_pos_per_split and y_all[te].sum() >= args.min_pos_per_split:
                good = (tr, te)
                break
        if good is None:
            raise SystemExit("Could not find a split with positives in both partitions.")
        tr, te = good
        print(
            f"[Split] train={len(tr)} (pos={int(y_all[tr].sum())}) | "
            f"test={len(te)} (pos={int(y_all[te].sum())})"
        )
        auprc, auroc = _fit_predict(tr, te, "holdout")
        pd.DataFrame(
            [
                {
                    "model": "fusion(logreg)",
                    "feats": "+".join(feats),
                    "auprc": auprc,
                    "auroc": auroc,
                    "test_pos": int(y_all[te].sum()),
                    "test_total": int(len(te)),
                }
            ]
        ).to_csv(outdir / "summary.csv", index=False)

    print(f"[debug] Saving outputs to: {outdir.resolve()}")
    print("✓ saved models, metrics, PR/ROC plots, predictions, and merged_features_debug.csv")


if __name__ == "__main__":
    main()
