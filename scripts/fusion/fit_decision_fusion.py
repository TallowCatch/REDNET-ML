#!/usr/bin/env python3
# fit_decision_fusion.py
import argparse, json, re, hashlib, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
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

# ---------------- plotting ----------------
def _pr_envelope(rec, prec):
    """Make precision monotone non-increasing vs recall (upper envelope).
    Also collapses duplicate recall points to a single best-precision point."""
    if len(rec) == 0:
        return rec, prec
    rec = np.asarray(rec, dtype=float)
    prec = np.asarray(prec, dtype=float)
    # collapse to unique recall by taking max precision at each recall
    df = pd.DataFrame({"rec": rec, "prec": prec})
    df = df.groupby("rec", as_index=False)["prec"].max().sort_values("rec")
    rec_u = df["rec"].values
    prec_u = df["prec"].values
    # enforce monotone envelope from right to left
    prec_env = np.maximum.accumulate(prec_u[::-1])[::-1]
    return rec_u, prec_env

def _pr_plot(rec, prec, auprc, base, outpng):
    plt.close("all")
    # remove (0,1) artifact if present
    if len(rec) and rec[0] == 0 and len(prec) and prec[0] == 1:
        rec, prec = rec[1:], prec[1:]
    rec, prec = _pr_envelope(rec, prec)

    plt.figure(figsize=(5.4, 4.2))
    plt.step(rec, prec, where="post", lw=1.8, alpha=0.95)
    # only scatter when there aren't too many points
    if len(rec) <= 200:
        plt.scatter(rec, prec, s=8, alpha=0.8)
    if base is not None:
        plt.hlines(base, 0, 1, linestyles="--", alpha=0.6)
    base_str = f"{base:.3f}" if base is not None else "N/A"
    plt.title(f"PR (AUPRC={auprc:.3f}, baseline={base_str})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.tight_layout(); plt.savefig(outpng, dpi=160); plt.close()

def _roc_plot(fpr, tpr, auroc, outpng):
    plt.close("all")
    plt.figure(figsize=(5.4, 4.2))
    plt.plot(fpr, tpr, lw=1.8)
    plt.plot([0, 1], [0, 1], "--", alpha=0.6)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC (AUROC={auroc:.3f})")
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.tight_layout(); plt.savefig(outpng, dpi=160); plt.close()

# --------------- helpers ------------------
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

def _normalize_ids(s: pd.Series) -> pd.Series:
    return s.astype(str).apply(lambda x: Path(str(x)).name)

def _swap_ext(name: str) -> str:
    p = Path(name)
    if p.suffix.lower() in (".jpg", ".jpeg"): return p.with_suffix(".png").name
    if p.suffix.lower() == ".png":           return p.with_suffix(".jpg").name
    return p.name

# ----- canonical scene/month/region keys -----
REGION_RE  = re.compile(r"r\d{3}_c\d{3}", re.IGNORECASE)      # detector grid: r000_c000
MGRS_RE    = re.compile(r"T\d{2}[A-Z]{3}")                    # Sentinel-2 MGRS, e.g., T39QXG
DATE8_RE   = re.compile(r"(20\d{2})(\d{2})(\d{2})")           # YYYYMMDD
MONTH_RE   = re.compile(r"(20\d{2})[._-]?(\d{2})")            # YYYY-MM or YYYYMM
RANGE_RE   = re.compile(r"(20\d{2})(\d{2})(\d{2})_(20\d{2})(\d{2})(\d{2})")  # YYYYMMDD_YYYYMMDD

def _canonical_scene_key(x: str) -> str:
    stem = Path(str(x)).stem
    stem = re.sub(r"_(\d{4})$", "", stem)  # drop _0000 suffixes
    return stem

def _extract_month_key_from_scene(scene_id: str) -> str | None:
    m = DATE8_RE.search(scene_id)
    if m: return f"{m.group(1)}-{m.group(2)}"
    m2 = MONTH_RE.search(scene_id)
    return f"{m2.group(1)}-{m2.group(2)}" if m2 else None

def _extract_month_key_from_modis(fname: str) -> str | None:
    r = RANGE_RE.search(fname)
    if r: return f"{r.group(1)}-{r.group(2)}"
    m = MONTH_RE.search(fname)
    return f"{m.group(1)}-{m.group(2)}" if m else None

def _extract_region_key(s: str) -> str | None:
    m = REGION_RE.search(s)
    if m: return m.group(0).lower()
    m2 = MGRS_RE.search(s)
    if m2: return m2.group(0).upper()
    return None

def _guess_score_col(df: pd.DataFrame, id_col: str) -> str:
    numeric = [c for c in df.columns if c != id_col and pd.api.types.is_numeric_dtype(df[c])]
    cand = [c for c in numeric if str(c).lower().startswith("p_") or "score" in str(c).lower()]
    cand = [c for c in cand if not str(c).lower().endswith("count")]
    if cand: return cand[0]
    numeric = [c for c in numeric if not str(c).lower().endswith("count")]
    if numeric: return numeric[0]
    for c in df.columns:
        if c != id_col: return c
    raise SystemExit("Could not guess score column.")

def _load_coco_map(coco_json: str):
    if not coco_json: return None, None
    with open(coco_json, "r") as f:
        coco = json.load(f)
    id2name = {int(im["id"]): str(im["file_name"]) for im in coco["images"]}
    name2id = {str(v): int(k) for k, v in id2name.items()}
    return id2name, name2id

def _maybe_apply_coco_map(df: pd.DataFrame, id_col: str, id2name: dict | None, src: str) -> pd.DataFrame:
    df = df.copy()
    if id2name is None:
        df[id_col] = _normalize_ids(df[id_col]); return df
    ser = df[id_col]
    numeric_like = pd.to_numeric(ser, errors="coerce"); n_numeric = numeric_like.notna().sum()
    if n_numeric / max(1, len(ser)) >= 0.90:
        before = len(df)
        keys = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
        df[id_col] = keys.map(lambda k: id2name.get(int(k)) if pd.notna(k) and int(k) in id2name else None)
        mapped = df[id_col].notna().sum()
        if mapped == 0:
            print(f"[warn] {src}: COCO mapping found 0 matches; check instances_*.json.")
        else:
            print(f"[info] {src}: COCO mapped {mapped}/{before} ids to filenames.")
    df[id_col] = _normalize_ids(df[id_col].fillna(""))
    return df

# ----- merge strategies -----
def _merge_on_id(base: pd.DataFrame, det_df: pd.DataFrame, id_col: str, score_name: str):
    merged = base.merge(det_df[[id_col, score_name]], on=id_col, how="left")
    na = merged[score_name].isna().sum()
    if na < len(merged):
        print(f"[info] Detector '{score_name}' merged on {id_col}. Missing: {na}/{len(merged)} "
              f"(coverage={(len(merged)-na)/len(merged)*100:.1f}%).")
        return merged, True
    # retry ext swap
    det_tmp = det_df.copy(); det_tmp[id_col] = det_tmp[id_col].apply(_swap_ext)
    merged2 = base.merge(det_tmp[[id_col, score_name]], on=id_col, how="left")
    na2 = merged2[score_name].isna().sum()
    if na2 < len(merged2):
        print(f"[info] Detector '{score_name}' merged on {id_col} via ext swap. Missing: {na2}/{len(merged2)}.")
        return merged2, True
    return base, False

def _merge_on_scene(base: pd.DataFrame, det_df: pd.DataFrame, group_col: str, score_name: str):
    if group_col not in base.columns: return base, False
    det_tmp = det_df.copy()
    if det_tmp.shape[1] < 2:
        print(f"[warn] scene-merge: detector frame has <2 columns: {list(det_tmp.columns)}"); return base, False
    det_tmp = det_tmp.iloc[:, :2].copy(); det_tmp.columns = ["__file__", score_name]
    det_tmp["__scene_key__"] = det_tmp["__file__"].astype(str).map(_canonical_scene_key)
    base2 = base.copy(); base2["__scene_key__"] = base[group_col].astype(str).map(_canonical_scene_key)
    merged = base2.merge(det_tmp[["__scene_key__", score_name]], on="__scene_key__", how="left") \
                  .drop(columns=["__scene_key__"], errors="ignore")
    na = merged[score_name].isna().sum()
    if na < len(merged):
        cov = 100.0 * (len(merged) - na) / max(1, len(merged))
        print(f"[info] Detector '{score_name}' merged on scene. Missing: {na}/{len(merged)} (coverage={cov:.1f}%).")
        return merged, True
    # try ext swap path -> scene key
    det_tmp2 = det_df.iloc[:, :2].copy(); det_tmp2.columns = ["__file__", score_name]
    det_tmp2["__file__"] = det_tmp2["__file__"].map(_swap_ext)
    det_tmp2["__scene_key__"] = det_tmp2["__file__"].astype(str).map(_canonical_scene_key)
    merged2 = base2.merge(det_tmp2[["__scene_key__", score_name]], on="__scene_key__", how="left") \
                   .drop(columns=["__scene_key__"], errors="ignore")
    na2 = merged2[score_name].isna().sum()
    if na2 < len(merged2):
        cov2 = 100.0 * (len(merged2) - na2) / max(1, len(merged2))
        print(f"[info] Detector '{score_name}' merged on scene via ext swap. Missing: {na2}/{len(merged2)} (coverage={cov2:.1f}%).")
        return merged2, True
    return base, False

def _attach_month_region_keys_for_base(base: pd.DataFrame, id_col: str, group_col: str):
    base = base.copy()
    # prefer explicit datetime column if present (handles your 8-day cadence cleanly)
    if "datetime" in base.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt = pd.to_datetime(base["datetime"], errors="coerce", utc=True)
        base["month_key"] = dt.dt.strftime("%Y-%m")
    else:
        base["month_key"] = base[group_col].astype(str).map(_extract_month_key_from_scene)

    # region: r###_c### or MGRS (TxxYYY)
    base["region_key"] = base[group_col].astype(str).map(_extract_region_key)
    mask = base["region_key"].isna()
    if mask.any():
        base.loc[mask, "region_key"] = base.loc[mask, id_col].astype(str).map(_extract_region_key)
    return base

def _attach_month_region_keys_for_det(det_df: pd.DataFrame, id_col: str, det_name: str):
    det = det_df.copy()
    det["month_key"]  = det[id_col].astype(str).map(_extract_month_key_from_modis)
    det["region_key"] = det[id_col].astype(str).map(_extract_region_key)
    return det

def _agg(df: pd.DataFrame, by_cols, score_name: str, how: str) -> pd.DataFrame:
    if how == "max":
        return df.groupby(by_cols, as_index=False)[score_name].max()
    if how == "mean":
        return df.groupby(by_cols, as_index=False)[score_name].mean()
    if how == "median":
        return df.groupby(by_cols, as_index=False)[score_name].median()
    raise ValueError(f"Unknown agg '{how}'")

def _merge_on_month_region(base: pd.DataFrame, det_df: pd.DataFrame, score_name: str, agg: str):
    det_ok = det_df.copy()

    # 1) month+region (deduplicated)
    if {"month_key", "region_key", score_name}.issubset(det_ok.columns):
        det_mr = _agg(det_ok.dropna(subset=["month_key", "region_key"]),
                      ["month_key", "region_key"], score_name, agg)
        merged = base.merge(det_mr, on=["month_key", "region_key"], how="left")
        na = merged[score_name].isna().sum()
        if na < len(merged):
            cov = 100.0 * (len(merged) - na) / max(1, len(merged))
            print(f"[info] Detector '{score_name}' merged on month+region ({agg}). "
                  f"Missing: {na}/{len(merged)} (coverage={cov:.1f}%).")
            return merged, True

    # 2) month-only (deduplicated)
    if {"month_key", score_name}.issubset(det_ok.columns):
        det_m = _agg(det_ok.dropna(subset=["month_key"]), ["month_key"], score_name, agg)
        merged2 = base.merge(det_m, on="month_key", how="left")
        na2 = merged2[score_name].isna().sum()
        if na2 < len(merged2):
            cov2 = 100.0 * (len(merged2) - na2) / max(1, len(merged2))
            print(f"[info] Detector '{score_name}' merged on month-only ({agg}). "
                  f"Missing: {na2}/{len(merged2)} (coverage={cov2:.1f}%).")
            return merged2, True

    return base, False

# ----- threshold & eval -----
def _pick_threshold(prec, rec, thr, target_precision=None, target_recall=None):
    thr_star = None
    if target_precision is not None:
        cand = np.where(prec[:-1] >= target_precision)[0]
        if len(cand):
            thr_star = float(thr[cand[0]])
            print(f"[thr] Using threshold for precision≥{target_precision}: {thr_star:.3f}")
    if thr_star is None and target_recall is not None:
        cand = np.where(rec[:-1] >= target_recall)[0]
        if len(cand):
            thr_star = float(thr[cand[-1]])
            print(f"[thr] Using threshold for recall≥{target_recall}: {thr_star:.3f}")
    if thr_star is None:
        f1s = (2 * prec * rec / (prec + rec + 1e-12))[:-1]
        i = int(np.argmax(f1s)) if len(f1s) else 0
        thr_star = float(thr[i]) if len(thr) else 0.5
        print(f"[thr] Using max-F1 threshold: {thr_star:.3f}")
    return thr_star

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
    base_rate, auprc, auroc, prec, rec, thr, fpr, tpr = _safe_metrics(y_true, p)
    yhat = (p >= thr_star).astype(int)
    cm = confusion_matrix(y_true, yhat).tolist()
    rep = classification_report(y_true, yhat, digits=3, zero_division=0)

    # diagnostics
    uniq = len(np.unique(np.round(p, 6)))
    print(f"[diag:{tag}] n={len(y_true)} pos={int(y_true.sum())} base={base_rate:.3f} "
          f"unique_scores≈{uniq}")

    preds = ids_df.copy()
    preds["p_fused"] = p
    preds["yhat"] = yhat
    preds["thr_star"] = thr_star
    preds.to_csv(outdir / f"predictions_{tag}.csv", index=False)

    _pr_plot(rec, prec, auprc, base_rate, outdir / f"pr_fusion_{tag}.png")
    _roc_plot(fpr, tpr, auroc if not np.isnan(auroc) else 0.0, outdir / f"roc_fusion_{tag}.png")

    (outdir / f"metrics_{tag}.json").write_text(json.dumps({
        "feats": feats,
        "auprc": float(auprc),
        "auroc": float(auroc) if not np.isnan(auroc) else None,
        "thr_star": float(thr_star),
        "base_rate": base_rate,
        "cm": cm,
        "n": int(len(y_true)),
        "pos": int(y_true.sum()),
        "unique_scores": uniq
    }, indent=2))

    print(f"[{tag}] AUPRC={auprc:.3f} AUROC={(auroc if not np.isnan(auroc) else float('nan')):.3f} "
          f"thr*={thr_star:.3f} (baseline={base_rate:.3f})")
    print(f"[{tag}] Confusion matrix [[TN,FP],[FN,TP]]: {cm}")
    print(rep)
    return auprc, auroc

# ---------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabular_csv", required=True,
                    help="CSV with p_tab and labels (columns: id_col, scene_id, hab_label, p_tab[, datetime])")
    ap.add_argument("--det", nargs="*", default=[],
                    help="named detector CSVs: name=path.csv (CSV must contain id_col and a score column)")
    ap.add_argument("--outdir", default="runs/fusion/decision_fusion")
    ap.add_argument("--id_col", default="chip_id")
    ap.add_argument("--group_by", default="scene_id")
    ap.add_argument("--min_pos_per_split", type=int, default=2)
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--max_tries", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv_folds", type=int, default=0,
                    help="If >1, run (Stratified)GroupKFold using group_by as groups.")
    ap.add_argument("--coco_json", default="", help="COCO JSON for detector split (chip_id→file_name)")
    ap.add_argument("--require_overlap", action="store_true",
                    help="Fail if a detector has zero overlap after all strategies.")
    ap.add_argument("--target_precision", type=float, default=None,
                    help="Pick the smallest threshold achieving at least this precision.")
    ap.add_argument("--target_recall", type=float, default=None,
                    help="Pick the largest threshold achieving at least this recall.")
    ap.add_argument("--det_agg", choices=["max", "mean", "median"], default="max",
                    help="Aggregation used when collapsing detector scores per month(/region).")
    ap.add_argument("--season_flags", action="store_true",
                    help="Append two features: season_winter(Jan–Mar), season_summer(Jul–Sep).")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # COCO id↔name maps (optional)
    id2name, _ = _load_coco_map(args.coco_json)

    # --- load base/tabular ---
    base = pd.read_csv(args.tabular_csv)
    base = _clean_columns(base, args.tabular_csv)
    base = _coerce_id(base, args.id_col, args.tabular_csv)
    base[args.id_col] = _normalize_ids(base[args.id_col])

    need = {args.id_col, args.group_by, "hab_label", "p_tab"}
    missing = need - set(base.columns)
    if missing:
        raise SystemExit(f"{args.tabular_csv} missing columns: {sorted(missing)}")

    base = base[[args.id_col, args.group_by, "hab_label", "p_tab"] + ([ "datetime"] if "datetime" in base.columns else [])].copy()
    base[args.group_by] = base[args.group_by].astype(str).map(_canonical_scene_key)
    base = _attach_month_region_keys_for_base(base, args.id_col, args.group_by)

    # --- merge detector scores ---
    feats = ["p_tab"]
    for spec in args.det:
        if "=" not in spec:
            raise SystemExit("Use name=path.csv for --det (e.g., frcnn_r50=runs/fusion/p_frcnn_r50.csv)")
        name, path = spec.split("=", 1)

        df = pd.read_csv(path)
        df = _clean_columns(df, path)
        df = _coerce_id(df, args.id_col, path)
        df = _maybe_apply_coco_map(df, args.id_col, id2name, path)
        print(f"[debug] sample detector filenames for {name}: {df[args.id_col].astype(str).head(3).tolist()}")
        df[args.id_col] = _normalize_ids(df[args.id_col])

        score_col = _guess_score_col(df, args.id_col)
        if score_col != name:
            df = df.rename(columns={score_col: name})

        df = _attach_month_region_keys_for_det(df, args.id_col, name)

        # Try in order: id → scene → month+region/month (deduped & aggregated)
        merged, ok = _merge_on_id(base, df[[args.id_col, name]], args.id_col, name)
        if not ok:
            if name in base.columns: base = base.drop(columns=[name])
            det_scene_df = df[[args.id_col, name]].copy()
            det_scene_df.columns = ["__file__", name]
            merged, ok = _merge_on_scene(base, det_scene_df, args.group_by, name)
        if not ok:
            if name in base.columns: base = base.drop(columns=[name])
            merged, ok = _merge_on_month_region(base, df[[args.id_col, "month_key", "region_key", name]], name, args.det_agg)

        if not ok:
            left_only = sorted(list(set(base[args.id_col]) - set(df[args.id_col])), key=str)[:5]
            print(f"[warn] Detector '{name}' still 0% coverage after all strategies.")
            print(f"       examples from base not in detector: {left_only}")
            if args.require_overlap:
                raise SystemExit(f"[fatal] '{name}' contributed 0 matches.")

        base = merged
        if name not in base.columns:
            print(f"[warn] Detector '{name}' missing from base after merge; creating empty column.")
            base[name] = np.nan

        feats.append(name)

    # diagnostics: month overlap
    base_months = sorted([m for m in base["month_key"].dropna().unique()])
    print(f"[diag] Base months (#={len(base_months)}): {base_months[:12]} ...")
    for nm in feats[1:]:
        det_months = sorted([m for m in pd.Series(base[nm]).dropna().index.map(lambda i: base.loc[i, "month_key"]).unique() if m])
        # The above line is just to satisfy the loop; better report coverage directly:
        overlap = base.loc[~base[nm].isna(), "month_key"].dropna().unique()
        print(f"[diag] Detector '{nm}': months={base.loc[~base[nm].isna(), 'month_key'].nunique()}; overlap_with_base={len(overlap)}")

    # coverage before filling
    for nm in feats[1:]:
        na = base[nm].isna().sum()
        print(f"[info] Coverage for '{nm}': {(len(base)-na)}/{len(base)} "
              f"({100.0*(len(base)-na)/max(1,len(base)):.1f}%) non-NA")

    print(f"[info] Final feature columns: {feats}")
    print(f"[info] Base columns now: {list(base.columns)}")

    # Save merged features for debugging (pre-fill)
    base.to_csv(outdir / "merged_features_debug.csv", index=False)

    # optional season flags (simple, non-leaky)
    if args.season_flags:
        month_int = base["month_key"].str[-2:].astype(int)
        base["season_winter"] = month_int.isin([1,2,3]).astype(float)  # Jan–Mar
        base["season_summer"] = month_int.isin([7,8,9]).astype(float)  # Jul–Sep
        feats += ["season_winter", "season_summer"]

    # fill missing detector scores with 0 (neutral evidence)
    base[feats] = base[feats].fillna(0.0)

    # --- matrices ---
    X = base[feats].values
    y = base["hab_label"].astype(int).values
    groups = base[args.group_by].astype(str).values

    # deterministic but dataset-dependent seed
    dataset_tag = Path(args.tabular_csv).stem
    dataset_hash = int(hashlib.sha1(dataset_tag.encode()).hexdigest(), 16) % (10**6)
    final_seed = int(args.seed) + dataset_hash % 100000
    print(f"[debug] Using dataset-dependent seed: {final_seed} (from '{dataset_tag}')")

    def _fit_predict(train_idx, test_idx, tag):
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ])
        pipe.fit(X[train_idx], y[train_idx])
        p_te = pipe.predict_proba(X[test_idx])[:, 1]
        prec, rec, thr = precision_recall_curve(y[test_idx], p_te)
        thr_star = _pick_threshold(prec, rec, thr, args.target_precision, args.target_recall)
        ids_df = base.iloc[test_idx][[args.id_col, args.group_by, "hab_label", "month_key", "region_key"]].copy()
        auprc, auroc = _eval_and_save(outdir, tag, feats, y[test_idx], p_te, thr_star, ids_df)
        joblib.dump({"pipe": pipe, "features": feats}, outdir / f"fusion_model_{tag}.joblib")
        return auprc, auroc

    summaries = []

    if args.cv_folds and args.cv_folds > 1:
        if HAS_SGF:
            print(f"[cv] Scene-level stratification with {args.cv_folds} requested folds.")
            # scene-wise indices
            base_df = pd.DataFrame({"scene": base[args.group_by]})
            # map each scene to its row indices
            scene_to_idx = base_df.groupby("scene").apply(lambda g: g.index.values).to_dict()
            scenes = np.array(list(scene_to_idx.keys()))
            rng = np.random.RandomState(final_seed)
            rng.shuffle(scenes)
            folds = np.array_split(scenes, args.cv_folds)
            for i, te_scenes in enumerate(folds, 1):
                te_idx = np.concatenate([scene_to_idx[s] for s in te_scenes])
                tr_idx = np.setdiff1d(np.arange(len(base)), te_idx, assume_unique=False)
                print(f"[CV{i}] train={len(tr_idx)} (pos={int(y[tr_idx].sum())}) | test={len(te_idx)} (pos={int(y[te_idx].sum())})")
                auprc, auroc = _fit_predict(tr_idx, te_idx, f"cv{i}")
                summaries.append({"fold": i, "auprc": auprc, "auroc": auroc,
                                  "test_pos": int(y[te_idx].sum()), "test_total": int(len(te_idx))})
        else:
            print(f"[cv] StratifiedGroupKFold unavailable; falling back to GroupKFold(n_splits={args.cv_folds})")
            gkf = GroupKFold(n_splits=args.cv_folds)
            fold_id = 0
            for tr, te in gkf.split(X, y, groups=groups):
                fold_id += 1
                print(f"[CV{fold_id}] train={len(tr)} (pos={int(y[tr].sum())}) | test={len(te)} (pos={int(y[te].sum())})")
                auprc, auroc = _fit_predict(tr, te, f"cv{fold_id}")
                summaries.append({"fold": fold_id, "auprc": auprc, "auroc": auroc,
                                  "test_pos": int(y[te].sum()), "test_total": int(len(te))})

        df_sum = pd.DataFrame(summaries)
        df_sum.loc["mean"] = {"fold": "mean",
                              "auprc": df_sum["auprc"].mean(),
                              "auroc": df_sum["auroc"].mean(),
                              "test_pos": df_sum["test_pos"].sum(),
                              "test_total": df_sum["test_total"].sum()}
        df_sum.to_csv(outdir / "summary_cv.csv", index=False)
        print("[cv] Averages:", df_sum.loc["mean"].to_dict())

    else:
        # Single holdout split with GroupShuffleSplit
        rng = np.random.RandomState(final_seed)
        good = None
        for t in range(1, args.max_tries + 1):
            gss = GroupShuffleSplit(test_size=args.test_size, random_state=int(rng.randint(0, 10_000)))
            tr, te = next(gss.split(X, y, groups=groups))
            if y[tr].sum() >= args.min_pos_per_split and y[te].sum() >= args.min_pos_per_split:
                good = (tr, te); tries = t; break
        if good is None:
            raise SystemExit("Could not find a split with positives in both partitions; lower --min_pos_per_split.")
        tr, te = good
        print(f"[Split] train={len(tr)} (pos={int(y[tr].sum())}, neg={len(tr)-int(y[tr].sum())}) | "
              f"test={len(te)} (pos={int(y[te].sum())}, neg={len(te)-int(y[te].sum())}) | tries={tries}")
        auprc, auroc = _fit_predict(tr, te, "holdout")
        pd.DataFrame([{
            "model": "fusion(logreg)",
            "feats": "+".join(feats),
            "auprc": auprc, "auroc": auroc,
            "test_pos": int(y[te].sum()),
            "test_total": int(len(te))
        }]).to_csv(outdir / "summary.csv", index=False)

    print(f"[debug] Saving outputs to: {outdir.resolve()}")
    print(f"✓ saved models, metrics, PR/ROC plots, predictions, and merged_features_debug.csv")

if __name__ == "__main__":
    main()
