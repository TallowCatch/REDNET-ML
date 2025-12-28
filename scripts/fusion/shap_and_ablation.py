#!/usr/bin/env python3
"""
shap_and_ablation.py

1) SHAP directionality (CatBoost SHAP values) for key features:
   - sst_anom, chlor_a, nflh (+ anything else)
   Outputs:
     - shap_mean_abs.csv (global importance)
     - shap_bar_top20.png
     - shap_dependence_<feature>.png (directionality)

2) Ablation sanity check:
   Retrain CatBoost with and without a feature (default: month_sin),
   using chronological CV folds (same logic style as your fit script):
     - split months into K blocks
     - train on months < earliest_test_month
     - test on block months
   Compares:
     - AUPRC, AUROC (mean across valid folds)
     - permutation importance deltas for key features (optional)

IMPORTANT:
- This script is designed to load your joblib bundle created by fit_decision_fusion.py.
- Your bundle includes a custom calibrator that may have been pickled as __main__._SigmoidCalibrator.
  To make unpickling work, we define _SigmoidCalibrator/_IsotonicCalibrator here too.
"""

import argparse
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from catboost import Pool

from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


# ---------------------------
# Calibrator classes (to allow joblib unpickle)
# ---------------------------
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

def _apply_calibrator(cal, p):
    p = np.asarray(p, dtype=float)
    if cal is None:
        return np.clip(p, 1e-6, 1 - 1e-6)
    return np.clip(cal.transform(p), 1e-6, 1 - 1e-6)


# ---------------------------
# Helpers
# ---------------------------
def _safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, p)

def _month_to_ordinal(s):
    if not isinstance(s, str) or "-" not in s:
        return -10**9
    try:
        y, m = s.split("-")
        return int(y) * 12 + int(m)
    except Exception:
        return -10**9

def _load_bundle(run_dir: Path, model_file: str = ""):
    if model_file:
        mp = run_dir / model_file
        if not mp.exists():
            raise SystemExit(f"❌ model_file not found: {mp}")
        bundle = joblib.load(mp)
        return mp, bundle

    candidates = sorted(run_dir.glob("fusion_model_*.joblib"))
    if not candidates:
        raise SystemExit(f"❌ No fusion_model_*.joblib found in {run_dir}")
    mp = candidates[0]
    bundle = joblib.load(mp)
    return mp, bundle

def _require_catboost_model(model):
    from catboost import CatBoostClassifier
    if not isinstance(model, CatBoostClassifier):
        raise SystemExit("❌ This script is for CatBoost models only. Your bundle model is not CatBoost.")
    return model

def _shap_bar(mean_abs, outpng, topk=20):
    df = mean_abs.sort_values("mean_abs_shap", ascending=False).head(topk).iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(df["feature"], df["mean_abs_shap"])
    plt.xlabel("Mean(|SHAP|) on raw margin")
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()

def _dependence_plot(x, shapv, feature, outpng):
    # SHAP sign: + increases model raw score (log-odds), - decreases it
    plt.figure(figsize=(6.2, 4.8))
    plt.scatter(x, shapv, s=10, alpha=0.6)
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel(feature)
    plt.ylabel(f"SHAP({feature}) on raw margin")
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()

def _train_catboost(X_tr, y_tr, X_va, y_va, params, seed):
    from catboost import CatBoostClassifier
    cb = CatBoostClassifier(
        iterations=int(params.get("cb_iters", 800)),
        depth=int(params.get("cb_depth", 6)),
        learning_rate=float(params.get("cb_lr", 0.05)),
        l2_leaf_reg=float(params.get("cb_l2", 6.0)),
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=int(seed),
        verbose=False,
        auto_class_weights="Balanced",
        od_type="Iter",
        od_wait=int(params.get("cb_early_stop", 80)),
    )
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    return cb


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="runs/fusion/<your_run>")
    ap.add_argument("--model_file", default="", help="Optional: fusion_model_cv2.joblib etc")
    ap.add_argument("--outdir", default="", help="Optional: default = <run_dir>/shap_ablation")
    ap.add_argument("--key_features", default="sst_anom,chlor_a,nflh", help="CSV list for dependence plots")

    # SHAP controls
    ap.add_argument("--shap_max_rows", type=int, default=5000, help="Cap rows for SHAP speed")
    ap.add_argument("--shap_seed", type=int, default=123)
    ap.add_argument("--shap_split", choices=["train", "test", "raw"], default="test",
                help="Which merged_features_debug CSV to explain.")
    ap.add_argument("--shap_fold", default="cv4",
                    help="Which fold prefix to use, e.g. cv2, cv3, cv4. Use 'all' to combine all folds.")


    # Ablation controls
    ap.add_argument("--do_ablation", action="store_true")
    ap.add_argument("--ablate_feature", default="month_sin")
    ap.add_argument("--cv_time_folds", type=int, default=3)
    ap.add_argument("--calibrate", choices=["none", "sigmoid", "isotonic"], default="sigmoid")
    ap.add_argument("--calib_frac", type=float, default=0.20)
    ap.add_argument("--perm_importance", action="store_true", help="Also compute permutation importance on test folds")
    ap.add_argument("--perm_repeats", type=int, default=10)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    model_path, bundle = _load_bundle(run_dir, args.model_file)

    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        raise SystemExit("❌ Expected bundle dict with keys: model, features")

    model = bundle["model"]
    features = list(bundle["features"])
    saved_args = bundle.get("args", {})

    outdir = Path(args.outdir) if args.outdir else (run_dir / "shap_ablation")
    outdir.mkdir(parents=True, exist_ok=True)

    def pick_shap_csv(run_dir: Path, split: str, fold: str) -> Path:
        if split == "raw":
            p = run_dir / "merged_features_debug_raw.csv"
            return p

        # train/test case
        if fold == "all":
            # we'll merge all matching fold CSVs into one temporary dataframe later
            return Path("__ALL__")

        p = run_dir / f"merged_features_debug_{fold}_{split}.csv"
        return p

    merged_path = pick_shap_csv(run_dir, args.shap_split, args.shap_fold)

    if str(merged_path) == "__ALL__":
            pattern = f"merged_features_debug_*_{args.shap_split}.csv"
            files = sorted(run_dir.glob(pattern))
            if not files:
                raise SystemExit(f"❌ No files matching {pattern} in {run_dir}")
            print(f"[shap] Combining {len(files)} files for split={args.shap_split}:")
            for f in files:
                print("  -", f.name)
            df = pd.concat([pd.read_csv(f) for f in files], axis=0, ignore_index=True)
    else:
        if not merged_path.exists():
                raise SystemExit(f"❌ Missing SHAP CSV: {merged_path}")
        df = pd.read_csv(merged_path)


    if "hab_label" not in df.columns:
        raise SystemExit("❌ merged_features_debug.csv missing hab_label")

    # ---------------------------
    # 1) SHAP (CatBoost native SHAP values)
    # ---------------------------
    print(f"📦 Loaded bundle: {model_path.name}")
    print(f"🧠 Features: {len(features)}")

    # Ensure CatBoost is usable + model is CatBoost
    cb_model = _require_catboost_model(model)

    X = df[features].astype(float).fillna(0.0).values
    y = df["hab_label"].astype(int).values

    # optional subsample for SHAP speed
    if args.shap_max_rows and len(df) > args.shap_max_rows:
        rng = np.random.RandomState(args.shap_seed)
        idx = rng.choice(len(df), size=args.shap_max_rows, replace=False)
        X_shap = X[idx]
        y_shap = y[idx]
        df_shap = df.iloc[idx].copy()
        print(f"[shap] Subsampled to {len(idx)} rows")
    else:
        X_shap = X
        y_shap = y
        df_shap = df.copy()

    print("🧩 Computing CatBoost SHAP values (raw margin contributions)...")
    # CatBoost returns (n, num_features+1) with last col = expected value
    pool = Pool(X_shap, feature_names=features)
    shap_full = cb_model.get_feature_importance(type="ShapValues", data=pool)
    shap_vals = shap_full[:, :-1]  # per-feature contributions to raw margin
    base_val = shap_full[0, -1] if shap_full.shape[0] else 0.0

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    df_mean = pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    df_mean.to_csv(outdir / "shap_mean_abs.csv", index=False)

    _shap_bar(df_mean, outdir / "shap_bar_top20.png", topk=20)

    # dependence plots for key features
    key_feats = [s.strip() for s in args.key_features.split(",") if s.strip()]
    for f in key_feats:
        if f not in features:
            print(f"[shap] skip dependence for '{f}' (not in features)")
            continue
        j = features.index(f)
        _dependence_plot(
            df_shap[f].astype(float).values,
            shap_vals[:, j],
            f,
            outdir / f"shap_dependence_{f}.png",
        )

    print(f"✅ SHAP outputs saved to: {outdir}")

    # ---------------------------
    # 2) Ablation sanity check (remove month_sin)
    # ---------------------------
    if not args.do_ablation:
        print("ℹ️ Ablation disabled. Re-run with --do_ablation to compare removing month_sin.")
        return

    print("\n🧪 Running ablation sanity check...")
    if "month_key" not in df.columns:
        raise SystemExit("❌ merged_features_debug.csv missing month_key (needed for time CV).")

    # Build chronological folds like your training script (month blocks)
    df["_month_ord"] = df["month_key"].apply(_month_to_ordinal)
    uniq_months = np.array(sorted([m for m in df["_month_ord"].unique() if m >= 0]))
    if len(uniq_months) < args.cv_time_folds:
        raise SystemExit(f"❌ Not enough unique months ({len(uniq_months)}) for cv_time_folds={args.cv_time_folds}")

    month_blocks = np.array_split(uniq_months, args.cv_time_folds)

    # group label for group-aware calibration split
    if "region_key" not in df.columns:
        print("[warn] region_key missing; calibration split will be random (less safe).")
        groups = np.array(["_nogroup_"] * len(df))
    else:
        groups = df["region_key"].astype(str).values

    # CatBoost params from saved args (fallbacks)
    cb_params = {
        "cb_iters": saved_args.get("cb_iters", 800),
        "cb_depth": saved_args.get("cb_depth", 6),
        "cb_lr": saved_args.get("cb_lr", 0.05),
        "cb_l2": saved_args.get("cb_l2", 6.0),
        "cb_early_stop": saved_args.get("cb_early_stop", 80),
    }
    seed = int(saved_args.get("seed", 42)) + 17

    def fit_calibrator(method, p_cal, y_cal):
        if method == "none":
            return None
        if method == "sigmoid":
            return _SigmoidCalibrator().fit(p_cal, y_cal)
        if method == "isotonic":
            return _IsotonicCalibrator().fit(p_cal, y_cal)
        raise ValueError(method)

    def group_cal_split(tr_idx, frac, rng_seed):
        # group-aware split, but robust to tiny group counts
        frac = float(np.clip(frac, 0.05, 0.45))
        g = groups[tr_idx]
        uniq_g = np.unique(g)
        rng = np.random.RandomState(rng_seed)

        # if not enough groups to split, fall back to random row split
        if len(uniq_g) < 2:
            perm = rng.permutation(tr_idx)
            n_cal = max(1, int(round(frac * len(tr_idx))))
            cal = perm[:n_cal]
            fit = perm[n_cal:]
            return fit, cal

        rng.shuffle(uniq_g)
        n_cal_g = max(1, int(round(frac * len(uniq_g))))
        cal_g = set(uniq_g[:n_cal_g])
        cal_mask = np.array([gg in cal_g for gg in g], dtype=bool)
        cal = tr_idx[cal_mask]
        fit = tr_idx[~cal_mask]

        # if fit is empty, relax by moving one group back
        if len(fit) == 0:
            cal_g = set(uniq_g[: max(1, n_cal_g - 1)])
            cal_mask = np.array([gg in cal_g for gg in g], dtype=bool)
            cal = tr_idx[cal_mask]
            fit = tr_idx[~cal_mask]
        return fit, cal

    def eval_fold(cb, cal, feat_list, te_idx, cal_method):
        X_te = df.loc[te_idx, feat_list].astype(float).fillna(0.0).values
        y_te = df.loc[te_idx, "hab_label"].astype(int).values

        # calibration on CAL set (within-train)
        X_cal = df.loc[cal, feat_list].astype(float).fillna(0.0).values
        y_cal = df.loc[cal, "hab_label"].astype(int).values
        p_cal_raw = cb.predict_proba(X_cal)[:, 1]
        # conservative calibration guard (avoid unstable calibration)
        min_n, min_pos, min_neg = 40, 8, 8
        if (cal_method != "none"
            and len(y_cal) >= min_n
            and (y_cal == 1).sum() >= min_pos
            and (y_cal == 0).sum() >= min_neg):
            calibrator = fit_calibrator(cal_method, p_cal_raw, y_cal)
        else:
            calibrator = None


        p_te_raw = cb.predict_proba(X_te)[:, 1]
        p_te = _apply_calibrator(calibrator, p_te_raw)

        auprc = average_precision_score(y_te, p_te) if y_te.sum() > 0 else 0.0
        auroc = _safe_auc(y_te, p_te)
        return auprc, auroc, p_te, y_te

    def run_setting(feat_list, setting_name):
        rows = []
        perm_rows = []

        for fold_id, block in enumerate(month_blocks, 1):
            te_mask = df["_month_ord"].isin(block)
            earliest_test = int(np.min(block))
            tr_mask = df["_month_ord"] < earliest_test

            tr_idx = df.index[tr_mask].to_numpy()
            te_idx = df.index[te_mask].to_numpy()

            if len(tr_idx) == 0:
                print(f"[ablation] skip fold {fold_id}: no earlier months to train")
                continue
            if df.loc[te_idx, "hab_label"].sum() < 2 or df.loc[tr_idx, "hab_label"].sum() < 2:
                print(f"[ablation] skip fold {fold_id}: insufficient positives")
                continue

            fit_idx, cal_idx = group_cal_split(tr_idx, args.calib_frac, rng_seed=seed + fold_id)

            X_fit = df.loc[fit_idx, feat_list].astype(float).fillna(0.0).values
            y_fit = df.loc[fit_idx, "hab_label"].astype(int).values
            X_cal = df.loc[cal_idx, feat_list].astype(float).fillna(0.0).values
            y_cal = df.loc[cal_idx, "hab_label"].astype(int).values

            cb = _train_catboost(X_fit, y_fit, X_cal, y_cal, cb_params, seed=seed + 100 * fold_id)

            auprc, auroc, p_te, y_te = eval_fold(cb, cal_idx, feat_list, te_idx, args.calibrate)

            rows.append({"fold": fold_id, "auprc": auprc, "auroc": auroc, "test_n": int(len(te_idx)), "test_pos": int(y_te.sum())})

            if args.perm_importance:
                # Permutation importance on the TEST fold (AUPRC scorer)
                X_test = df.loc[te_idx, feat_list].astype(float).fillna(0.0).values
                y_test = y_te

                def scorer(est, Xt, yt):
                    pp = est.predict_proba(Xt)[:, 1]
                    pp = _apply_calibrator(None, pp)  # raw probs; calibration handled in eval; permutation is relative anyway
                    return average_precision_score(yt, pp) if yt.sum() > 0 else 0.0

                perm = permutation_importance(
                    cb, X_test, y_test,
                    scoring=scorer,
                    n_repeats=args.perm_repeats,
                    random_state=123 + fold_id,
                    n_jobs=-1,
                )
                perm_rows.append(pd.DataFrame({
                    "feature": feat_list,
                    "importance_mean": perm.importances_mean,
                    "importance_std": perm.importances_std,
                    "fold": fold_id,
                    "setting": setting_name
                }))

            print(f"[{setting_name} fold{fold_id}] AUPRC={auprc:.3f} AUROC={(auroc if not np.isnan(auroc) else float('nan')):.3f}")

        df_sum = pd.DataFrame(rows)
        if len(df_sum) == 0:
            raise SystemExit(f"❌ No valid folds for setting: {setting_name}")

        df_sum.loc["mean"] = {
            "fold": "mean",
            "auprc": df_sum["auprc"].mean(),
            "auroc": df_sum["auroc"].mean(),
            "test_n": df_sum["test_n"].sum(),
            "test_pos": df_sum["test_pos"].sum(),
        }
        return df_sum, (pd.concat(perm_rows, axis=0) if perm_rows else None)

    # features for full vs ablated
    ablate = args.ablate_feature
    if ablate not in features:
        raise SystemExit(f"❌ ablate_feature '{ablate}' not found in bundle features.")

    feats_full = features
    feats_drop = [f for f in features if f != ablate]

    df_full, perm_full = run_setting(feats_full, "FULL")
    df_drop, perm_drop = run_setting(feats_drop, f"DROP_{ablate}")

    # save summaries
    df_full.to_csv(outdir / "ablation_summary_full.csv", index=False)
    df_drop.to_csv(outdir / f"ablation_summary_drop_{ablate}.csv", index=False)

    # compare mean row
    mean_full = df_full[df_full["fold"].astype(str) == "mean"].iloc[0]
    mean_drop = df_drop[df_drop["fold"].astype(str) == "mean"].iloc[0]

    comp = pd.DataFrame([{
        "setting": "FULL",
        "mean_auprc": float(mean_full["auprc"]),
        "mean_auroc": float(mean_full["auroc"]) if str(mean_full["auroc"]) != "nan" else None,
    },{
        "setting": f"DROP_{ablate}",
        "mean_auprc": float(mean_drop["auprc"]),
        "mean_auroc": float(mean_drop["auroc"]) if str(mean_drop["auroc"]) != "nan" else None,
    }])
    comp["delta_auprc"] = comp["mean_auprc"].diff()
    comp["delta_auroc"] = comp["mean_auroc"].diff()
    comp.to_csv(outdir / f"ablation_compare_drop_{ablate}.csv", index=False)

    print("\n📌 Ablation comparison (means):")
    print(comp.to_string(index=False))

    # optional: compare permutation importances averaged across folds (if enabled)
    if perm_full is not None and perm_drop is not None:
        pf = perm_full.groupby(["feature","setting"], as_index=False)["importance_mean"].mean()
        pd_ = perm_drop.groupby(["feature","setting"], as_index=False)["importance_mean"].mean()
        both = pd.concat([pf, pd_], axis=0)
        both.to_csv(outdir / f"ablation_perm_mean_drop_{ablate}.csv", index=False)

        # show key features deltas
        key = [s.strip() for s in args.key_features.split(",") if s.strip()]
        pivot = both.pivot_table(index="feature", columns="setting", values="importance_mean", aggfunc="mean")
        for k in key:
            if k in pivot.index:
                full_v = float(pivot.loc[k].get("FULL", np.nan))
                drop_v = float(pivot.loc[k].get(f"DROP_{ablate}", np.nan))
                print(f"[perm-mean] {k}: FULL={full_v:.6f}  DROP_{ablate}={drop_v:.6f}  Δ={drop_v-full_v:+.6f}")

    print(f"\n✅ Ablation outputs saved to: {outdir}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
