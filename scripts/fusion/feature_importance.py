#!/usr/bin/env python3
"""
feature_importance_fusion.py

Feature importance for fusion models (LogReg / CatBoost),
compatible with:
  - calibrated probabilities
  - pipelines
  - CatBoost native importances
  - permutation importance using AUPRC (HAB-safe)

Outputs:
  - feature_importances_catboost.csv (if CatBoost)
  - feature_importances_perm.csv
  - top-20 bar plots
"""

import argparse
from pathlib import Path
import joblib
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score


# ---------------------------------------------------------------------
# Minimal dummy class so old joblibs with WeightedScaler still load
# ---------------------------------------------------------------------
class WeightedScaler:
    def __init__(self, *args, **kwargs): pass
    def fit(self, X, y=None): return self
    def transform(self, X): return X
    def fit_transform(self, X, y=None): return X

class _SigmoidCalibrator:
    def __init__(self, *args, **kwargs): pass
    def fit(self, p, y=None): return self
    def transform(self, p): return p

class _IsotonicCalibrator:
    def __init__(self, *args, **kwargs): pass
    def fit(self, p, y=None): return self
    def transform(self, p): return p

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def load_joblib_any(path: Path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def unwrap_model(obj):
    """
    Extract the *actual estimator* used for prediction.
    Handles:
      - dict wrappers
      - sklearn Pipeline
      - calibrated models
    """
    if isinstance(obj, dict):
        obj = obj.get("model", obj)

    # sklearn Pipeline
    if hasattr(obj, "named_steps"):
        return obj.named_steps.get("clf", obj)

    return obj


def find_catboost(estimator):
    """
    Return CatBoostClassifier if present, else None.
    """
    try:
        from catboost import CatBoostClassifier
        if isinstance(estimator, CatBoostClassifier):
            return estimator
    except Exception:
        pass
    return None


def ap_scorer(estimator, X, y):
    p = estimator.predict_proba(X)[:, 1]
    return average_precision_score(y, p)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--model_file", default="")
    ap.add_argument("--perm_repeats", type=int, default=15)
    ap.add_argument("--perm_max_rows", type=int, default=6000)
    args = ap.parse_args()

    run_dir = Path(args.model_dir)

    # -------------------------------------------------
    # Locate model file
    # -------------------------------------------------
    if args.model_file:
        model_path = run_dir / args.model_file
    else:
        candidates = sorted(run_dir.glob("fusion_model_*.joblib"))
        if not candidates:
            raise SystemExit("❌ No fusion_model_*.joblib found")
        model_path = candidates[0]

    print(f"📂 Loading model: {model_path.name}")

    bundle = load_joblib_any(model_path)

    if not isinstance(bundle, dict):
        raise SystemExit("❌ Expected joblib dict with model + features")

    model = bundle["model"]
    feature_cols = list(bundle["features"])

    merged_path = run_dir / "merged_features_debug.csv"
    if not merged_path.exists():
        raise SystemExit("❌ merged_features_debug.csv missing")

    merged = pd.read_csv(merged_path)

    X = merged[feature_cols].astype(float).fillna(0.0).values
    y = merged["hab_label"].astype(int).values

    # Optional subsample for speed
    if args.perm_max_rows and len(y) > args.perm_max_rows:
        rng = np.random.RandomState(123)
        idx = rng.choice(len(y), size=args.perm_max_rows, replace=False)
        X, y = X[idx], y[idx]
        print(f"[perm] Subsampled to {len(y)} rows")

    # -------------------------------------------------
    # CATBOOST NATIVE IMPORTANCE (BEST)
    # -------------------------------------------------
    estimator = unwrap_model(model)
    cb = find_catboost(estimator)

    if cb is not None:
        print("🐱 Detected CatBoost model — using native importances")

        importances = cb.get_feature_importance(
            type="PredictionValuesChange"
        )

        df_cb = (
            pd.DataFrame({
                "feature": feature_cols,
                "importance": importances
            })
            .sort_values("importance", ascending=False)
        )

        df_cb.to_csv(run_dir / "feature_importances_catboost.csv", index=False)

        plt.figure(figsize=(7, 5))
        plt.barh(df_cb.head(20)["feature"][::-1],
                 df_cb.head(20)["importance"][::-1])
        plt.xlabel("CatBoost importance (PredictionValuesChange)")
        plt.tight_layout()
        plt.savefig(run_dir / "feature_importances_catboost_top20.png")

        print("✅ Saved CatBoost native importances")
        print("\nTop 10 (CatBoost):")
        for _, r in df_cb.head(10).iterrows():
            print(f"  {r.feature:35s} → {r.importance:.4f}")

    else:
        print("ℹ️ Not a CatBoost model — skipping native importance")

    # -------------------------------------------------
    # PERMUTATION IMPORTANCE (CALIBRATION-SAFE)
    # -------------------------------------------------
    print("🔁 Computing permutation importance (AUPRC scorer)")

    perm = permutation_importance(
        model,
        X,
        y,
        scoring=lambda est, Xt, yt: ap_scorer(est, Xt, yt),
        n_repeats=args.perm_repeats,
        random_state=123,
        n_jobs=-1,
    )

    df_perm = (
        pd.DataFrame({
            "feature": feature_cols,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
    )

    df_perm.to_csv(run_dir / "feature_importances_perm.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.barh(df_perm.head(20)["feature"][::-1],
             df_perm.head(20)["importance_mean"][::-1])
    plt.xlabel("Permutation importance (ΔAUPRC)")
    plt.tight_layout()
    plt.savefig(run_dir / "feature_importances_perm_top20.png")

    print("✅ Saved permutation importances")
    print("\nTop 10 (Permutation ΔAUPRC):")
    for _, r in df_perm.head(10).iterrows():
        print(f"  {r.feature:35s} → {r.importance_mean:.6f} ± {r.importance_std:.6f}")


if __name__ == "__main__":
    main()
