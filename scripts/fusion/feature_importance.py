#!/usr/bin/env python3
import argparse
from pathlib import Path
import joblib
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- dummy WeightedScaler so unpickling works ---
class WeightedScaler:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X


def _load_model(model_file: Path):
    """Load a fusion model, handling dict-wrapped models and custom scalers."""
    try:
        clf = joblib.load(model_file)
    except Exception as e:
        print(f"[warn] joblib.load failed: {e}")
        with open(model_file, "rb") as f:
            clf = pickle.load(f)

    model = clf
    feature_cols = None

    # if we saved as {"model": clf, "features": feat_cols, "args": ...}
    if isinstance(clf, dict):
        if "model" in clf:
            model = clf["model"]
        if "features" in clf:
            feature_cols = clf["features"]

    return model, feature_cols


def _get_final_estimator(model):
    """Unwrap Pipeline to get the final estimator."""
    # Pipeline case
    if hasattr(model, "named_steps"):
        final = model.named_steps.get("clf") or model.named_steps.get("estimator")
        return final if final is not None else model
    return model


def _get_importances(estimator):
    """
    Get feature importances from the final estimator.

    Priority:
      1) feature_importances_ (trees/GBDT)
      2) coef_ (linear models: we use |coef| as importance)
    """
    # tree-based / GBDT
    if hasattr(estimator, "feature_importances_"):
        imp = estimator.feature_importances_
        return np.asarray(imp, dtype=float)

    # linear models (LogisticRegression, etc.)
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        if coef.ndim == 1:
            imp = np.abs(coef)
        else:
            # average magnitude across classes
            imp = np.mean(np.abs(coef), axis=0)
        return imp

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Folder from a fusion run")
    args = ap.parse_args()
    p = Path(args.model_dir)

    # --- detect model file ---
    candidates = [
        "fusion_model_cv2.joblib",
        "fusion_model_cv3.joblib",
        "model_cv2.joblib",
        "model_cv3.joblib",
        "clf_cv2.joblib",
        "clf_cv3.joblib",
    ]
    model_file = next((p / c for c in candidates if (p / c).exists()), None)
    if not model_file:
        raise SystemExit(f"❌ No model file found in {p}")
    print(f"📂 Loading model from {model_file.name}")

    # --- load model & (optionally) feature list from joblib ---
    model, feature_cols = _load_model(model_file)

    # --- if no feature list in joblib, fall back to merged_features_debug.csv ---
    if feature_cols is None:
        merged_path = p / "merged_features_debug.csv"
        if not merged_path.exists():
            raise SystemExit(f"❌ Missing merged_features_debug.csv in {p}")
        merged = pd.read_csv(merged_path)

        non_feature_cols = ["tile", "scene_id", "datetime", "hab_label",
                            "month_key", "region_key", "group_for_cv", "_month_ord"]
        feature_cols = [c for c in merged.columns if c not in non_feature_cols]

    feature_cols = list(feature_cols)

    # --- unwrap to final estimator and get "importances" ---
    final = _get_final_estimator(model)
    importances = _get_importances(final)

    if importances is None:
        raise SystemExit("❌ This model doesn't expose feature_importances_ or coef_")

    if len(importances) != len(feature_cols):
        raise SystemExit(
            f"❌ Mismatch: model has {len(importances)} weights, "
            f"but we found {len(feature_cols)} feature columns. "
            "Check feature_cols logic."
        )

    df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
    )
    df.to_csv(p / "feature_importances.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.barh(df.head(20)["feature"][::-1], df.head(20)["importance"][::-1])
    plt.xlabel("Importance (|coef| or feature_importances_)")
    plt.tight_layout()
    plt.savefig(p / "feature_importances_top20.png")

    print("✅ Saved feature_importances.csv and feature_importances_top20.png")
    print("\nTop 10 features:")
    for _, row in df.head(10).iterrows():
        print(f"  {row.feature:30s} → {row.importance:.4f}")


if __name__ == "__main__":
    main()
