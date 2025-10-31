#!/usr/bin/env python3
"""
feature_importances.py
----------------------
Extracts and visualizes feature importances (or coefficient magnitudes)
from a trained fusion model directory (e.g., fusion_enriched_norm_f1_v4).

Supports:
  • sklearn LogisticRegression (coefficients)
  • sklearn models with .feature_importances_
  • Pipeline / CalibratedClassifierCV wrappers
"""

import argparse, joblib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def flatten_model(model):
    """Return the innermost estimator (handles Pipeline and CalibratedClassifierCV)."""
    if hasattr(model, "best_estimator_"):
        model = model.best_estimator_
    if hasattr(model, "base_estimator"):
        model = model.base_estimator
    if hasattr(model, "estimator"):
        model = model.estimator
    if hasattr(model, "named_steps"):
        # For pipeline objects
        for step_name, step_obj in model.named_steps.items():
            if hasattr(step_obj, "feature_importances_") or hasattr(step_obj, "coef_"):
                return step_obj
    return model


def load_model_from_dir(model_dir: Path):
    """Try to load a model or dict from a fusion run directory."""
    for cand in model_dir.glob("*.joblib"):
        try:
            obj = joblib.load(cand)
            print(f"📂 Loaded {cand.name}")
            if isinstance(obj, dict) and "model" in obj:
                return obj["model"], obj.get("features", [])
            else:
                return obj, []
        except Exception as e:
            print(f"[warn] Failed to load {cand}: {e}")
    raise SystemExit("❌ No loadable model found in the directory.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True,
                    help="Path to fusion run directory (e.g., fusion_enriched_norm_f1_v4)")
    ap.add_argument("--top", type=int, default=20, help="How many top features to show in plot")
    args = ap.parse_args()
    p = Path(args.model_dir)

    model, feat_names = load_model_from_dir(p)
    model_inner = flatten_model(model)

    if not feat_names and (p / "merged_features_debug.csv").exists():
        df = pd.read_csv(p / "merged_features_debug.csv")
        excl = ["tile", "scene_id", "datetime", "hab_label", "month_key", "region_key", "date_key"]
        feat_names = [c for c in df.columns if c not in excl]

    print(f"🧩 Found {len(feat_names)} feature names.")

    # Determine importances or coefficients
    if hasattr(model_inner, "feature_importances_"):
        importances = np.asarray(model_inner.feature_importances_, dtype=float)
    elif hasattr(model_inner, "coef_"):
        importances = np.abs(model_inner.coef_).ravel()
    else:
        raise SystemExit("❌ Model has no feature_importances_ or coef_ attribute.")

    # Align lengths
    if len(importances) != len(feat_names):
        print(f"[warn] Length mismatch: {len(importances)} vs {len(feat_names)}")
        feat_names = feat_names[: len(importances)]

    df_imp = pd.DataFrame({"feature": feat_names, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False)
    df_imp.to_csv(p / "feature_importances.csv", index=False)

    # --- Plot ---
    plt.figure(figsize=(7, 6))
    top_df = df_imp.head(args.top)
    plt.barh(top_df["feature"], top_df["importance"], color="skyblue")
    plt.gca().invert_yaxis()
    plt.xlabel("Importance (|coefficient|)")
    plt.title(f"Top {args.top} Feature Importances")
    plt.tight_layout()
    plt.savefig(p / f"feature_importances_top{args.top}.png", dpi=200)
    plt.close()

    print("✅ Saved:")
    print(f"   • feature_importances.csv")
    print(f"   • feature_importances_top{args.top}.png\n")

    # --- Quick summary printout ---
    print("🔝 Top 10 features:")
    for _, row in top_df.head(10).iterrows():
        print(f"   {row['feature']:25s}  →  {row['importance']:.4f}")

if __name__ == "__main__":
    main()
