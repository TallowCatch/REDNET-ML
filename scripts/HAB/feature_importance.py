#!/usr/bin/env python3
#revisit it later
import argparse, joblib, json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Folder from a fusion run (contains merged_features_debug.csv and trained model)")
    args = ap.parse_args()
    p = Path(args.model_dir)

    # --- Try to detect trained model file ---
    model_file = None
    for cand in ["model_cv2.joblib", "model_cv2.pkl", "clf_cv2.joblib"]:
        if (p / cand).exists():
            model_file = p / cand
            break
    if model_file is None:
        raise SystemExit("❌ No model file found in " + str(p))

    print(f"📂 Loading model from {model_file}")
    clf = joblib.load(model_file)

    # --- Load feature matrix ---
    merged = pd.read_csv(p / "merged_features_debug.csv")
    feature_cols = [c for c in merged.columns if c not in ["tile","scene_id","datetime","hab_label"]]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "named_steps"):
        # For sklearn pipeline
        final = clf.named_steps.get("clf") or clf.named_steps.get("estimator")
        importances = getattr(final, "feature_importances_", None)
    else:
        importances = None

    if importances is None:
        raise SystemExit("❌ This model doesn't expose feature_importances_")

    df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    df = df.sort_values("importance", ascending=False)
    df.to_csv(p / "feature_importances.csv", index=False)

    plt.figure(figsize=(6,4))
    df.head(20).plot.bar(x="feature", y="importance", legend=False)
    plt.tight_layout()
    plt.savefig(p / "feature_importances_top20.png")
    print("✅ Saved feature_importances.csv and feature_importances_top20.png")

if __name__ == "__main__":
    main()
