import joblib
b = joblib.load("runs/fusion/fusion_alllabels_cv5/fusion_model_cv5.joblib")
print("has year in features?", "year" in b["features"])
print("n_features:", len(b["features"]))