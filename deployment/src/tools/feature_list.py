import json
from pathlib import Path
import joblib
from datetime import datetime, timezone

run_dir = Path("runs/fusion/frozen/catboost_timecv4_minpos15")

bundle = joblib.load(run_dir / "fusion_model_cv2.joblib")
features = list(bundle["features"])

# naive dtype guess: mark month_num int, everything else float
dtypes = {f: ("int" if f in ["month_num"] else "float") for f in features}

features_json = {
  "schema_version": "1.0",
  "label_col": "hab_label",
  "month_col": "month_key",
  "features_ordered": features,
  "dtypes": dtypes,
  "fillna": {"default_float": 0.0, "default_int": 0}
}

meta_json = {
  "artifact_name": "fusion_catboost_timecv4_minpos15",
  "created_utc": datetime.now(timezone.utc).isoformat(),
  "source_run_dir": str(run_dir),
  "models": [
    {"name": "fusion_model_cv2.joblib", "role": "ensemble_member"},
    {"name": "fusion_model_cv3.joblib", "role": "ensemble_member"},
    {"name": "fusion_model_cv4.joblib", "role": "ensemble_member"},
  ],
  "inference_mode": "ensemble_mean",
  "notes": "Frozen deployment artifact"
}

out = Path("deployment/artifacts")
out.mkdir(parents=True, exist_ok=True)

(out / "features.json").write_text(json.dumps(features_json, indent=2))
(out / "metadata.json").write_text(json.dumps(meta_json, indent=2))

print("wrote:", out / "features.json")
print("wrote:", out / "metadata.json")
