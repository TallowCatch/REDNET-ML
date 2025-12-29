# deployment/src/inference.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd

from build_features import FeatureConfig, build_features


def load_json(p: Path) -> dict:
    return json.loads(p.read_text())


def load_bundles(model_dir: Path, model_names: List[str]) -> List[dict]:
    bundles = []
    for name in model_names:
        bp = model_dir / name
        bundles.append(joblib.load(bp))
    return bundles


def predict_proba_from_bundle(bundle: dict, X: np.ndarray) -> np.ndarray:
    """
    Bundle created by your training script usually has:
      - bundle["model"] : CatBoostClassifier
      - bundle["calibrator"] optional
    We will support either.
    """
    model = bundle["model"]
    p = model.predict_proba(X)[:, 1]

    cal = bundle.get("calibrator", None)
    if cal is not None:
        # your calibrator has .transform(p)
        p = np.asarray(cal.transform(p), dtype=float)

    return np.clip(p, 1e-6, 1 - 1e-6)


def ensemble_mean_predict(bundles: List[dict], X: np.ndarray) -> np.ndarray:
    ps = [predict_proba_from_bundle(b, X) for b in bundles]
    return np.mean(np.vstack(ps), axis=0)


def run_inference(
    raw_input: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    artifacts_dir: Path,
) -> pd.DataFrame:
    features_cfg = load_json(artifacts_dir / "features.json")
    meta = load_json(artifacts_dir / "metadata.json")
    thr = load_json(artifacts_dir / "thresholds.json")

    cfg = FeatureConfig(
        features_ordered=features_cfg["features_ordered"],
        dtypes=features_cfg["dtypes"],
        default_float=features_cfg["fillna"]["default_float"],
        default_int=features_cfg["fillna"]["default_int"],
    )

    # normalize input to DF
    if isinstance(raw_input, dict):
        raw_df = pd.DataFrame([raw_input])
    elif isinstance(raw_input, list):
        raw_df = pd.DataFrame(raw_input)
    else:
        raw_df = raw_input.copy()

    X_df = build_features(raw_df, cfg, month_key_col=features_cfg.get("month_col", "month_key"))
    X = X_df.values

    model_dir = artifacts_dir / "model"
    model_names = [m["name"] for m in meta["models"]]

    bundles = load_bundles(model_dir, model_names)

    mode = meta.get("inference_mode", "ensemble_mean")
    if mode == "ensemble_mean":
        p = ensemble_mean_predict(bundles, X)
    else:
        # fallback: use first bundle only
        p = predict_proba_from_bundle(bundles[0], X)

    # threshold
    t = float(thr.get("default_threshold", 0.5))
    yhat = (p >= t).astype(int)

    out = raw_df.copy()
    out["prob"] = p
    out["pred"] = yhat
    out["threshold_used"] = t
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", required=True)
    ap.add_argument("--input_json", required=True, help="Path to JSON dict or list[dict]")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    raw = json.loads(Path(args.input_json).read_text())

    out = run_inference(raw, artifacts_dir)
    print(out.to_string(index=False))
