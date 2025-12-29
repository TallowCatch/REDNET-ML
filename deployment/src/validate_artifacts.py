# deployment/src/validate_artifacts.py
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from build_features import FeatureConfig, build_features


def load_json(p: Path) -> dict:
    return json.loads(p.read_text())


def main():
    base = Path(__file__).resolve().parents[1]
    artifacts = base / "artifacts"

    features_cfg = load_json(artifacts / "features.json")
    meta = load_json(artifacts / "metadata.json")

    # load one bundle to validate schema
    model_dir = artifacts / "model"
    b0 = joblib.load(model_dir / meta["models"][0]["name"])

    bundle_features = list(b0["features"])
    contract_features = list(features_cfg["features_ordered"])

    if bundle_features != contract_features:
        raise SystemExit(
            "❌ Feature order mismatch!\n"
            f"bundle has {len(bundle_features)} features, contract has {len(contract_features)}.\n"
            "Fix deployment/artifacts/features.json to match bundle['features'] exactly."
        )

    print(f"✅ Feature schema matches bundle (n={len(contract_features)})")

    # minimal smoke input
    sample = {
        "month_key": "2023-09",
        "nflh": 0.2,
        "chlor_a": 0.8,
        "kd490": 0.09,
        "sst": 28.0,
        "sst_anom": 0.5,
        "sst_anom_z": 0.0,
        "sst_clim_rm": 0.0,
        "fai_mean": 0.1,
        "ndwi_mean": 0.2,
        "ndwi_std": 0.05,
        "rednir_mean": 0.03,
        "rednir_std": 0.0,
        "p_frcnn_r50_med": 0.9,
        "p_frcnn_mb_med": 0.25,
        "p_ssd_mb_med": 0.55,
        "p_tab": 0.2
    }

    cfg = FeatureConfig(
        features_ordered=contract_features,
        dtypes=features_cfg["dtypes"],
        default_float=features_cfg["fillna"]["default_float"],
        default_int=features_cfg["fillna"]["default_int"],
    )

    Xdf = build_features(pd.DataFrame([sample]), cfg, month_key_col=features_cfg.get("month_col", "month_key"))
    X = Xdf.values

    # predict
    model = b0["model"]
    p = model.predict_proba(X)[:, 1]
    print(f"✅ Smoke prediction ok. prob={float(p[0]):.6f}")


if __name__ == "__main__":
    main()
