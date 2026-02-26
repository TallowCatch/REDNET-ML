from __future__ import annotations

from types import ModuleType

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def score_module(module_loader) -> ModuleType:
    return module_loader("scripts/HAB/score_existing_csv.py")


@pytest.mark.unit
def test_harmonize_modis_columns_variants(score_module: ModuleType) -> None:
    df = pd.DataFrame({"Kd_490": [0.12], "flh": [0.34]})
    out = score_module.harmonize_modis_columns(df)

    assert "kd490" in out.columns
    assert "nflh" in out.columns
    assert out.loc[0, "kd490"] == pytest.approx(0.12)
    assert out.loc[0, "nflh"] == pytest.approx(0.34)


@pytest.mark.unit
def test_add_time_features_extracts_month_number(score_module: ModuleType) -> None:
    df = pd.DataFrame({"datetime": ["2025-09-12T00:00:00Z", "bad"]})
    out = score_module.add_time_features(df)

    assert int(out.loc[0, "month_num"]) == 9
    assert np.isnan(out.loc[1, "month_num"])
    assert "month_sin" in out.columns
    assert "month_cos" in out.columns


@pytest.mark.unit
def test_add_engineered_features_clips_and_derived_ratios(score_module: ModuleType) -> None:
    df = pd.DataFrame(
        {
            "kd490": [0.0, 2.0],
            "chlor_a": [1.0, 4.0],
            "nflh": [0.5, 1.0],
            "sst": [28.0, 29.0],
            "fai_mean": [0.1, 0.2],
            "ndwi_mean": [0.3, 0.4],
            "rednir_mean": [0.5, 0.6],
        }
    )
    out = score_module.add_engineered_features(df)

    assert out.loc[0, "log_kd490"] == pytest.approx(np.log(1e-9))
    assert out.loc[1, "log_chlor_a"] == pytest.approx(np.log(4.0))
    assert np.isfinite(out.loc[0, "ratio_chl_kd"])
    assert out.loc[1, "ratio_nflh_kd"] == pytest.approx(0.5)
    assert "chl_times_nflh" in out.columns


@pytest.mark.unit
def test_safe_fill_uses_bundle_fill_values_and_zero_fallback(score_module: ModuleType) -> None:
    df = pd.DataFrame({"a": [np.nan], "b": [1.0]})
    feature_cols = ["a", "b", "c"]

    out = score_module.safe_fill(df, feature_cols, {"fill_values": {"a": 2.5}})

    assert set(feature_cols).issubset(out.columns)
    assert out.loc[0, "a"] == pytest.approx(2.5)
    assert out.loc[0, "b"] == pytest.approx(1.0)
    assert out.loc[0, "c"] == pytest.approx(0.0)
    assert not out[feature_cols].isna().any().any()


@pytest.mark.unit
def test_safe_fill_accepts_feature_fill_values_alias(score_module: ModuleType) -> None:
    df = pd.DataFrame({"x": [np.nan]})
    out = score_module.safe_fill(df, ["x"], {"feature_fill_values": {"x": 4.2}})
    assert out.loc[0, "x"] == pytest.approx(4.2)
