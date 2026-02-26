from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def ops_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "viewer" / "build_ops_payload.py"
    spec = importlib.util.spec_from_file_location("build_ops_payload", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_safe_float_and_classify_edges(ops_module: ModuleType) -> None:
    assert ops_module.safe_float("3.25") == pytest.approx(3.25)
    assert ops_module.safe_float(float("inf")) is None
    assert ops_module.safe_float("bad-value") is None

    assert ops_module.classify(None) == "unknown"
    assert ops_module.classify(ops_module.WATCH_THRESHOLD - 1e-9) == "normal"
    assert ops_module.classify(ops_module.WATCH_THRESHOLD) == "watch"
    assert ops_module.classify(ops_module.ACTION_THRESHOLD) == "action"


@pytest.mark.unit
def test_robust_scale_constant_series_returns_half(ops_module: ModuleType) -> None:
    out = ops_module.robust_scale(pd.Series([5.0, 5.0, np.nan]))

    assert out.iloc[0] == pytest.approx(0.5)
    assert out.iloc[1] == pytest.approx(0.5)
    assert np.isnan(out.iloc[2])


@pytest.mark.unit
def test_weighted_average_handles_partial_coverage(ops_module: ModuleType) -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 2.0],
            "b": [2.0, 2.0, np.nan],
        }
    )
    values, coverage = ops_module.weighted_average(df, {"a": 2.0, "b": 1.0, "missing": 5.0})

    assert values.iloc[0] == pytest.approx((2.0 * 1.0 + 1.0 * 2.0) / 3.0)
    assert values.iloc[1] == pytest.approx(2.0)
    assert values.iloc[2] == pytest.approx(2.0)
    assert coverage.iloc[0] == pytest.approx(3.0 / 8.0)
    assert coverage.iloc[1] == pytest.approx(1.0 / 8.0)
    assert coverage.iloc[2] == pytest.approx(2.0 / 8.0)


@pytest.mark.unit
def test_compute_ops_signals_falls_back_to_hab_prob_without_context(ops_module: ModuleType) -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"], utc=True),
            "hab_prob": [0.8, 0.3],
            "p_frcnn_r50_med": [np.nan, np.nan],
            "p_frcnn_mb_med": [np.nan, np.nan],
            "p_ssd_mb_med": [np.nan, np.nan],
            "sst": [np.nan, np.nan],
            "chlor_a": [np.nan, np.nan],
            "kd490": [np.nan, np.nan],
            "nflh": [np.nan, np.nan],
        }
    )

    out = ops_module.compute_ops_signals(df)

    assert out["ops_risk"].iloc[0] == pytest.approx(0.8)
    assert out["ops_risk"].iloc[1] == pytest.approx(0.3)
    assert out["status"].iloc[0] == "action"
    assert out["status"].iloc[1] in {"normal", "watch"}


@pytest.mark.unit
def test_compute_ops_signals_treats_zero_ocean_color_as_missing(ops_module: ModuleType) -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-08-01T00:00:00Z", "2025-08-09T00:00:00Z"], utc=True),
            "hab_prob": [0.7, 0.7],
            "p_frcnn_r50_med": [0.7, 0.7],
            "p_frcnn_mb_med": [0.7, 0.7],
            "p_ssd_mb_med": [0.7, 0.7],
            "sst": [27.0, 29.0],
            "chlor_a": [0.0, 0.0],
            "kd490": [0.0, 0.0],
            "nflh": [0.0, 0.0],
        }
    )

    out = ops_module.compute_ops_signals(df)

    assert out["chlor_a_norm"].isna().all()
    assert out["kd490_norm"].isna().all()
    assert out["nflh_norm"].isna().all()
    assert out["oci_coverage"].iloc[0] == pytest.approx(0.1, abs=1e-6)
    assert out["oci_coverage"].iloc[1] == pytest.approx(0.1, abs=1e-6)
    assert out["ops_risk"].between(0.0, 1.0).all()


@pytest.mark.unit
def test_derive_ops_thresholds_quantile_calibration(ops_module: ModuleType) -> None:
    hab = np.linspace(0.0, 1.0, 101)
    ops = np.power(hab, 1.2)
    fit = ops_module.derive_ops_thresholds(
        [pd.DataFrame({"hab_prob": hab, "ops_risk": ops})],
        base_watch=0.55,
        base_action=0.62,
    )

    assert fit["method"] == "quantile_match_to_legacy_alert_load"
    assert 0.0 < float(fit["watch"]) < float(fit["action"]) < 1.0
    assert float(fit["action"]) - float(fit["watch"]) >= ops_module.MIN_THRESHOLD_GAP - 1e-12
    assert abs(float(fit["actual_watch_rate"]) - float(fit["target_watch_rate"])) < 0.02
    assert abs(float(fit["actual_action_rate"]) - float(fit["target_action_rate"])) < 0.02


@pytest.mark.unit
def test_derive_ops_thresholds_fallback_without_hab_prob(ops_module: ModuleType) -> None:
    fit = ops_module.derive_ops_thresholds([pd.DataFrame({"ops_risk": [0.1, 0.4, 0.7]})])

    assert fit["method"] == "fallback_static_missing_hab_prob"
    assert fit["watch"] == pytest.approx(ops_module.WATCH_THRESHOLD)
    assert fit["action"] == pytest.approx(ops_module.ACTION_THRESHOLD)


@pytest.mark.unit
def test_monthly_table_and_top_events_logic(ops_module: ModuleType) -> None:
    df = pd.DataFrame(
        {
            "month": ["2025-01", "2025-01", "2025-02"],
            "datetime": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z", "2025-02-02T00:00:00Z"], utc=True
            ),
            "scene_id": ["A", "A", "B"],
            "tile": ["t1", "t2", "t3"],
            "chip_id": ["c1", "c2", "c3"],
            "ops_risk": [0.3, 0.9, 0.2],
            "hab_prob": [0.3, 0.8, 0.2],
            "p_frcnn_r50_med": [0.2, 0.9, 0.1],
            "p_frcnn_mb_med": [0.2, 0.8, 0.1],
            "p_ssd_mb_med": [0.2, 0.7, 0.1],
            "oci_proxy": [0.1, 0.2, 0.3],
            "oci_proxy_adj": [0.1, 0.2, 0.3],
            "seasonality_proxy_adj": [0.2, 0.3, 0.4],
            "sst": [27.1, 28.2, 26.9],
            "chlor_a": [0.4, 0.5, 0.2],
            "kd490": [0.1, 0.2, 0.1],
            "nflh": [0.05, 0.06, 0.04],
        }
    )

    monthly = ops_module.monthly_table(df, watch_threshold=0.5, action_threshold=0.8)
    top = ops_module.top_events(df, n=5, risk_col="ops_risk")

    assert [row["month"] for row in monthly] == ["2025-01", "2025-02"]
    assert monthly[0]["status"] == "action"
    assert monthly[1]["status"] == "normal"
    assert len(top) == 2  # duplicated scene/time in 2025-01 is deduplicated
    assert top[0]["scene_id"] == "A"
    assert top[0]["ops_risk"] == pytest.approx(0.9)


@pytest.mark.unit
def test_cadence_and_month_helpers(ops_module: ModuleType) -> None:
    cadence = ops_module.cadence_hours(
        pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2025-01-01T00:00:00Z", "2025-01-01T06:00:00Z", "2025-01-01T18:00:00Z"], utc=True
                )
            }
        )
    )

    assert cadence["median"] == pytest.approx(9.0)
    assert cadence["p90"] == pytest.approx(11.4)
    assert cadence["max"] == pytest.approx(12.0)

    start_end = ops_module._month_start_end("2024-02")
    assert start_end is not None
    assert start_end[0] == date(2024, 2, 1)
    assert start_end[1] == date(2024, 2, 29)
    assert ops_module._month_start_end("bad-input") is None

    assert ops_module._date_midpoint_from_name("AQUA_MODIS.20250301_20250308.L3m.8D.CHL.chlor_a.4km.nc") == date(
        2025, 3, 4
    )
    assert ops_module._date_midpoint_from_name("AQUA_MODIS.20250301.L3m.MO.CHL.chlor_a.4km.nc") == date(2025, 3, 1)
