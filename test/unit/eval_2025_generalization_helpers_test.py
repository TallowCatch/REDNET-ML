from __future__ import annotations

from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def eval_module(module_loader) -> ModuleType:
    return module_loader("scripts/HAB/eval_2025_generalization.py")


@pytest.mark.unit
def test_real_plant_regex(eval_module: ModuleType) -> None:
    assert eval_module.is_real_plant("osm_way_123")
    assert not eval_module.is_real_plant("osm_way_abc")
    assert not eval_module.is_real_plant("plant_123")


@pytest.mark.unit
def test_drift_metrics_psi_and_ks(eval_module: ModuleType) -> None:
    ref = np.linspace(0.0, 1.0, 100)
    cur_same = ref.copy()
    cur_shifted = np.linspace(0.2, 1.2, 100)

    assert eval_module.psi(ref, cur_same, n_bins=10) == pytest.approx(0.0, abs=1e-12)
    assert eval_module.psi(ref, cur_shifted, n_bins=10) > 0.0
    assert eval_module.ks_statistic(ref, cur_same) == pytest.approx(0.0, abs=1e-12)
    assert eval_module.ks_statistic(ref, cur_shifted) > 0.0


@pytest.mark.unit
def test_parse_year_month_and_plant_id_helpers(eval_module: ModuleType) -> None:
    from_dt = pd.DataFrame({"datetime": ["2025-07-02T00:00:00Z"]})
    out_dt = eval_module.parse_year_month(from_dt)
    assert int(out_dt["year_"].iloc[0]) == 2025
    assert int(out_dt["month_"].iloc[0]) == 7

    from_cols = pd.DataFrame({"year": [2024], "month_num": [9]})
    out_cols = eval_module.parse_year_month(from_cols)
    assert int(out_cols["year_"].iloc[0]) == 2024
    assert int(out_cols["month_"].iloc[0]) == 9

    with pytest.raises(ValueError):
        eval_module.parse_year_month(pd.DataFrame({"x": [1]}))

    plant = eval_module.plant_from_train_filename(Path("plant_1079022886_hab.csv"))
    assert plant == "osm_way_1079022886"


@pytest.mark.unit
def test_monthly_risk_and_top_events(eval_module: ModuleType) -> None:
    df = pd.DataFrame(
        {
            "year_": [2025, 2025, 2025, 2025],
            "month_": [7, 7, 8, 8],
            "hab_prob": [0.2, 0.7, 0.4, 0.9],
            "tile": ["t1", "t2", "t3", "t4"],
            "scene_id": ["s1", "s2", "s3", "s4"],
            "datetime": [
                "2025-07-01T00:00:00Z",
                "2025-07-03T00:00:00Z",
                "2025-08-02T00:00:00Z",
                "2025-08-07T00:00:00Z",
            ],
            "month_key": ["2025-07", "2025-07", "2025-08", "2025-08"],
            "p_frcnn_r50_med": [0.1, 0.8, 0.2, 0.7],
            "p_frcnn_mb_med": [0.2, 0.7, 0.3, 0.8],
            "p_ssd_mb_med": [0.3, 0.6, 0.2, 0.9],
        }
    )

    monthly = eval_module.monthly_risk_table(df, "hab_prob", threshold=0.5)
    assert len(monthly) == 2
    july = monthly[(monthly["year_"] == 2025) & (monthly["month_"] == 7)].iloc[0]
    august = monthly[(monthly["year_"] == 2025) & (monthly["month_"] == 8)].iloc[0]
    assert july["alert_rate"] == pytest.approx(0.5)
    assert august["alert_rate"] == pytest.approx(0.5)
    assert july["prob_p95"] > july["prob_mean"]

    events = eval_module.top_events(df, "hab_prob", topk=3)
    assert len(events) == 3
    assert events.iloc[0]["hab_prob"] == pytest.approx(0.9)
    assert events.iloc[1]["hab_prob"] == pytest.approx(0.7)
