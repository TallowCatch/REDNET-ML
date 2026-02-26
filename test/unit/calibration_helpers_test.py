from __future__ import annotations

from types import ModuleType

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def calibration_module(module_loader) -> ModuleType:
    return module_loader("scripts/eval/calibration.py")


@pytest.mark.unit
def test_calibration_table_counts_and_rates(calibration_module: ModuleType) -> None:
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]

    table = calibration_module.calibration_table(y_true, y_prob, bins=2)

    assert isinstance(table, pd.DataFrame)
    assert int(table["count"].sum()) == 4
    assert table.loc[0, "true_rate"] == pytest.approx(0.0)
    assert table.loc[1, "true_rate"] == pytest.approx(1.0)
    assert table.loc[0, "mean_prob"] < table.loc[1, "mean_prob"]


@pytest.mark.unit
def test_calibration_table_retains_empty_bins(calibration_module: ModuleType) -> None:
    y_true = [0, 1]
    y_prob = [0.05, 0.07]

    table = calibration_module.calibration_table(y_true, y_prob, bins=4)

    assert len(table) == 4
    assert int((table["count"] == 0).sum()) >= 2
    assert table.loc[table["count"] == 0, "mean_prob"].isna().all()
