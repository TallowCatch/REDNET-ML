from __future__ import annotations

from types import ModuleType

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def prep_module(module_loader) -> ModuleType:
    return module_loader("scripts/HAB/preparation/make_HAB_train_nonleaky.py")


@pytest.mark.unit
def test_season_of_month_mapping(prep_module: ModuleType) -> None:
    assert prep_module.season_of_month(1) == "winter"
    assert prep_module.season_of_month(3) == "spring"
    assert prep_module.season_of_month(6) == "summer"
    assert prep_module.season_of_month(10) == "autumn"
    assert prep_module.season_of_month(12) == "winter"


@pytest.mark.unit
def test_derive_month_key_variants(prep_module: ModuleType) -> None:
    assert prep_module.derive_month_key("S2A_MSIL2A_20250723T065621") == "2025-07"
    assert prep_module.derive_month_key("scene.2025_08_xyz") == "2025-08"
    assert prep_module.derive_month_key("scene-2025-09-run") == "2025-09"
    assert prep_module.derive_month_key("scene-2025-99-run") is None
    assert prep_module.derive_month_key("no-date") is None


@pytest.mark.unit
def test_add_time_features_outputs_expected_columns(prep_module: ModuleType) -> None:
    df = pd.DataFrame(
        {
            "datetime": [
                "2025-01-15T00:00:00Z",
                "2025-07-15T00:00:00Z",
                "bad-date",
            ]
        }
    )
    out = prep_module.add_time_features(df)

    expected = {
        "year",
        "month",
        "month_sin",
        "month_cos",
        "season",
        "season_winter",
        "season_spring",
        "season_summer",
        "season_autumn",
    }
    assert expected.issubset(out.columns)
    assert int(out.loc[0, "month"]) == 1
    assert int(out.loc[1, "month"]) == 7
    # invalid datetime falls back to winter via fillna(1)
    assert out.loc[2, "season"] == "winter"
    assert int(
        out.loc[0, ["season_winter", "season_spring", "season_summer", "season_autumn"]].sum()
    ) == 1


@pytest.mark.unit
def test_add_time_features_no_datetime_is_noop(prep_module: ModuleType) -> None:
    df = pd.DataFrame({"x": [1, 2]})
    out = prep_module.add_time_features(df)
    assert list(out.columns) == ["x"]
