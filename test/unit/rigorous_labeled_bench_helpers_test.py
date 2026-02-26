from __future__ import annotations

import math
from types import ModuleType

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def bench_module(module_loader) -> ModuleType:
    return module_loader("scripts/eval/rigorous_labeled_bench.py")


@pytest.mark.unit
def test_basic_helpers_pick_existing_and_sigmoid(bench_module: ModuleType) -> None:
    df = pd.DataFrame({"chip_id": ["scene_001"]})

    assert bench_module._pick_first_existing(df, ["scene_id", "chip_id"]) == "chip_id"
    assert bench_module._pick_first_existing(df, ["scene_id", "tile"]) is None
    assert bench_module.sigmoid(0.0) == pytest.approx(0.5)


@pytest.mark.unit
def test_safe_auc_helpers_handle_single_class(bench_module: ModuleType) -> None:
    y = np.array([1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4])

    assert math.isnan(bench_module.safe_roc_auc(y, s))
    assert math.isnan(bench_module.safe_pr_auc(y, s))


@pytest.mark.unit
def test_best_f1_threshold_and_eval_metrics(bench_module: ModuleType) -> None:
    y = np.array([0, 0, 1, 1], dtype=int)
    s = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)

    thr = bench_module.best_f1_threshold(y, s, n_grid=401)
    metrics = bench_module.eval_threshold_metrics(y, s, thr=0.5)

    assert 0.2 < thr < 0.8
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["kappa"] == pytest.approx(1.0)


@pytest.mark.unit
def test_bootstrap_ci_small_and_large_sample_behavior(bench_module: ModuleType) -> None:
    small_y = np.array([0, 1, 0], dtype=int)
    small_s = np.array([0.1, 0.9, 0.2], dtype=float)
    point, lo, hi = bench_module.bootstrap_ci(small_y, small_s, bench_module.safe_pr_auc, n_boot=50)

    assert math.isfinite(point)
    assert math.isnan(lo)
    assert math.isnan(hi)

    y = np.array(([0] * 20) + ([1] * 20), dtype=int)
    s = np.linspace(0.0, 1.0, 40)
    point2, lo2, hi2 = bench_module.bootstrap_ci(y, s, bench_module.safe_roc_auc, n_boot=200, seed=123)

    assert math.isfinite(point2)
    assert math.isfinite(lo2)
    assert math.isfinite(hi2)
    assert lo2 <= point2 <= hi2


@pytest.mark.unit
def test_rolling_time_and_group_splits(bench_module: ModuleType) -> None:
    df = pd.DataFrame(
        {
            "year_": [2023, 2024, 2025, 2025],
            "scene_id": ["a", "b", "c", "d"],
            "x": [1, 2, 3, 4],
        }
    )

    splits = list(bench_module.rolling_year_splits(df, start_year=2024, end_year=2025))
    assert len(splits) == 2
    assert splits[0][2] == 2024
    assert splits[1][2] == 2025

    tr, te = bench_module.time_split(df, train_end_year=2024, test_year=2025)
    assert set(tr["year_"].unique()) == {2023, 2024}
    assert set(te["year_"].unique()) == {2025}

    grouped = pd.DataFrame(
        {
            "scene_id": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
            "y": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    tr2, te2 = bench_module.group_split(grouped, test_size=0.25, seed=42)
    assert set(tr2["scene_id"]).isdisjoint(set(te2["scene_id"]))


@pytest.mark.unit
def test_group_id_uses_scene_root_for_chip_ids(bench_module: ModuleType) -> None:
    df = pd.DataFrame({"chip_id": ["sceneA_0001", "sceneA_0002", "sceneB_0001"]})
    ids = bench_module.make_group_id(df)

    assert ids.tolist() == ["sceneA", "sceneA", "sceneB"]

    no_group_df = pd.DataFrame({"foo": [1, 2, 3]})
    fallback = bench_module.make_group_id(no_group_df)
    assert np.array_equal(fallback, np.arange(3))


@pytest.mark.unit
def test_mcnemar_and_delong_edge_cases(bench_module: ModuleType) -> None:
    stable = bench_module.mcnemar_exact([1, 0, 1], [1, 0, 1], [1, 0, 1])
    assert stable["b"] == pytest.approx(0.0)
    assert stable["c"] == pytest.approx(0.0)
    assert stable["p_value"] == pytest.approx(1.0)

    asym = bench_module.mcnemar_exact([1, 1, 1, 0], [1, 1, 1, 1], [0, 0, 0, 1])
    assert asym["b"] == pytest.approx(3.0)
    assert asym["c"] == pytest.approx(0.0)
    assert asym["p_value"] == pytest.approx(0.25)

    de_long = bench_module.delong_roc_test(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
        np.array([0.2, 0.3, 0.7, 0.8]),
    )
    assert math.isnan(de_long["auc1"])
    assert math.isnan(de_long["auc2"])
    assert math.isnan(de_long["p_value"])
