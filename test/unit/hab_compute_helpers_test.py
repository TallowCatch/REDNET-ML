from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest


@pytest.fixture(scope="module")
def compute_module(module_loader) -> ModuleType:
    return module_loader("scripts/HAB/compute.py")


@pytest.mark.unit
def test_pr_points_shapes_and_f1_bounds(compute_module: ModuleType) -> None:
    y = np.array([0, 1, 0, 1], dtype=int)
    p = np.array([0.1, 0.9, 0.2, 0.8], dtype=float)

    prec, rec, thr, f1 = compute_module.pr_points(y, p)

    assert len(prec) == len(rec) == len(f1) == len(thr) + 1
    assert np.nanmax(f1) <= 1.0 + 1e-12


@pytest.mark.unit
def test_best_f1_point_returns_valid_operating_point(compute_module: ModuleType) -> None:
    y = np.array([0, 0, 1, 1], dtype=int)
    p = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    prec, rec, thr, f1 = compute_module.pr_points(y, p)

    op = compute_module.best_f1_point(prec, rec, thr, f1)

    assert 0.0 <= op["threshold"] <= 1.0
    assert 0.0 <= op["precision"] <= 1.0
    assert 0.0 <= op["recall"] <= 1.0
    assert 0.0 <= op["f1"] <= 1.0


@pytest.mark.unit
def test_pick_for_precision_and_recall_logic(compute_module: ModuleType) -> None:
    prec = np.array([1.0, 0.6, 0.7, 0.9])
    rec = np.array([0.0, 0.8, 0.8, 0.4])
    thr = np.array([0.2, 0.4, 0.6])
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)

    p_op = compute_module.pick_for_precision(prec, rec, thr, f1, target_prec=0.65)
    assert p_op is not None
    assert p_op["threshold"] == pytest.approx(0.4)
    assert p_op["precision"] == pytest.approx(0.7)
    assert compute_module.pick_for_precision(prec, rec, thr, f1, target_prec=0.95) is None

    r_op = compute_module.pick_for_recall(prec, rec, thr, f1, target_rec=0.75)
    assert r_op is not None
    assert r_op["threshold"] == pytest.approx(0.4)
    assert r_op["precision"] == pytest.approx(0.7)
    assert compute_module.pick_for_recall(prec, rec, thr, f1, target_rec=0.95) is None
