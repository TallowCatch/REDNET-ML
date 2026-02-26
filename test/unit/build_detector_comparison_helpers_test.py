from __future__ import annotations

from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def compare_module(module_loader) -> ModuleType:
    return module_loader("scripts/HAB/compare/build_detector_comparison.py")


@pytest.mark.unit
def test_binary_metrics_default_threshold(compare_module: ModuleType) -> None:
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]

    m = compare_module.binary_metrics(y_true, y_score)

    assert m["acc"] == pytest.approx(1.0)
    assert m["prec"] == pytest.approx(1.0)
    assert m["rec"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)


@pytest.mark.unit
def test_find_first_existing_respects_root_priority(compare_module: ModuleType, tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    (second / "target.csv").write_text("x\n1\n")
    (first / "other.csv").write_text("x\n2\n")

    found = compare_module.find_first_existing(
        candidates=["missing.csv", "target.csv", "other.csv"],
        roots=[first, second],
    )
    assert found is not None
    assert found.name == "other.csv"
