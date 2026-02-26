from __future__ import annotations

from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def drop_module(module_loader) -> ModuleType:
    return module_loader("scripts/postprocess/drop_empty_columns.py")


@pytest.fixture(scope="module")
def clean_module(module_loader) -> ModuleType:
    return module_loader("scripts/postprocess/clean_chip_indices.py")


@pytest.mark.unit
def test_is_empty_cell_tokens(drop_module: ModuleType) -> None:
    assert drop_module.is_empty_cell(None)
    assert drop_module.is_empty_cell("")
    assert drop_module.is_empty_cell("  ")
    assert drop_module.is_empty_cell("NaN")
    assert drop_module.is_empty_cell("null")
    assert not drop_module.is_empty_cell("0")
    assert not drop_module.is_empty_cell("value")


@pytest.mark.unit
def test_clean_chip_numeric_helpers(clean_module: ModuleType) -> None:
    assert clean_module.to_float("3.14") == pytest.approx(3.14)
    assert clean_module.to_float("bad") is None

    assert clean_module.near_zero(None)
    assert clean_module.near_zero(0.0)
    assert clean_module.near_zero(1e-12)
    assert not clean_module.near_zero(1e-3)
