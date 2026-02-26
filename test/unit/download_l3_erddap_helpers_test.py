from __future__ import annotations

from datetime import date
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def erddap_module(module_loader) -> ModuleType:
    return module_loader("scripts/download/download_l3_erddap.py")


@pytest.mark.unit
def test_month_iter_inclusive_and_mid_month(erddap_module: ModuleType) -> None:
    months = list(erddap_module.month_iter(2024, 2024))

    assert len(months) == 12
    assert months[0] == date(2024, 1, 15)
    assert months[1] == date(2024, 2, 15)
    assert months[-1] == date(2024, 12, 15)


@pytest.mark.unit
def test_month_iter_multi_year(erddap_module: ModuleType) -> None:
    months = list(erddap_module.month_iter(2023, 2024))

    assert len(months) == 24
    assert months[0] == date(2023, 1, 15)
    assert months[-1] == date(2024, 12, 15)
