from __future__ import annotations

import sys
from datetime import date
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def monthly_module(module_loader) -> ModuleType:
    return module_loader("scripts/download/make_obpg_filelists.py")


@pytest.fixture(scope="module")
def eight_day_module(module_loader) -> ModuleType:
    return module_loader("scripts/download/make_obpg_filelists_8d.py")


@pytest.mark.unit
def test_monthly_filename_generation_count_and_bounds(monthly_module: ModuleType) -> None:
    names = list(monthly_module.file_names(2024, 2024, "CHL", "chlor_a"))

    assert len(names) == 12
    assert names[0] == "AQUA_MODIS.20240101_20240131.L3m.MO.CHL.chlor_a.4km.nc"
    assert names[1] == "AQUA_MODIS.20240201_20240229.L3m.MO.CHL.chlor_a.4km.nc"
    assert names[-1] == "AQUA_MODIS.20241201_20241231.L3m.MO.CHL.chlor_a.4km.nc"


@pytest.mark.unit
def test_monthly_filename_generation_cross_year(monthly_module: ModuleType) -> None:
    names = list(monthly_module.file_names(2023, 2024, "KD", "Kd_490"))

    assert len(names) == 24
    assert names[0].endswith(".L3m.MO.KD.Kd_490.4km.nc")
    assert names[-1].startswith("AQUA_MODIS.20241201_20241231")


@pytest.mark.unit
def test_eight_day_bins_cover_full_year(eight_day_module: ModuleType) -> None:
    bins_2025 = list(eight_day_module.eight_day_bins(2025))
    bins_2024 = list(eight_day_module.eight_day_bins(2024))

    days_2025 = sum((e - s).days + 1 for s, e in bins_2025)
    days_2024 = sum((e - s).days + 1 for s, e in bins_2024)

    assert len(bins_2025) == 46
    assert len(bins_2024) == 46
    assert bins_2025[0] == (date(2025, 1, 1), date(2025, 1, 8))
    assert bins_2025[-1] == (date(2025, 12, 27), date(2025, 12, 31))
    assert bins_2024[-1] == (date(2024, 12, 26), date(2024, 12, 31))
    assert days_2025 == 365
    assert days_2024 == 366


@pytest.mark.unit
def test_eight_day_filename_format(eight_day_module: ModuleType) -> None:
    names = list(eight_day_module.file_names_8d(2025, 2025, "SST", "sst"))

    assert len(names) == 46
    assert names[0] == "AQUA_MODIS.20250101_20250108.L3m.8D.SST.sst.4km.nc"
    assert names[-1] == "AQUA_MODIS.20251227_20251231.L3m.8D.SST.sst.4km.nc"


@pytest.mark.unit
def test_monthly_main_writes_expected_filelists(tmp_path, monkeypatch, monthly_module: ModuleType) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--years", "2025", "2025", "--outdir", str(tmp_path)],
    )
    monthly_module.main()

    for key in monthly_module.PRODUCTS:
        path = tmp_path / f"filelist_{key}.txt"
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 12


@pytest.mark.unit
def test_eight_day_main_writes_expected_filelists(tmp_path, monkeypatch, eight_day_module: ModuleType) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--years", "2025", "2025", "--outdir", str(tmp_path)],
    )
    eight_day_module.main()

    for key in eight_day_module.PRODUCTS:
        path = tmp_path / f"filelist_8d_{key}.txt"
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 46
