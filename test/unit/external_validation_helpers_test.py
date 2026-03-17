from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def common_module(module_loader) -> ModuleType:
    return module_loader("scripts/validation/common.py")


@pytest.fixture(scope="module")
def build_module(module_loader) -> ModuleType:
    return module_loader("scripts/validation/build_external_events_table.py")


@pytest.fixture(scope="module")
def match_module(module_loader) -> ModuleType:
    return module_loader("scripts/validation/match_external_validation.py")


@pytest.fixture(scope="module")
def score_module(module_loader) -> ModuleType:
    return module_loader("scripts/validation/score_external_validation.py")


@pytest.mark.unit
def test_normalize_events_rejects_chlorophyll_only_positive(build_module: ModuleType) -> None:
    df = pd.DataFrame(
        [
            {
                "event_id": "bad_event",
                "source_type": "bulletin",
                "source_url": "https://example.com",
                "event_date_start": "2024-01-01",
                "event_date_end": "2024-01-01",
                "location_name": "Muscat",
                "plant_id": "",
                "lat": "",
                "lon": "",
                "event_class": "chlorophyll_only",
                "severity": "low",
                "evidence_text": "chlorophyll value only",
                "confidence": "medium",
                "external_positive": 1,
            }
        ]
    )

    with pytest.raises(SystemExit):
        build_module.normalize_events(df)


@pytest.mark.unit
def test_assign_plant_prefers_location_map_and_nearest(common_module: ModuleType) -> None:
    plants = pd.DataFrame(
        [
            {"plant_id": "1", "name": "Alpha", "lat": 23.6, "lon": 58.4},
            {"plant_id": "2", "name": "Beta", "lat": 22.6, "lon": 59.4},
        ]
    )
    location_map = {"qurum": "1"}

    by_map = common_module.assign_plant({"location_name": "Qurum"}, plants, location_map)
    assert by_map[0] == "1"
    assert by_map[1] == "location_map"

    by_nearest = common_module.assign_plant({"lat": 22.61, "lon": 59.39}, plants, {}, max_distance_km=30.0)
    assert by_nearest[0] == "2"
    assert by_nearest[1] == "nearest_plant"


@pytest.mark.unit
def test_match_event_rows_assigns_and_matches_window(match_module: ModuleType) -> None:
    plants = pd.DataFrame([{"plant_id": "386838289", "name": "Ghubrah", "lat": 23.6, "lon": 58.4}])
    location_map = {"qurum": "386838289"}
    pred = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2023-08-02T06:00:00Z", "2023-08-10T06:00:00Z"], utc=True),
            "hab_prob": [0.8, 0.2],
            "p_frcnn_r50_med": [0.9, 0.1],
            "p_frcnn_mb_med": [0.6, 0.1],
            "p_ssd_mb_med": [0.5, 0.1],
            "hab_label": [0, 0],
        }
    )
    plant_predictions = {"386838289": pred}
    events = pd.DataFrame(
        [
            {
                "event_id": "evt1",
                "source_type": "public_news",
                "source_url": "https://example.com",
                "event_date_start": "2023-08-02",
                "event_date_end": "2023-08-02",
                "location_name": "Qurum",
                "plant_id": "",
                "lat": "",
                "lon": "",
                "event_class": "red_tide_health_warning",
                "severity": "high",
                "evidence_text": "test",
                "confidence": "high",
                "external_positive": 1,
            }
        ]
    )

    out = match_module.match_event_rows(
        events,
        plants,
        location_map,
        plant_predictions,
        primary_window_days=3,
        sensitivity_window_days=7,
        max_distance_km=100.0,
    )

    assert out.loc[0, "assigned_plant_id"] == "386838289"
    assert bool(out.loc[0, "within_primary_window"]) is True
    assert out.loc[0, "match_status"] == "matched"
    assert float(out.loc[0, "matched_hab_prob"]) == pytest.approx(0.8)


@pytest.mark.unit
def test_score_event_mode_falls_back_without_class_balance(
    score_module: ModuleType, tmp_path: Path
) -> None:
    matched = pd.DataFrame(
        [
            {
                "event_id": "evt1",
                "external_positive": 1,
                "assigned_plant_id": "386838289",
                "event_date_start": "2023-08-02",
                "event_date_end": "2023-08-02",
                "matched_hab_prob": 0.8,
                "matched_prediction_datetime": "2023-08-02T06:00:00Z",
                "match_day_diff": 0.0,
                "within_primary_window": True,
                "within_sensitivity_window": True,
            }
        ]
    )
    matched_csv = tmp_path / "matched.csv"
    matched.to_csv(matched_csv, index=False)

    plant_csv = tmp_path / "plant_386838289_hab.csv"
    pd.DataFrame(
        {
            "datetime": ["2023-08-02T06:00:00Z", "2023-08-10T06:00:00Z"],
            "hab_prob": [0.8, 0.1],
        }
    ).to_csv(plant_csv, index=False)

    outdir = tmp_path / "out"
    score_module.score_event_mode(matched_csv, str(tmp_path / "plant_*_hab.csv"), outdir, 0.55, 0.62)

    summary = pd.read_json(outdir / "event_validation_summary.json", typ="series")
    assert summary["mode"] == "event_validation"
    assert summary["auroc"] is None
    assert summary["auprc"] is None
