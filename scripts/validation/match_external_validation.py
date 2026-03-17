#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common


def _add_match_columns(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    out["assigned_plant_id"] = None
    out["assigned_plant_name"] = None
    out["assigned_by"] = None
    out["assigned_distance_km"] = np.nan
    out["matched_prediction_datetime"] = pd.Series([None] * len(out), dtype="object")
    out["match_day_diff"] = np.nan
    out["within_primary_window"] = False
    out["within_sensitivity_window"] = False
    out["match_status"] = "unmatched"
    for col in ["ops_risk", "hab_prob", "p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med", "hab_label"]:
        out[f"matched_{col}"] = np.nan
    return out


def match_event_rows(
    df: pd.DataFrame,
    plants: pd.DataFrame,
    location_map: dict[str, str],
    plant_predictions: dict[str, pd.DataFrame],
    primary_window_days: int,
    sensitivity_window_days: int,
    max_distance_km: float,
) -> pd.DataFrame:
    common.ensure_columns(df, common.EVENT_REQUIRED_COLUMNS, "event_validation")
    out = _add_match_columns(df)

    out["event_date_start_ts"] = out["event_date_start"].map(common.parse_date_start)
    out["event_date_end_ts"] = out["event_date_end"].map(common.parse_date_end)
    out["event_midpoint_utc"] = [
        common.midpoint_timestamp(s, e) for s, e in zip(out["event_date_start_ts"], out["event_date_end_ts"])
    ]

    plant_lookup = plants.set_index("plant_id")[["name"]].to_dict("index")

    for idx, row in out.iterrows():
        assigned_id, assigned_by, dist_km = common.assign_plant(
            row.to_dict(), plants, location_map, max_distance_km=max_distance_km
        )
        if assigned_id is None:
            out.at[idx, "match_status"] = "no_plant_assignment"
            continue

        out.at[idx, "assigned_plant_id"] = assigned_id
        out.at[idx, "assigned_plant_name"] = plant_lookup[assigned_id]["name"]
        out.at[idx, "assigned_by"] = assigned_by
        out.at[idx, "assigned_distance_km"] = dist_km

        pred_df = plant_predictions.get(assigned_id)
        if pred_df is None or pred_df.empty:
            out.at[idx, "match_status"] = "missing_plant_predictions"
            continue

        target_ts = row["event_midpoint_utc"]
        match = common.nearest_prediction_row(pred_df, target_ts)
        if match is None:
            out.at[idx, "match_status"] = "no_temporal_match"
            continue

        delta_days = common.day_delta(match["datetime"], target_ts)
        out.at[idx, "matched_prediction_datetime"] = match["datetime"]
        out.at[idx, "match_day_diff"] = delta_days
        out.at[idx, "within_primary_window"] = bool(delta_days is not None and delta_days <= primary_window_days)
        out.at[idx, "within_sensitivity_window"] = bool(
            delta_days is not None and delta_days <= sensitivity_window_days
        )
        out.at[idx, "match_status"] = "matched"
        for col in ["ops_risk", "hab_prob", "p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med", "hab_label"]:
            if col in match.index:
                out.at[idx, f"matched_{col}"] = match[col]

    return out


def match_insitu_rows(
    df: pd.DataFrame,
    plants: pd.DataFrame,
    location_map: dict[str, str],
    plant_predictions: dict[str, pd.DataFrame],
    primary_window_days: int,
    sensitivity_window_days: int,
    max_distance_km: float,
) -> pd.DataFrame:
    common.ensure_columns(df, common.INSITU_REQUIRED_COLUMNS, "insitu_validation")
    out = _add_match_columns(df)
    out["datetime_utc"] = out["datetime_utc"].map(common.parse_datetime_utc)
    out["plant_id"] = out["plant_id"].map(common.normalize_plant_id)
    out["hab_event"] = pd.to_numeric(out["hab_event"], errors="coerce").fillna(0).astype(int).clip(0, 1)

    plant_lookup = plants.set_index("plant_id")[["name"]].to_dict("index")

    for idx, row in out.iterrows():
        assigned_id, assigned_by, dist_km = common.assign_plant(
            row.to_dict(), plants, location_map, max_distance_km=max_distance_km
        )
        if assigned_id is None:
            out.at[idx, "match_status"] = "no_plant_assignment"
            continue

        out.at[idx, "assigned_plant_id"] = assigned_id
        out.at[idx, "assigned_plant_name"] = plant_lookup[assigned_id]["name"]
        out.at[idx, "assigned_by"] = assigned_by
        out.at[idx, "assigned_distance_km"] = dist_km

        pred_df = plant_predictions.get(assigned_id)
        if pred_df is None or pred_df.empty:
            out.at[idx, "match_status"] = "missing_plant_predictions"
            continue

        target_ts = row["datetime_utc"]
        match = common.nearest_prediction_row(pred_df, target_ts)
        if match is None:
            out.at[idx, "match_status"] = "no_temporal_match"
            continue

        delta_days = common.day_delta(match["datetime"], target_ts)
        out.at[idx, "matched_prediction_datetime"] = match["datetime"]
        out.at[idx, "match_day_diff"] = delta_days
        out.at[idx, "within_primary_window"] = bool(delta_days is not None and delta_days <= primary_window_days)
        out.at[idx, "within_sensitivity_window"] = bool(
            delta_days is not None and delta_days <= sensitivity_window_days
        )
        out.at[idx, "match_status"] = "matched"
        for col in ["ops_risk", "hab_prob", "p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med", "hab_label"]:
            if col in match.index:
                out.at[idx, f"matched_{col}"] = match[col]

    return out


def _read_csv_or_empty(path: Path, required_cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Input CSV not found: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=required_cols)
    if df.empty:
        return pd.DataFrame(columns=required_cols)
    return df


def main() -> None:
    ap = argparse.ArgumentParser("Match external event or in-situ records to REDNET plant prediction series.")
    ap.add_argument("--mode", choices=["event", "insitu"], required=True)
    ap.add_argument("--events_csv", default="data/external_validation/oman_hab_events.csv")
    ap.add_argument("--insitu_csv", default="data/external_validation/insitu_records.csv")
    ap.add_argument("--plants_json", default="rednet-risk-viewer/public/data/plants.json")
    ap.add_argument("--prediction_glob", default="rednet-risk-viewer/public/data/plant_*_hab.csv")
    ap.add_argument("--location_map_csv", default="data/external_validation/location_to_plant_map.csv")
    ap.add_argument("--primary_window_days", type=int, default=common.DEFAULT_PRIMARY_WINDOW_DAYS)
    ap.add_argument("--sensitivity_window_days", type=int, default=common.DEFAULT_SENSITIVITY_WINDOW_DAYS)
    ap.add_argument("--max_nearest_plant_km", type=float, default=common.DEFAULT_MAX_PLANT_DISTANCE_KM)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    plants = common.load_plants(Path(args.plants_json))
    location_map = common.load_location_map(Path(args.location_map_csv))
    plant_predictions = common.load_prediction_series(args.prediction_glob)

    out_csv = Path(args.out_csv) if args.out_csv else Path(
        "runs/eval/external_validation/matched_external_events.csv"
        if args.mode == "event"
        else "runs/eval/external_validation/matched_insitu_records.csv"
    )

    if args.mode == "event":
        base = _read_csv_or_empty(Path(args.events_csv), common.EVENT_REQUIRED_COLUMNS)
        matched = match_event_rows(
            base,
            plants,
            location_map,
            plant_predictions,
            args.primary_window_days,
            args.sensitivity_window_days,
            args.max_nearest_plant_km,
        )
    else:
        base = _read_csv_or_empty(Path(args.insitu_csv), common.INSITU_REQUIRED_COLUMNS)
        matched = match_insitu_rows(
            base,
            plants,
            location_map,
            plant_predictions,
            args.primary_window_days,
            args.sensitivity_window_days,
            args.max_nearest_plant_km,
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(out_csv, index=False)
    print(f"[ok] wrote {out_csv} (rows={len(matched)})")


if __name__ == "__main__":
    main()
