#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common


def normalize_events(df: pd.DataFrame) -> pd.DataFrame:
    common.ensure_columns(df, common.EVENT_REQUIRED_COLUMNS, "event_validation")
    out = df.copy()
    for col in ["event_id", "source_type", "source_url", "location_name", "event_class", "severity", "evidence_text"]:
        out[col] = out[col].map(common.normalize_nullable_text)

    out["plant_id"] = out["plant_id"].map(common.normalize_plant_id)
    out["confidence"] = (
        out["confidence"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": "unknown", "": "unknown"})
    )
    out["external_positive"] = (
        pd.to_numeric(out["external_positive"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    )
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")

    starts = out["event_date_start"].map(common.parse_date_start)
    ends = out["event_date_end"].map(common.parse_date_end)
    bad_dates = starts.isna() | ends.isna() | (ends < starts)
    if bool(bad_dates.any()):
        bad_ids = out.loc[bad_dates, "event_id"].tolist()
        raise SystemExit(f"Invalid event date range for event_id values: {bad_ids}")
    out["event_date_start"] = starts.dt.strftime("%Y-%m-%d")
    out["event_date_end"] = ends.dt.strftime("%Y-%m-%d")

    pos_bad = (
        (out["external_positive"] == 1)
        & out["event_class"].fillna("").str.strip().str.lower().isin(common.DISALLOWED_PRIMARY_EVENT_CLASSES)
    )
    if bool(pos_bad.any()):
        bad_ids = out.loc[pos_bad, "event_id"].tolist()
        raise SystemExit(
            "Public chlorophyll-only or commentary rows cannot be marked external_positive=1. "
            f"Bad event_id values: {bad_ids}"
        )

    out = out.drop_duplicates(subset=["event_id"]).sort_values(["event_date_start", "event_id"]).reset_index(drop=True)
    return out[common.EVENT_REQUIRED_COLUMNS].copy()


def main() -> None:
    ap = argparse.ArgumentParser("Validate and normalize externally curated HAB event sources.")
    ap.add_argument("--in_csv", default="data/external_validation/oman_hab_events_seed.csv")
    ap.add_argument("--out_csv", default="data/external_validation/oman_hab_events.csv")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)
    if not in_csv.exists():
        raise SystemExit(f"Input CSV not found: {in_csv}")

    df = pd.read_csv(in_csv)
    out = normalize_events(df)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[ok] wrote normalized external events table to {out_csv} (rows={len(out)})")


if __name__ == "__main__":
    main()
