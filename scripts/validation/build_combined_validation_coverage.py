#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing {label} CSV: {path}")
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser("Combine strict event validation and regional advisory coverage by AOI.")
    ap.add_argument(
        "--strict_csv",
        default="runs/eval/external_validation/aoi_event_validation_by_plant.csv",
    )
    ap.add_argument(
        "--advisory_csv",
        default="runs/eval/external_validation/regional_advisories/aoi_event_validation_by_plant.csv",
    )
    ap.add_argument(
        "--out_csv",
        default="runs/eval/external_validation/combined_aoi_validation_coverage.csv",
    )
    ap.add_argument(
        "--out_md",
        default="runs/eval/external_validation/combined_aoi_validation_coverage.md",
    )
    args = ap.parse_args()

    strict = _load_csv(Path(args.strict_csv), "strict validation")
    advisory = _load_csv(Path(args.advisory_csv), "regional advisory validation")

    keep = ["plant_id", "plant_name", "n_events_total", "n_primary_matched", "n_sensitivity_matched"]
    strict = strict[keep].rename(
        columns={
            "n_events_total": "strict_events_total",
            "n_primary_matched": "strict_primary_matched",
            "n_sensitivity_matched": "strict_sensitivity_matched",
        }
    )
    advisory = advisory[keep].rename(
        columns={
            "n_events_total": "advisory_events_total",
            "n_primary_matched": "advisory_primary_matched",
            "n_sensitivity_matched": "advisory_sensitivity_matched",
        }
    )

    combined = strict.merge(advisory, on=["plant_id", "plant_name"], how="outer").fillna(0)
    for col in combined.columns:
        if col not in {"plant_id", "plant_name"}:
            combined[col] = combined[col].astype(int)

    combined["has_strict_external_support"] = (combined["strict_events_total"] > 0).astype(int)
    combined["has_regional_advisory_support"] = (combined["advisory_events_total"] > 0).astype(int)
    combined["has_any_external_support"] = (
        (combined["has_strict_external_support"] == 1) | (combined["has_regional_advisory_support"] == 1)
    ).astype(int)

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)

    lines = [
        "# Combined AOI Validation Coverage",
        "",
        "This table combines strict public-event validation and lower-precision regional advisory support.",
        "Strict event validation is stronger. Regional advisory support is weaker and should not be treated as plant-day confirmation.",
        "",
        f"- AOIs with strict external support: {int(combined['has_strict_external_support'].sum())}",
        f"- AOIs with regional advisory support: {int(combined['has_regional_advisory_support'].sum())}",
        f"- AOIs with any external or advisory support: {int(combined['has_any_external_support'].sum())} / {len(combined)}",
        "",
        "## By Plant",
        "",
    ]
    for _, row in combined.iterrows():
        lines.extend(
            [
                f"### {row['plant_name']}",
                f"- Strict external rows: {row['strict_events_total']}",
                f"- Strict primary matched: {row['strict_primary_matched']}",
                f"- Regional advisory rows: {row['advisory_events_total']}",
                f"- Regional advisory primary matched: {row['advisory_primary_matched']}",
                f"- Any support: {'yes' if row['has_any_external_support'] else 'no'}",
                "",
            ]
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_md}")


if __name__ == "__main__":
    main()
