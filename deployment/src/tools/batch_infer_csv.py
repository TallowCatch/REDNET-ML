# deployment/src/batch_infer_csv.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

# Reuse your inference pipeline (keeps logic identical)
from inference import run_inference


def load_json(p: Path) -> dict:
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser(description="Batch inference on a CSV file.")
    ap.add_argument("--artifacts_dir", required=True, help="Path to deployment/artifacts")
    ap.add_argument("--input_csv", required=True, help="CSV to run inference on")
    ap.add_argument("--output_csv", required=True, help="Where to write output CSV")
    ap.add_argument(
        "--month_col",
        default=None,
        help="Optional override for month key column name (default: from features.json or 'month_key')",
    )
    ap.add_argument(
        "--id_cols",
        default="",
        help="Optional comma-separated list of ID columns to keep first in output (e.g. tile,scene_id,region_key)",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="If set, error on missing required raw columns; otherwise missing columns become NaN and get filled.",
    )
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    features_cfg = load_json(artifacts_dir / "features.json")

    # Read input
    in_path = Path(args.input_csv)
    df = pd.read_csv(in_path)

    # Optional: enforce expected raw columns exist (best for production safety)
    required_raw = set(features_cfg.get("raw_required_columns", []))
    if required_raw:
        missing = sorted([c for c in required_raw if c not in df.columns])
        if missing and args.strict:
            raise SystemExit(
                "❌ Missing required raw columns:\n"
                + "\n".join([f"  - {m}" for m in missing])
                + "\nFix your input CSV or update artifacts/features.json raw_required_columns."
            )

    # If user wants to override month col, rename into what build_features expects
    month_col = args.month_col or features_cfg.get("month_col", "month_key")
    if month_col != features_cfg.get("month_col", "month_key"):
        # We override what inference/build_features will read by temporarily renaming.
        # run_inference uses features_cfg['month_col'] if present, else 'month_key'.
        # Easiest: make sure df contains that column name.
        expected_month_col = features_cfg.get("month_col", "month_key")
        if month_col not in df.columns:
            raise SystemExit(f"❌ --month_col '{month_col}' not found in input CSV columns.")
        if expected_month_col not in df.columns:
            df = df.rename(columns={month_col: expected_month_col})
        elif expected_month_col != month_col:
            # both exist; keep expected and ignore override
            pass

    # Run inference (returns input df with appended prob/pred/threshold_used)
    out = run_inference(df, artifacts_dir=artifacts_dir)

    # Reorder columns (optional)
    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    front = [c for c in id_cols if c in out.columns]
    tail = [c for c in ["prob", "pred", "threshold_used"] if c in out.columns]
    middle = [c for c in out.columns if c not in set(front + tail)]
    out = out[front + middle + tail]

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"✅ Wrote: {out_path}  (rows={len(out)})")


if __name__ == "__main__":
    main()
