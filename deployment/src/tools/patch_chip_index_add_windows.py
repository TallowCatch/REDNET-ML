#!/usr/bin/env python3
"""
Patch-style helper: ensures chip index rows include window coordinates.

You must adapt the part marked "TODO" to your chip generation logic.
Goal: chip_indices_clean.csv must include x0,y0,w,h (pixel window in source raster)

Why:
- Without x0/y0/w/h you cannot compute per-chip lat/lon.
"""

import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip_index_csv", required=True, help="Existing chip_indices_clean.csv")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.chip_index_csv)

    # If you already have them, just pass through
    needed = {"x0", "y0", "w", "h"}
    if needed.issubset(df.columns):
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"✅ already had window cols, wrote {args.out_csv} rows={len(df)}")
        return

    # TODO: you must reconstruct these from your tiling scheme.
    # If you cannot reconstruct them, you need to re-run chip index generation
    # (NOT necessarily regenerate images) with deterministic tiling params.

    raise SystemExit(
        "❌ chip index has no x0/y0/w/h, and this script can't guess them.\n"
        "Fix: modify your chip index writer to include x0,y0,w,h per chip."
    )

if __name__ == "__main__":
    main()
