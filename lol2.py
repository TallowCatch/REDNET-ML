#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

MINED = Path("runs/datasets/plants_mined_train.csv")

# Folder where your plant CSVs live (adjust)
PLANT_DIR = Path("rednet-risk-viewer/public/data")

DET_COLS = ["p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med"]
KEY = "tile"

def norm_tile(x: str) -> str:
    # keep only the basename (handles accidental paths)
    return Path(str(x)).name

def main():
    mined = pd.read_csv(MINED)
    if KEY not in mined.columns:
        raise SystemExit(f"mined CSV missing '{KEY}'")

    mined[KEY] = mined[KEY].map(norm_tile)

    plant_files = sorted(PLANT_DIR.glob("plant_*_hab.csv"))
    if not plant_files:
        raise SystemExit(f"No plant_*.csv found in {PLANT_DIR}")

    parts = []
    for f in plant_files:
        df = pd.read_csv(f)
        if KEY not in df.columns:
            continue
        have = [c for c in DET_COLS if c in df.columns]
        if not have:
            continue

        df = df[[KEY] + have].copy()
        df[KEY] = df[KEY].map(norm_tile)
        parts.append(df)

    if not parts:
        raise SystemExit("Found no plant files containing detector columns.")

    det = pd.concat(parts, ignore_index=True)

    # If duplicates exist, take first non-null per tile
    det = (
        det.sort_values(DET_COLS)  # not perfect, but helps stability
           .groupby(KEY, as_index=False)
           .agg({c: "first" for c in DET_COLS if c in det.columns})
    )

    out = mined.merge(det, on=KEY, how="left")

    print("Rows:", len(out))
    for c in DET_COLS:
        if c in out.columns:
            print(f"{c} coverage: {out[c].notna().mean():.3f}")

    OUT = MINED.with_name(MINED.stem + "_with_detectors.csv")
    out.to_csv(OUT, index=False)
    print("Wrote:", OUT)

if __name__ == "__main__":
    main()
