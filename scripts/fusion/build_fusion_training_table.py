#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    bad = [c for c in df.columns if c.lower().startswith("unnamed")]
    if bad:
        df = df.drop(columns=bad)
    dup = pd.Index(df.columns).duplicated(keep="first")
    if dup.any():
        df = df.loc[:, ~dup]
    return df

def normalize_tile(series: pd.Series) -> pd.Series:
    return series.astype(str).apply(lambda x: Path(x).name)

def main():
    # --- paths (edit if needed) ---
    template_csv = Path("runs/fusion/training_tables_draft/fusion_training_table_clean.csv")  # your schema/template
    plants_dir   = Path("/Users/ameerfiras/REDNET-ML/rednet-risk-viewer/public/data")
    out_csv      = Path("runs/fusion/fusion_training_table_clean_populated.csv")

    tmpl = clean_columns(pd.read_csv(template_csv))
    template_cols = list(tmpl.columns)

    plant_files = sorted(plants_dir.glob("plant_*_hab.csv"))
    if not plant_files:
        raise SystemExit(f"No plant_*.csv found in: {plants_dir}")

    frames = []
    for f in plant_files:
        df = clean_columns(pd.read_csv(f))

        # normalize tile names
        if "tile" in df.columns:
            df["tile"] = normalize_tile(df["tile"])
        elif "chip_id" in df.columns:
            # if a plant file ever lacks tile but has chip_id, use chip_id as tile
            df["tile"] = normalize_tile(df["chip_id"])

        # build output strictly with template columns
        out = pd.DataFrame(index=df.index)
        for c in template_cols:
            if c in df.columns:
                out[c] = df[c]
            else:
                out[c] = np.nan

        # ensure required label columns exist (they will be NaN unless present)
        for lab in ["hab_label", "hab_label_heuristic", "hab_label_final"]:
            if lab in out.columns:
                # keep as-is (NaN is fine)
                pass

        frames.append(out)
        print(f"[ok] {f.name}: rows={len(out)}")

    merged = pd.concat(frames, ignore_index=True)

    # drop exact duplicates
    merged = merged.drop_duplicates()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"\n✓ saved: {out_csv}  rows={len(merged)} cols={len(merged.columns)}")

    # quick sanity check
    print("\n[coverage]")
    for c in ["p_tab","p_frcnn_r50_med","p_frcnn_mb_med","p_ssd_mb_med","sst","chlor_a","kd490","nflh"]:
        if c in merged.columns:
            print(f"  {c}: non-null {(merged[c].notna().mean()*100):.1f}%")

if __name__ == "__main__":
    main()
