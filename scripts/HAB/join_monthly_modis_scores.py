#!/usr/bin/env python3
"""
Join MODIS detector monthly scores with Sentinel-2 labeled rows,
while optionally preserving an existing p_tab column from a previous fusion table.

Inputs:
- Sentinel mined:     runs/datasets/hab_train_mined_aslabel.csv
- Sentinel nonleaky:  runs/datasets/hab_train_nonleaky.csv  (used only to backfill SST)
- MODIS det:          runs/fusion/p_frcnn_r50_mkey.csv
                      runs/fusion/p_frcnn_mb_mkey.csv
                      runs/fusion/p_frcnn_ssd_mkey.csv
- (Optional) old fusion table to preserve p_tab:
                      runs/fusion/fusion_training_table.csv

Output:
- runs/fusion/fusion_training_table.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

SENT_MINED = Path("runs/datasets/hab_train_mined_aslabel.csv")
SENT_NONLEAKY = Path("runs/datasets/hab_train_nonleaky.csv")

DET_R50 = Path("runs/fusion/p_frcnn_r50_mkey.csv")
DET_MB  = Path("runs/fusion/p_frcnn_mb_mkey.csv")
DET_SSD = Path("runs/fusion/p_frcnn_ssd_mkey.csv")

OUT_CSV = Path("runs/fusion/fusion_training_table.csv")


def ensure_month_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "month_key" in df.columns:
        return df

    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df["month_key"] = dt.dt.strftime("%Y-%m")
        return df

    if {"year", "month"} <= set(df.columns):
        df["month_key"] = (
            df["year"].astype("Int64").astype(str)
            + "-"
            + df["month"].astype("Int64").astype(str).str.zfill(2)
        )
        return df

    raise SystemExit("❌ No month_key and cannot build it (need datetime OR year+month).")


def choose_label_col(sent: pd.DataFrame) -> str:
    if "hab_label_final" in sent.columns:
        return "hab_label_final"
    if "hab_label" in sent.columns:
        return "hab_label"
    raise SystemExit("❌ Sentinel CSV has no hab_label / hab_label_final")


def backfill_sst_if_missing(sent: pd.DataFrame) -> pd.DataFrame:
    """If mined is missing sst, pull it from nonleaky via best available join keys."""
    if "sst" in sent.columns:
        return sent

    if not SENT_NONLEAKY.exists():
        print("⚠️  nonleaky CSV not found; cannot backfill sst.")
        return sent

    non = pd.read_csv(SENT_NONLEAKY)
    non = ensure_month_key(non)

    # Prefer exact tile+datetime match; else tile+month_key; else scene_id+datetime; else scene_id+month_key
    candidate_keys = [
        ["tile", "datetime"],
        ["tile", "month_key"],
        ["scene_id", "datetime"],
        ["scene_id", "month_key"],
    ]

    usable = None
    for keys in candidate_keys:
        if all(k in sent.columns for k in keys) and all(k in non.columns for k in keys):
            usable = keys
            break

    if usable is None:
        print("⚠️  Could not backfill sst (no usable join keys between mined and nonleaky).")
        return sent
    if "sst" not in non.columns:
        print("⚠️  Could not backfill sst (sst column not present in nonleaky).")
        return sent

    non_sub = non[usable + ["sst"]].drop_duplicates(usable)
    out = sent.merge(non_sub, on=usable, how="left")
    print(f"ℹ️  backfilled sst from nonleaky using keys={usable}")
    return out


def load_old_p_tab_map(old_csv: Path) -> tuple[dict[str, pd.DataFrame] | None, list[str] | None]:
    """
    Load old p_tab mapping keyed by the best available stable join keys.
    Returns (mapping_df_by_keys, chosen_keys)
    """
    if not old_csv.exists():
        return None, None

    old = pd.read_csv(old_csv)
    if "p_tab" not in old.columns:
        return None, None

    # Ensure month_key exists so we can fall back
    old = ensure_month_key(old)

    candidate_keys = [
        ["tile", "datetime"],
        ["tile", "month_key"],
        ["scene_id", "datetime"],
        ["scene_id", "month_key"],
    ]
    chosen = None
    for keys in candidate_keys:
        if all(k in old.columns for k in keys):
            chosen = keys
            break

    if chosen is None:
        return None, None

    old_map = old[chosen + ["p_tab"]].drop_duplicates(chosen)
    return {"df": old_map}, chosen


def merge_old_p_tab(fusion_df: pd.DataFrame, old_map: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """
    Merge in old p_tab as p_tab_old; then prefer old values over any newly-created p_tab.
    """
    out = fusion_df.copy()
    out = out.merge(old_map.rename(columns={"p_tab": "p_tab_old"}), on=keys, how="left")

    if "p_tab" in out.columns:
        # Keep old if present; else keep current p_tab
        out["p_tab"] = out["p_tab_old"].combine_first(out["p_tab"])
    else:
        out["p_tab"] = out["p_tab_old"]

    out = out.drop(columns=["p_tab_old"], errors="ignore")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", default=str(OUT_CSV))
    ap.add_argument("--preserve_p_tab_from", default=str(OUT_CSV),
                    help="Existing fusion table to preserve p_tab from (default: same as output path).")
    ap.add_argument("--recalc_p_tab", action="store_true",
                    help="If set, recompute p_tab (overrides preservation).")
    ap.add_argument("--p_tab_source", default="rednir_mean",
                    help="If --recalc_p_tab is set, use this column as the source feature (default: rednir_mean).")
    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    preserve_from = Path(args.preserve_p_tab_from)

    # Load mined Sentinel
    sent = pd.read_csv(SENT_MINED)
    label_col = choose_label_col(sent)
    sent = ensure_month_key(sent)

    # Backfill SST if missing in mined
    sent = backfill_sst_if_missing(sent)

    # Load old p_tab map (if available)
    old_map_pack, old_keys = load_old_p_tab_map(preserve_from)
    old_map = old_map_pack["df"] if old_map_pack else None

    if old_map is not None:
        print(f"ℹ️  Found old p_tab in {preserve_from}")
        print(f"ℹ️  Will preserve using keys={old_keys}")
    else:
        print("ℹ️  No old p_tab found to preserve (or file missing / no usable keys).")

    # Load detectors
    r50 = pd.read_csv(DET_R50)
    mb  = pd.read_csv(DET_MB)
    ssd = pd.read_csv(DET_SSD)

    # Collapse to per-month medians
    r50_m = r50.groupby("month_key")["p_frcnn_r50"].median().rename("p_frcnn_r50_med")
    mb_m  = mb.groupby("month_key")["p_frcnn_mb"].median().rename("p_frcnn_mb_med")

    ssd_score_col = None
    for cand in ["p_frcnn_ssd", "p_ssd_mb"]:
        if cand in ssd.columns:
            ssd_score_col = cand
            break
    if ssd_score_col is None:
        raise SystemExit("❌ SSD CSV is missing p_frcnn_ssd / p_ssd_mb")

    ssd_m = ssd.groupby("month_key")[ssd_score_col].median().rename("p_ssd_mb_med")

    det_month = pd.concat([r50_m, mb_m, ssd_m], axis=1).reset_index()

    # Merge with Sentinel by month_key
    fusion_df = sent.merge(det_month, on="month_key", how="left")

    # Label
    fusion_df["hab_label"] = fusion_df[label_col].astype(int)

    # Either preserve old p_tab, OR recalc (explicit)
    if args.recalc_p_tab:
        src = args.p_tab_source
        if src not in fusion_df.columns:
            raise SystemExit(f"❌ Cannot recalc p_tab: source column '{src}' not found in fusion table.")
        fusion_df["p_tab"] = fusion_df[src]
        print(f"✅ Recalculated p_tab from {src} (overwrites any previous p_tab).")
    else:
        # Preserve old if possible; otherwise do nothing (leave as-is if it already exists)
        if old_map is not None and old_keys is not None:
            before_nonnull = fusion_df["p_tab"].notna().sum() if "p_tab" in fusion_df.columns else 0
            fusion_df = merge_old_p_tab(fusion_df, old_map, old_keys)
            after_nonnull = fusion_df["p_tab"].notna().sum()
            print(f"✅ Preserved p_tab from old table. non-null before={before_nonnull}, after={after_nonnull}")
        else:
            print("ℹ️  Not recalculating p_tab, and nothing to preserve. (Leaving p_tab as-is if present.)")

    # Write
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fusion_df.to_csv(out_csv, index=False)

    print(f"✅ wrote {out_csv} with {len(fusion_df)} rows")
    print("   columns include sst?", "sst" in fusion_df.columns)
    print("   columns include p_tab?", "p_tab" in fusion_df.columns)
    print("   MODIS coverage:\n", fusion_df[["p_frcnn_r50_med","p_frcnn_mb_med","p_ssd_mb_med"]].notna().sum())


if __name__ == "__main__":
    main()
