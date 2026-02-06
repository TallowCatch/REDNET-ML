#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

import re

PLANT_RE = re.compile(r"^osm_way_\d+$")

def is_real_plant(name: str) -> bool:
    return bool(PLANT_RE.match(str(name)))



# ──────────────────────────────────────────────────────────────────────────────
# Drift metrics
# ──────────────────────────────────────────────────────────────────────────────
def psi(ref: np.ndarray, cur: np.ndarray, *, n_bins: int = 10, eps: float = 1e-6) -> float:
    ref = ref.astype(float)
    cur = cur.astype(float)

    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 10 or len(cur) < 10:
        return float("nan")

    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(ref, qs))
    if len(edges) < 3:
        return float("nan")

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_dist = ref_counts / max(ref_counts.sum(), 1)
    cur_dist = cur_counts / max(cur_counts.sum(), 1)

    ref_dist = np.clip(ref_dist, eps, None)
    cur_dist = np.clip(cur_dist, eps, None)

    return float(np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)))


def ks_statistic(ref: np.ndarray, cur: np.ndarray) -> float:
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 10 or len(cur) < 10:
        return float("nan")

    ref_sorted = np.sort(ref)
    cur_sorted = np.sort(cur)

    all_vals = np.sort(np.unique(np.concatenate([ref_sorted, cur_sorted])))
    ref_cdf = np.searchsorted(ref_sorted, all_vals, side="right") / len(ref_sorted)
    cur_cdf = np.searchsorted(cur_sorted, all_vals, side="right") / len(cur_sorted)
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def try_ks_pvalue(ref: np.ndarray, cur: np.ndarray) -> float:
    try:
        from scipy.stats import ks_2samp  # type: ignore
        out = ks_2samp(ref[np.isfinite(ref)], cur[np.isfinite(cur)], alternative="two-sided", mode="auto")
        return float(out.pvalue)
    except Exception:
        return float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def parse_year_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # prefer explicit year/month_num if present
    if "year" in df.columns and "month_num" in df.columns:
        df["year_"] = pd.to_numeric(df["year"], errors="coerce")
        df["month_"] = pd.to_numeric(df["month_num"], errors="coerce")
        return df
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df["year_"] = dt.dt.year
        df["month_"] = dt.dt.month
        return df
    raise ValueError("No usable time columns found (need year+month_num or datetime).")


def plant_from_train_filename(path: Path) -> str:
    """
    rednet-risk-viewer/public/data/plant_1079022886_hab.csv -> osm_way_1079022886
    """
    m = re.search(r"plant_(\d+)", path.name)
    if not m:
        return "unknown"
    return f"osm_way_{m.group(1)}"


def monthly_risk_table(df: pd.DataFrame, prob_col: str, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["alert"] = (pd.to_numeric(df[prob_col], errors="coerce") >= threshold).astype(int)
    g = df.groupby(["year_", "month_"], dropna=False)
    out = g.agg(
        n=("alert", "size"),
        alert_rate=("alert", "mean"),
        prob_mean=(prob_col, "mean"),
        prob_p95=(prob_col, lambda x: float(np.nanpercentile(pd.to_numeric(x, errors="coerce"), 95))),
        prob_max=(prob_col, "max"),
    ).reset_index()
    return out


def top_events(df: pd.DataFrame, prob_col: str, topk: int) -> pd.DataFrame:
    cols = [c for c in ["tile", "scene_id", "datetime", "month_key", prob_col,
                       "p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med"] if c in df.columns]
    return df.sort_values(prob_col, ascending=False).head(topk)[cols].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser("REDNET generalization eval (drift + top-events + threshold policy)")

    # 2025 deployment outputs
    ap.add_argument("--by_plant_root", required=True, help="deployment/outputs/by_plant")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--prob_col", default="hab_prob")
    ap.add_argument("--train_end_year", type=int, default=2024)
    ap.add_argument("--test_year", type=int, default=2025)

    # historic reference (2017–2024)
    ap.add_argument("--train_glob", required=True,
                    help='glob for historic plant files, e.g. "rednet-risk-viewer/public/data/plant_*_hab.csv"')
    ap.add_argument("--train_prob_col", default=None, help="prob column in train files (default: --prob_col)")

    # threshold policy
    ap.add_argument("--threshold", type=float, required=True, help="operational threshold to evaluate alerts")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--include_misc_regions", action="store_true",
                help="Include non-plant folders like osm_way_ in per-plant outputs")


    args = ap.parse_args()

    root = Path(args.by_plant_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prob_col = args.prob_col
    train_prob_col = args.train_prob_col or prob_col
    thr = float(args.threshold)

    import glob

    # -------------------------
    # Load TRAIN reference (<=2024) from train_glob
    # -------------------------
    train_paths = sorted([Path(p) for p in glob.glob(args.train_glob)])
    if not train_paths:
        raise SystemExit(f"❌ train_glob matched nothing: {args.train_glob}")

    train_frames: List[pd.DataFrame] = []
    for p in train_paths:
        df = pd.read_csv(p)
        df = parse_year_month(df)
        df["plant"] = plant_from_train_filename(p)
        train_frames.append(df)

    train_all = pd.concat(train_frames, ignore_index=True)

    if train_prob_col not in train_all.columns:
        raise SystemExit(
            f"❌ train_prob_col '{train_prob_col}' not in train files. "
            f"columns={list(train_all.columns)}"
        )

    train_all = train_all[train_all["year_"] <= args.train_end_year].copy()

    pooled_train = pd.to_numeric(train_all[train_prob_col], errors="coerce").to_numpy(dtype=float)
    print(f"[info] train ref rows (finite probs): {int(np.isfinite(pooled_train).sum())}")

    # per-plant train map
    train_by_plant: Dict[str, np.ndarray] = {}
    for plant, g in train_all.groupby("plant"):
        train_by_plant[plant] = pd.to_numeric(g[train_prob_col], errors="coerce").to_numpy(dtype=float)


    # -------------------------
    # Load TEST (2025) from deployment/outputs/by_plant/*/inference_all_months.csv
    # -------------------------
    test_files = sorted(root.rglob("inference_all_months.csv"))
    if not test_files:
        raise SystemExit(f"❌ No inference_all_months.csv found under {root}")

    print(f"[info] found {len(test_files)} plant inference files under {root}")

    drift_rows = []
    monthly_rows = []

    all_test_probs = []

    for f in test_files:
        plant = f.parent.name  # e.g. osm_way_1079022886
        if not is_real_plant(plant):
        # treat as misc region; skip from by-plant outputs unless explicitly included
            if not getattr(args, "include_misc_regions", False):
                continue
        df = pd.read_csv(f)
        if prob_col not in df.columns:
            print(f"[warn] skip {f} (missing {prob_col})")
            continue

        df = parse_year_month(df)
        test = df[df["year_"] == args.test_year].copy()

        cur = pd.to_numeric(test[prob_col], errors="coerce").to_numpy(dtype=float)

        # reference: per-plant if available else pooled
        ref = train_by_plant.get(plant, pooled_train)

        ref_n = int(np.isfinite(ref).sum())
        cur_n = int(np.isfinite(cur).sum())

        if ref_n < 10:
            print(f"[WARN] {plant}: too few train ref samples ({ref_n}) → drift will be NaN")
        if cur_n < 10:
            print(f"[WARN] {plant}: too few test samples ({cur_n}) → drift will be NaN")

        drift_rows.append({
            "plant": plant,
            "train_rows": ref_n,
            "test_rows": cur_n,
            "psi": psi(ref, cur, n_bins=args.bins),
            "ks_D": ks_statistic(ref, cur),
            "ks_pvalue": try_ks_pvalue(ref, cur),
        })

        # monthly risk + topk events (test year only)
        if len(test) > 0:
            m = monthly_risk_table(test, prob_col, thr)
            m.insert(0, "plant", plant)
            monthly_rows.append(m)

            te = top_events(test, prob_col, args.topk)
            te.insert(0, "plant", plant)
            te.to_csv(outdir / f"top_events_{plant}_{args.test_year}.csv", index=False)

        if cur_n > 0:
            all_test_probs.append(cur)

    # write per-plant drift
    drift_df = pd.DataFrame(drift_rows).sort_values(["psi", "ks_D"], ascending=False)
    drift_df.to_csv(outdir / f"drift_2017-2024_vs_{args.test_year}_by_plant.csv", index=False)

    # pooled drift
    if len(all_test_probs) > 0:
        pooled_test = np.concatenate(all_test_probs)
        overall = {
            "train_rows": int(np.isfinite(pooled_train).sum()),
            "test_rows": int(np.isfinite(pooled_test).sum()),
            "psi": psi(pooled_train, pooled_test, n_bins=args.bins),
            "ks_D": ks_statistic(pooled_train, pooled_test),
            "ks_pvalue": try_ks_pvalue(pooled_train, pooled_test),
        }
        (outdir / f"drift_overall_2017-2024_vs_{args.test_year}.json").write_text(json.dumps(overall, indent=2))

    # monthly risk table
    if monthly_rows:
        monthly_df = pd.concat(monthly_rows, ignore_index=True)
        monthly_df.to_csv(outdir / f"monthly_risk_{args.test_year}_by_plant.csv", index=False)

    # README
    (outdir / "README.txt").write_text(
        f"""REDNET Generalization Eval
==========================
Train ref:  <= {args.train_end_year} from {args.train_glob}
Test year:  {args.test_year} from {args.by_plant_root}

train_prob_col: {train_prob_col}
test_prob_col:  {prob_col}
threshold:      {thr}

Outputs:
- drift_2017-2024_vs_{args.test_year}_by_plant.csv
- drift_overall_2017-2024_vs_{args.test_year}.json
- monthly_risk_{args.test_year}_by_plant.csv
- top_events_<plant>_{args.test_year}.csv
"""
    )

    print("✅ Done.")
    print("→ wrote:", outdir)
    print("→ threshold used:", thr)


if __name__ == "__main__":
    main()
