#!/usr/bin/env python3
import argparse, glob, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def add_time_fields(df):
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df = df.copy()
        df["dt"] = dt
    return df

def maybe_add_centroids(df):
    have_lonlat = {"lon","lat"} <= set(df.columns)
    have_bounds = {"xmin","xmax","ymin","ymax"} <= set(df.columns)
    if have_lonlat:
        return df
    if not have_bounds:
        return df
    # centroids from bounds (these are in the image CRS; for quick QC it's fine as a relative map)
    lon = (to_num(df["xmin"]) + to_num(df["xmax"])) * 0.5
    lat = (to_num(df["ymin"]) + to_num(df["ymax"])) * 0.5
    out = df.copy()
    out["lon"] = lon
    out["lat"] = lat
    return out

def load_all(glob_pat):
    files = sorted(glob.glob(glob_pat))
    if not files:
        raise SystemExit(f"No CSVs matched: {glob_pat}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["__source__"] = Path(f).name
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    # enforce numeric
    for col in ["chlor_a","flh","nflh","kd490","Kd_490","valid_px"]:
        if col in df.columns:
            df[col] = to_num(df[col])
    # normalize FLH column name
    if "flh" not in df.columns and "nflh" in df.columns:
        df = df.rename(columns={"nflh":"flh"})
    # prefer kd490 (lowercase)
    if "kd490" not in df.columns and "Kd_490" in df.columns:
        df = df.rename(columns={"Kd_490":"kd490"})
    # label
    if "hab_label" not in df.columns:
        raise SystemExit("No 'hab_label' column in inputs. Did you run make_hab_labels.py?")
    df["hab_label"] = (df["hab_label"].astype(float) > 0.5).astype(int)
    df = add_time_fields(df)
    df = maybe_add_centroids(df)
    return df, files

def save_or_show(outdir, name):
    """Return a path to save plot; caller should plt.savefig(path, ...) then plt.close()."""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    return str(Path(outdir) / f"{name}.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_glob", required=True,
                    help="e.g. 'data/aerial_summer_2019/*_hab.csv' or 'data/aerial_*_2019/chip_indices_clean_hab.csv'")
    ap.add_argument("--outdir", default="qc_report",
                    help="folder to write PNG figures")
    ap.add_argument("--title", default=None, help="custom title prefix")
    ap.add_argument("--min_valid", type=int, default=0,
                    help="optional: drop rows with valid_px < min_valid")
    args = ap.parse_args()

    df, files = load_all(args.csv_glob)
    title_prefix = args.title or f"HAB QC ({len(files)} file{'s' if len(files)!=1 else ''})"

    # optional clean-ups
    if "valid_px" in df.columns and args.min_valid > 0:
        df = df[df["valid_px"] >= args.min_valid].copy()

    # --- Summary ---
    n_total = len(df)
    n_pos = int(df["hab_label"].sum())
    print(f"Loaded {n_total} rows from {len(files)} file(s). HAB positives = {n_pos} ({n_pos/n_total*100:.1f}%).")
    def safe_mean(x): return float(np.nanmean(x)) if len(x) else np.nan

    for col in ["chlor_a","flh","kd490"]:
        if col in df.columns:
            g = df.groupby("hab_label")[col].agg(["count","mean","median","std"])
            print(f"\n{col} by label:\n{g}")

    # --- Plot 1: chlor_a vs FLH (colored by label) ---
    if "chlor_a" in df.columns and "flh" in df.columns:
        plt.figure(figsize=(7.2,6))
        pos = df["hab_label"] == 1
        plt.scatter(df.loc[~pos, "flh"], df.loc[~pos, "chlor_a"], s=28, alpha=0.6, label="non-HAB", edgecolor="none")
        plt.scatter(df.loc[pos, "flh"],  df.loc[pos, "chlor_a"],  s=50, alpha=0.95, label="HAB", marker="o")
        plt.xlabel("FLH")
        plt.ylabel("Chlorophyll-a (mg m$^{-3}$)")
        plt.title(f"{title_prefix}: chlor_a vs FLH")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(save_or_show(args.outdir, "01_scatter_chla_vs_flh"), dpi=160)
        plt.close()

    # --- Plot 2: chlor_a histogram split by label ---
    if "chlor_a" in df.columns:
        plt.figure(figsize=(7.2,6))
        nb = 20
        plt.hist(df.loc[df["hab_label"]==0, "chlor_a"].dropna(), bins=nb, alpha=0.65, label="non-HAB")
        plt.hist(df.loc[df["hab_label"]==1, "chlor_a"].dropna(), bins=nb, alpha=0.75, label="HAB")
        plt.xlabel("Chlorophyll-a (mg m$^{-3}$)")
        plt.ylabel("Count")
        plt.title(f"{title_prefix}: chlor_a distribution by label")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(save_or_show(args.outdir, "02_hist_chla_by_label"), dpi=160)
        plt.close()

    # --- Plot 3: chlor_a vs kd490 (if available) ---
    if "chlor_a" in df.columns and "kd490" in df.columns:
        plt.figure(figsize=(7.2,6))
        pos = df["hab_label"] == 1
        plt.scatter(df.loc[~pos,"kd490"], df.loc[~pos,"chlor_a"], s=28, alpha=0.6, label="non-HAB", edgecolor="none")
        plt.scatter(df.loc[pos,"kd490"],  df.loc[pos,"chlor_a"],  s=50, alpha=0.95, label="HAB")
        plt.xlabel("Kd490 (m$^{-1}$)")
        plt.ylabel("Chlorophyll-a (mg m$^{-3}$)")
        plt.title(f"{title_prefix}: chlor_a vs Kd490")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(save_or_show(args.outdir, "03_scatter_chla_vs_kd490"), dpi=160)
        plt.close()

    # --- Plot 4: time series (if dt present) ---
    if "dt" in df.columns and "chlor_a" in df.columns:
        plt.figure(figsize=(9,4.8))
        pos = df["hab_label"] == 1
        # jitter x to avoid overlaps when same time
        x_all = df["dt"].astype("int64") // 10**9
        x_all = x_all + np.random.normal(0, 12*3600, size=len(x_all))  # ±12h jitter
        plt.scatter(pd.to_datetime(x_all, unit="s"), df["chlor_a"], s=26, alpha=0.7, label="non-HAB", c="C0")
        plt.scatter(pd.to_datetime(x_all[pos], unit="s"), df.loc[pos,"chlor_a"], s=48, alpha=0.95, label="HAB", c="C3")
        plt.ylabel("Chlorophyll-a (mg m$^{-3}$)")
        plt.title(f"{title_prefix}: time series (chlor_a)")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(save_or_show(args.outdir, "04_timeseries_chla"), dpi=160)
        plt.close()

    # --- Plot 5 (optional): spatial scatter if lon/lat exist ---
    if {"lon","lat"} <= set(df.columns):
        plt.figure(figsize=(7,6))
        pos = df["hab_label"] == 1
        plt.scatter(df.loc[~pos,"lon"], df.loc[~pos,"lat"], s=6, alpha=0.4, label="non-HAB")
        plt.scatter(df.loc[pos,"lon"],  df.loc[pos,"lat"],  s=18, alpha=0.95, label="HAB")
        plt.xlabel("lon (approx)")
        plt.ylabel("lat (approx)")
        plt.title(f"{title_prefix}: spatial distribution")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(save_or_show(args.outdir, "05_map_scatter"), dpi=160)
        plt.close()

    print(f"\nQC figures written to: {args.outdir}")
    print("Tip: Red/HAB points should sit at higher chlor_a and/or FLH;")
    print("     if they don’t, revisit thresholds in make_hab_labels.py (q, q_floor, kd_floor).")

if __name__ == "__main__":
    main()
