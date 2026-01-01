#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path
from datetime import date, datetime, timedelta

import joblib
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# small utils
# ──────────────────────────────────────────────────────────────────────────────
def run(cmd: list[str | Path]) -> None:
    cmd = [str(c) for c in cmd]
    print("\n▶", " ".join(cmd))
    subprocess.run(cmd, check=True)


def month_range(start: date, end: date):
    cur = date(start.year, start.month, 1)
    while cur <= end:
        yield cur
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)


def end_of_month(d: date) -> date:
    return (date(d.year, d.month, 28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)


def stable_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def ensure_chip_id(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    if "chip_id" in df.columns:
        return df

    # Prefer path-like columns if present
    for c in ["chip_path", "png_path", "tile_path", "path", "fname", "file"]:
        if c in df.columns:
            df = df.copy()
            df["chip_id"] = df[c].astype(str).map(lambda p: Path(p).stem)
            return df

    # Otherwise build from scene+bounds if possible
    candidates = [c for c in ["scene_id", "tile", "xmin", "ymin", "xmax", "ymax"] if c in df.columns]
    df = df.copy()
    if candidates:
        def keyrow(r):
            return "|".join([f"{c}={r[c]}" for c in candidates])
        df["chip_id"] = df.apply(lambda r: f"{tag}_{stable_hash(keyrow(r))}", axis=1)
        return df

    df["chip_id"] = [f"{tag}_{i:06d}" for i in range(len(df))]
    return df


def ensure_scene_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Your detector pipeline talks in terms of scene_id.
    To avoid confusion + keep compatibility, ensure scene_id exists
    (aliases to chip_id if needed).
    """
    df = df.copy()
    if "scene_id" not in df.columns and "chip_id" in df.columns:
        df["scene_id"] = df["chip_id"]
    return df


def ensure_time_columns(df: pd.DataFrame, month_dt: date) -> pd.DataFrame:
    df = df.copy()
    mk = f"{month_dt.year}-{month_dt.month:02d}"
    if "month_key" not in df.columns:
        df["month_key"] = mk

    # Your downstream expects `datetime`
    if "datetime" not in df.columns or df["datetime"].isna().all():
        df["datetime"] = f"{mk}-01T00:00:00Z"
    return df


def season_of_month(m: int) -> str:
    return ("winter", "winter", "spring", "spring", "spring",
            "summer", "summer", "summer", "autumn", "autumn", "autumn", "winter")[m - 1]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df.get("datetime"), errors="coerce", utc=True)

    df["year"] = dt.dt.year
    df["month_num"] = dt.dt.month

    # month cyclical encoding
    ang = 2 * np.pi * (df["month_num"].astype(float) / 12.0)
    df["month_sin"] = np.sin(ang)
    df["month_cos"] = np.cos(ang)

    df["season"] = df["month_num"].fillna(1).astype(int).clip(1, 12).map(season_of_month)
    for s in ["winter", "spring", "summer", "autumn"]:
        df[f"season_{s}"] = (df["season"] == s).astype(int)

    return df


def harmonize_modis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Your training fusion table used: kd490, chlor_a, nflh, sst
    Your append script often writes: kd490, chlor_a, flh, sst
    So make sure nflh exists for inference.
    """
    df = df.copy()

    # Normalize common case-sensitivity issues
    ren = {}
    if "Kd_490" in df.columns and "kd490" not in df.columns:
        ren["Kd_490"] = "kd490"
    df = df.rename(columns=ren)

    # flh <-> nflh
    if "nflh" not in df.columns and "flh" in df.columns:
        df["nflh"] = df["flh"]
    if "flh" not in df.columns and "nflh" in df.columns:
        df["flh"] = df["nflh"]

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep this lightweight: only features you likely used in fusion training.
    If your bundle expects extra engineered cols, we still create missing later.
    """
    df = df.copy()

    for c in ["kd490", "chlor_a", "nflh", "sst", "fai_mean", "ndwi_mean", "rednir_mean"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # logs (safe)
    if "kd490" in df.columns:
        df["log_kd490"] = np.log(np.clip(df["kd490"].to_numpy(dtype=float), 1e-9, None))
    if "chlor_a" in df.columns:
        df["log_chlor_a"] = np.log(np.clip(df["chlor_a"].to_numpy(dtype=float), 1e-9, None))
    if "nflh" in df.columns:
        df["log_nflh"] = np.log(np.clip(df["nflh"].to_numpy(dtype=float), 1e-9, None))

    # ratios / interactions (safe)
    if "chlor_a" in df.columns and "kd490" in df.columns:
        df["ratio_chl_kd"] = df["chlor_a"] / np.clip(df["kd490"], 1e-9, None)
    if "chlor_a" in df.columns and "nflh" in df.columns:
        df["chl_times_nflh"] = df["chlor_a"] * df["nflh"]
    if "nflh" in df.columns and "kd490" in df.columns:
        df["ratio_nflh_kd"] = df["nflh"] / np.clip(df["kd490"], 1e-9, None)

    return df


def safe_fill_missing_features(df: pd.DataFrame, feature_cols: list[str], bundle: dict) -> pd.DataFrame:
    df = df.copy()
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    fill_values = bundle.get("fill_values") or bundle.get("feature_fill_values")
    if isinstance(fill_values, dict):
        for c, v in fill_values.items():
            if c in df.columns:
                df[c] = df[c].fillna(v)

    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# MODIS download-on-demand (space-safe)
# ──────────────────────────────────────────────────────────────────────────────
DATE_RANGE = re.compile(r"(\d{8})[_\-](\d{8})")
DATE_ONE = re.compile(r"(\d{8})")


def file_mid_date_from_line(line: str) -> date | None:
    """
    Filelist line usually contains filename like:
      AQUA_MODIS.20190218_20190225.L3m.8D.CHL.chlor_a.4km.nc
    """
    m = DATE_RANGE.search(line)
    if m:
        d0 = datetime.strptime(m.group(1), "%Y%m%d").date()
        d1 = datetime.strptime(m.group(2), "%Y%m%d").date()
        # midpoint
        return d0 + (d1 - d0) // 2
    m = DATE_ONE.search(line)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    return None


def filter_filelist_for_window(filelist_path: Path, window_start: date, window_end: date, max_days: int) -> list[str]:
    """
    Keep only the entries whose mid-date is within [window_start-max_days, window_end+max_days].
    """
    lo = window_start - timedelta(days=max_days)
    hi = window_end + timedelta(days=max_days)

    out = []
    with open(filelist_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            md = file_mid_date_from_line(s)
            if md is None:
                continue
            if lo <= md <= hi:
                out.append(s)
    return out


def download_modis_subset(
    *,
    prod: str,
    filelist_path: Path,
    tmp_dir: Path,
    window_start: date,
    window_end: date,
    max_days: int,
    obpg_appkey: str,
    verbose: bool = False,
) -> None:
    """
    Writes a temporary filtered filelist, downloads into tmp_dir/prod,
    using your existing scripts/obdaac_download.py.
    """
    subset = filter_filelist_for_window(filelist_path, window_start, window_end, max_days=max_days)
    if not subset:
        if verbose:
            print(f"⚠️ MODIS {prod}: no matching lines in filelist for window {window_start}..{window_end}")
        return

    prod_dir = tmp_dir / prod
    prod_dir.mkdir(parents=True, exist_ok=True)

    subset_path = prod_dir / f"filelist_subset_{window_start}_{window_end}.txt"
    subset_path.write_text("\n".join(subset) + "\n")

    cmd = [
        "python", "scripts/download/obdaac_download.py",
        "--filelist", subset_path,
        "--odir", prod_dir,
        "--appkey", obpg_appkey,
    ]
    if verbose:
        cmd.append("-v")
    run(cmd)


def cleanup_tmp_nc(tmp_dir: Path) -> None:
    """
    Delete any .nc / .nc.gz / .bz2 etc under tmp_dir
    """
    if not tmp_dir.exists():
        return
    for p in tmp_dir.rglob("*"):
        if p.is_file() and (".nc" in p.name):
            try:
                p.unlink()
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# Detector + fusion utilities (run once at end)
# ──────────────────────────────────────────────────────────────────────────────
DROP_COLS = [
    "p_tab",
    "sst_clim_rm",
    "sst_anom",
    "sst_anom_z",
    "sst_anom_x_chlor_a",
    "sst_anom_x_nflh",
    "sst_anom_x_fai_mean",
    "sst_anom_x_kd490",
    "sst_anom_x_month_sin",
    "sst_anom_x_month_cos",
]


def move_detector_scores(det_out_dir: Path, root_dir: Path) -> None:
    """
    Moves: det_out_dir/<month>_detector_scores.csv  ->  root_dir/<month>/detector_scores.csv
    """
    if not det_out_dir.exists():
        print(f"⚠️ detector out_dir missing: {det_out_dir} (skipping move)")
        return

    moved = 0
    for f in sorted(det_out_dir.glob("*_detector_scores.csv")):
        m = f.name.replace("_detector_scores.csv", "")
        dest = root_dir / m
        if dest.is_dir():
            target = dest / "detector_scores.csv"
            print(f"→ moving {f} → {target}")
            try:
                f.replace(target)
                moved += 1
            except Exception as e:
                print(f"⚠️ failed move {f} → {target}: {e}")
        else:
            print(f"⚠ skipped {f} (no folder {dest})")

    if moved == 0:
        print("⚠️ no detector score files were moved (check naming / months).")


def merge_all_months(root_dir: Path) -> Path | None:
    """
    Rebuilds root_dir/inference_all_months.csv from root_dir/<month>/inference.csv
    """
    dfs = []
    for m in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        f = m / "inference.csv"
        if f.exists():
            df = pd.read_csv(f)
            df["month"] = m.name
            dfs.append(df)

    if not dfs:
        print("⚠️ merge_all_months: no per-month inference.csv found.")
        return None

    out = pd.concat(dfs, ignore_index=True)
    out_path = root_dir / "inference_all_months.csv"
    out.to_csv(out_path, index=False)
    print(f"✓ overwrote {out_path} ({len(out)} rows)")
    return out_path


def drop_unused_cols_per_month(root_dir: Path) -> None:
    """
    Drops only the known empty/unneeded columns from each month inference.csv (post-fusion).
    """
    for month in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        csv = month / "inference.csv"
        if not csv.exists():
            continue

        df = pd.read_csv(csv)
        before = len(df.columns)

        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

        after = len(df.columns)
        df.to_csv(csv, index=False)
        print(f"✓ {month.name}: dropped {before - after} columns")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser("End-to-end monthly HAB inference (space-safe, no retrain) + detectors + fusion")

    ap.add_argument("--aoi", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--model", required=True)

    # MODIS on-demand download
    ap.add_argument("--filelists_dir", default="filelists/8d", help="folder containing filelist_8d_*.txt")
    ap.add_argument("--max_days", type=int, default=30)
    ap.add_argument("--modis_products", nargs="+", default=["chlor_a", "Kd_490", "nflh", "sst"])
    ap.add_argument("--tmp_modis_root", default="data/l3/tmp_infer", help="temp folder used for downloads (auto-cleaned)")
    ap.add_argument("--delete_tmp_each", choices=["month", "year"], default="month",
                    help="delete temporary MODIS downloads after each month or after each year")

    ap.add_argument("--cloud", default="30")
    ap.add_argument("--per_window", default="1")
    ap.add_argument("--size", default="640")
    ap.add_argument("--stride", default="256")

    ap.add_argument("--debug_modis", action="store_true")

    # Detectors + fusion (run once at the end)
    ap.add_argument("--skip_detectors", action="store_true", help="do not run detectors/fusion at end")
    ap.add_argument("--detectors_script", default="deployment/src/inference/run_detectors_on_chips.py")
    ap.add_argument("--fusion_script", default="deployment/src/inference/rerun_fusion_with_detectors.py")
    ap.add_argument("--det_out_dir", default="detector_scores_by_month",
                    help="where run_detectors_on_chips.py writes *_detector_scores.csv (relative to CWD)")
    ap.add_argument("--frcnn_r50", default="runs/detect/frcnn_resnet50/best_resnet.pt")
    ap.add_argument("--frcnn_mb", default="runs/detect/frcnn_mobilenet/best.pt")
    ap.add_argument("--ssd_mb", default="runs/detect/ssd_mobilenet/best_ssd.pt")

    args = ap.parse_args()

    obpg_appkey = os.environ.get("OBPG_APPKEY")
    if not obpg_appkey:
        print("❌ OBPG_APPKEY is not set. Do: export OBPG_APPKEY=YOUR_KEY")
        sys.exit(1)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tmp_modis_root = Path(args.tmp_modis_root)
    tmp_modis_root.mkdir(parents=True, exist_ok=True)

    # Load model bundle
    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_cols = bundle["features"]
    calibrator = bundle.get("calibrator")

    # Resolve filelists per product
    filelists_dir = Path(args.filelists_dir)
    filelist_map = {
        "chlor_a": filelists_dir / "filelist_8d_chlor_a_filtered.txt",
        "Kd_490":  filelists_dir / "filelist_8d_Kd_490_filtered.txt",
        "nflh":    filelists_dir / "filelist_8d_nflh_filtered.txt",
        "sst":     filelists_dir / "filelist_8d_sst.txt",
    }

    for p in args.modis_products:
        if p not in filelist_map:
            print(f"❌ unknown product: {p} (expected one of {list(filelist_map.keys())})")
            sys.exit(1)
        if not filelist_map[p].exists():
            print(f"❌ missing filelist for {p}: {filelist_map[p]}")
            sys.exit(1)

    all_months = []
    current_year = None

    for m in month_range(start, end):
        tag = f"{m.year}-{m.month:02d}"
        print(f"\n==================== {tag} ====================")

        # If we are doing delete per year, detect year changes
        if current_year is None:
            current_year = m.year
        if args.delete_tmp_each == "year" and m.year != current_year:
            # cleanup previous year's temp downloads
            if args.debug_modis:
                print(f"🧹 cleaning temp MODIS downloads for year {current_year}")
            cleanup_tmp_nc(tmp_modis_root / str(current_year))
            current_year = m.year

        month_dir = out_root / tag
        chips_dir = month_dir / "chips"
        month_dir.mkdir(parents=True, exist_ok=True)
        chips_dir.mkdir(parents=True, exist_ok=True)

        m_start = date(m.year, m.month, 1)
        m_end = end_of_month(m)

        # 1) Sentinel-2 chipping
        run([
            "python", "scripts/download/s2_chip_8day.py",
            "--aoi", args.aoi,
            "--start", m_start.isoformat(),
            "--end", m_end.isoformat(),
            "--cloud", str(args.cloud),
            "--per_window", str(args.per_window),
            "--size", str(args.size),
            "--stride", str(args.stride),
            "--out", chips_dir
        ])

        index_csv = chips_dir / "index.csv"
        if not index_csv.exists():
            print("⚠️ No index.csv produced, skipping month.")
            continue

        # If 0 tiles, skip early
        try:
            idx = pd.read_csv(index_csv)
            if len(idx) == 0:
                print("⚠️ 0 chips for this month, skipping.")
                continue
        except Exception:
            pass

        # 2) Chip indices
        run(["python", "scripts/download/s2_compute_chip_indices.py", "--folder", chips_dir])
        chip_idx_csv = chips_dir / "chip_indices.csv"
        if not chip_idx_csv.exists():
            print("⚠️ chip_indices.csv missing, skipping.")
            continue

        df = pd.read_csv(chip_idx_csv)
        if len(df) == 0:
            print("⚠️ chip_indices.csv empty, skipping.")
            continue

        df = ensure_chip_id(df, tag=tag)
        df = ensure_scene_id(df)              # <-- keep detector naming happy
        df = ensure_time_columns(df, month_dt=m)
        df.to_csv(chip_idx_csv, index=False)

        # 3) MODIS per-month download → append → delete
        # Put temp downloads under tmp_modis_root/<year>/<product>/
        tmp_year_dir = tmp_modis_root / str(m.year)
        tmp_year_dir.mkdir(parents=True, exist_ok=True)

        for prod in args.modis_products:
            download_modis_subset(
                prod=prod,
                filelist_path=filelist_map[prod],
                tmp_dir=tmp_year_dir,
                window_start=m_start,
                window_end=m_end,
                max_days=args.max_days,
                obpg_appkey=obpg_appkey,
                verbose=args.debug_modis,
            )

            # append for this product only (so it can work even if other prods had 0 downloads)
            run([
                "python", "scripts/HAB/preparation/append_modis_features_8d.py",
                "--chips_csv_glob", str(chip_idx_csv),
                "--modis_root", str(tmp_year_dir / prod),
                "--max_days", str(args.max_days),
                "--products", prod,
            ])

            # delete product downloads immediately if month-level cleanup
            if args.delete_tmp_each == "month":
                cleanup_tmp_nc(tmp_year_dir / prod)

        # reload after append(s)
        df = pd.read_csv(chip_idx_csv)
        df = ensure_chip_id(df, tag=tag)
        df = ensure_scene_id(df)
        df = ensure_time_columns(df, month_dt=m)
        df = harmonize_modis_columns(df)
        df = add_time_features(df)
        df = add_engineered_features(df)

        # 4) Prepare model matrix
        df = safe_fill_missing_features(df, feature_cols=feature_cols, bundle=bundle)

        X = df[feature_cols]
        try:
            probs = model.predict_proba(X)[:, 1]
        except Exception as e:
            print("❌ model.predict_proba failed (feature mismatch / dtype issue).")
            print("   error:", e)
            print("   dtypes sample:\n", X.dtypes.head(50))
            sys.exit(1)

        if calibrator is not None:
            probs = calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]

        df["hab_prob"] = probs
        df["month"] = tag

        out_csv = month_dir / "inference.csv"
        df.to_csv(out_csv, index=False)
        print(f"✅ wrote {out_csv} (rows={len(df)})")

        all_months.append(df)

    # if year-level cleanup, clean last year too
    if args.delete_tmp_each == "year" and current_year is not None:
        if args.debug_modis:
            print(f"🧹 cleaning temp MODIS downloads for year {current_year}")
        cleanup_tmp_nc(tmp_modis_root / str(current_year))

    if not all_months:
        print("\n⚠️ No outputs produced.")
        sys.exit(0)

    # Baseline merge (pre-detectors)
    merged = pd.concat(all_months, ignore_index=True)
    merged_csv = out_root / "inference_all_months.csv"
    merged.to_csv(merged_csv, index=False)
    print(f"\n✅ wrote {merged_csv}")

    # ──────────────────────────────────────────────────────────────────────────
    # Run detectors ONCE, then fuse ONCE, then clean + merge ONCE
    # ──────────────────────────────────────────────────────────────────────────
    if args.skip_detectors:
        return

    det_out_dir = Path(args.det_out_dir)

    # 1) run detectors on all months under out_root
    run([
        "python", args.detectors_script,
        "--root_dir", out_root,
        "--out_dir", det_out_dir,
        "--frcnn_r50", args.frcnn_r50,
        "--frcnn_mb", args.frcnn_mb,
        "--ssd_mb", args.ssd_mb,
    ])

    # 2) move detector scores into each month folder
    move_detector_scores(det_out_dir=det_out_dir, root_dir=out_root)

    # 3) rerun fusion with detectors (updates each month inference.csv)
    run([
        "python", args.fusion_script,
        "--root_dir", out_root,
        "--model", args.model,
    ])

    # 4) drop only the empty/unused columns you listed
    drop_unused_cols_per_month(out_root)

    # 5) rebuild merged inference_all_months.csv from month folders (post-fusion)
    merge_all_months(out_root)


if __name__ == "__main__":
    main()
