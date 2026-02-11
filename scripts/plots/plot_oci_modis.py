#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
import warnings
from pathlib import Path
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ─────────────────────────────────────────────────────────────
# USER CONFIG
# ─────────────────────────────────────────────────────────────

YEAR  = 2024
MONTH = 8
AGG_MODE = "year"  # "month" or "year"

# Your big geojson bounds (kept)
LON_MIN, LON_MAX = 51.5, 60.8
LAT_MIN, LAT_MAX = 15.5, 26.5

# Slight zoom-in target to look like the paper figure
ZOOM_LON_MIN, ZOOM_LON_MAX = 49.5, 60.8
ZOOM_LAT_MIN, ZOOM_LAT_MAX = 16.0, 27.0

FILELIST_CHL   = Path("data/filelists/8d/filelist_8d_chlor_a_filtered.txt")
FILELIST_SST = Path("data/filelists/8d/filelist_8d_sst.txt")  
TMP_MODIS_ROOT = Path("data/l3/tmp_oci")

OUT_DIR = Path("runs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CMAP = "turbo"
MODIS_VMAX = 4.5
GRID_RES_DEG = 0.01  # ~1 km

# ─────────────────────────────────────────────────────────────
# SMALL UTILS
# ─────────────────────────────────────────────────────────────

DATE_RANGE   = re.compile(r"(\d{8})[_\-](\d{8})")
DATE_ONE     = re.compile(r"(\d{8})")
MGRS_TILE_RE = re.compile(r"_T(\d{2}[A-Z][A-Z]{2})_")  # e.g. 39QXG

def run(cmd):
    cmd = [str(c) for c in cmd]
    print("▶", " ".join(cmd))
    subprocess.run(cmd, check=True)

def file_mid_date_from_line(line: str) -> date | None:
    m = DATE_RANGE.search(line)
    if m:
        d0 = datetime.strptime(m.group(1), "%Y%m%d").date()
        d1 = datetime.strptime(m.group(2), "%Y%m%d").date()
        return d0 + (d1 - d0) // 2
    m = DATE_ONE.search(line)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    return None

def filter_filelist(filelist: Path, start: date, end: date):
    out = []
    with open(filelist) as f:
        for line in f:
            md = file_mid_date_from_line(line)
            if md and start <= md <= end:
                out.append(line.strip())
    return out

def _slice_in_coord_order(coord_values: xr.DataArray, vmin: float, vmax: float) -> slice:
    if coord_values.size < 2:
        return slice(vmin, vmax)
    ascending = bool(coord_values.values[0] < coord_values.values[-1])
    return slice(vmin, vmax) if ascending else slice(vmax, vmin)

def period_start_end() -> tuple[date, date, str]:
    if AGG_MODE == "year":
        return date(YEAR, 1, 1), date(YEAR, 12, 31), f"{YEAR}"
    start = date(YEAR, MONTH, 1)
    end   = (start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    return start, end, f"{YEAR}_{MONTH:02d}"

def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _utm_epsg_from_mgrs_tile(mgrs_tile: str) -> int:
    """
    mgrs_tile like "39QXG"
      - zone = first 2 digits
      - band = third char; >= 'N' is Northern hemisphere
    """
    zone = int(mgrs_tile[:2])
    band = mgrs_tile[2]
    north = (band >= "N")
    return (32600 + zone) if north else (32700 + zone)

def _norm_tile_name(x) -> str:
    if pd.isna(x):
        return ""
    return Path(str(x)).name

def _pick_datetime_col(df: pd.DataFrame) -> str | None:
    candidates = [
        "datetime", "Datetime", "DATE_TIME",
        "timestamp", "Timestamp", "ts",
        "acquisition_time", "acq_time",
        "time", "Time",
        "date", "Date",
        "sensing_time", "sensing_datetime",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = str(c).lower()
        if "datetime" in lc or "timestamp" in lc or (("date" in lc) and ("update" not in lc)) or ("time" in lc):
            return c
    return None

# ─────────────────────────────────────────────────────────────
# CRS FALLBACK FIX (no loops, deterministic)
# ─────────────────────────────────────────────────────────────

def mgrs_tile_center_lonlat(mgrs_tile: str) -> tuple[float, float]:
    """
    Return (lon, lat) for the center of a 100km MGRS tile.
    Uses 'mgrs' (pip install mgrs).
    """
    try:
        from mgrs import MGRS
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'mgrs'. Install it with:\n"
            "  pip install mgrs\n"
            "Then re-run."
        ) from e

    m = MGRS()
    mgrs_center = f"{mgrs_tile}5000050000"  # 100km tile center
    lat, lon = m.toLatLon(mgrs_center)
    return float(lon), float(lat)

def _xy_looks_broken(df: pd.DataFrame) -> bool:
    """
    Detect the exact failure you hit:
    - x/y essentially constant across the dataset, or
    - too few unique coordinate pairs, or
    - mostly NaN
    """
    xs = df["x_center"].to_numpy(dtype=float)
    ys = df["y_center"].to_numpy(dtype=float)

    if np.isfinite(xs).sum() < 10 or np.isfinite(ys).sum() < 10:
        return True

    if np.nanstd(xs) < 1e-6 and np.nanstd(ys) < 1e-6:
        return True

    pairs = np.vstack([xs, ys]).T
    ok = np.isfinite(xs) & np.isfinite(ys)
    if ok.sum() < 10:
        return True
    uniq = np.unique(pairs[ok], axis=0)
    if len(uniq) < 5:
        return True

    return False

# ─────────────────────────────────────────────────────────────
# AUTO-CRS HELPERS (kept, but now with safe fallback)
# ─────────────────────────────────────────────────────────────

def _looks_like_lonlat(x: np.ndarray, y: np.ndarray) -> bool:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return False
    xs = x[m]; ys = y[m]
    return (xs.min() >= -180 and xs.max() <= 180 and ys.min() >= -90 and ys.max() <= 90)

def _count_in_aoi(lon: np.ndarray, lat: np.ndarray) -> int:
    m = np.isfinite(lon) & np.isfinite(lat)
    if m.sum() == 0:
        return 0
    lon = lon[m]; lat = lat[m]
    ok = (
        (lon >= ZOOM_LON_MIN) & (lon <= ZOOM_LON_MAX) &
        (lat >= ZOOM_LAT_MIN) & (lat <= ZOOM_LAT_MAX)
    )
    return int(ok.sum())

def _auto_project_to_lonlat(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Try multiple CRS hypotheses for (x_center,y_center) and pick the one that
    yields the most points inside the zoom AOI.
    Returns lon, lat, chosen_label.
    """
    xs = df["x_center"].to_numpy(dtype=float)
    ys = df["y_center"].to_numpy(dtype=float)

    # Case 1: already lon/lat degrees
    if _looks_like_lonlat(xs, ys):
        return xs, ys, "EPSG:4326 (already lon/lat)"

    try:
        from pyproj import Transformer
    except Exception as e:
        raise RuntimeError("Missing dependency 'pyproj'. Install it with: pip install pyproj") from e

    candidates_global = [
        ("EPSG:3857", 3857),     # Web Mercator
        ("EPSG:32640", 32640),   # UTM 40N
        ("EPSG:32641", 32641),   # UTM 41N
        ("EPSG:32740", 32740),   # UTM 40S
        ("EPSG:32741", 32741),   # UTM 41S
    ]

    best_label = "none"
    best_count = -1
    best_lon = None
    best_lat = None

    for label, epsg in candidates_global:
        tr = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(xs, ys)
        lon = np.asarray(lon); lat = np.asarray(lat)
        c = _count_in_aoi(lon, lat)
        if c > best_count:
            best_label, best_count, best_lon, best_lat = label, c, lon, lat

    # Try per-row inferred UTM from MGRS
    lons = np.full(len(df), np.nan, dtype=float)
    lats = np.full(len(df), np.nan, dtype=float)
    epsg_series = df["mgrs_tile"].map(_utm_epsg_from_mgrs_tile)

    for epsg, sub_idx in df.groupby(epsg_series).groups.items():
        sub_x = df.loc[sub_idx, "x_center"].to_numpy(dtype=float)
        sub_y = df.loc[sub_idx, "y_center"].to_numpy(dtype=float)
        tr = Transformer.from_crs(f"EPSG:{int(epsg)}", "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(sub_x, sub_y)
        lons[sub_idx] = lon
        lats[sub_idx] = lat

    c_utm = _count_in_aoi(lons, lats)
    if c_utm > best_count:
        best_label, best_count, best_lon, best_lat = "UTM inferred from MGRS", c_utm, lons, lats

    if best_count <= 0 or best_lon is None or best_lat is None:
        raise RuntimeError(
            "Could not infer CRS for index.csv bbox centers.\n"
            f"x_center range: [{np.nanmin(xs):.3f}, {np.nanmax(xs):.3f}]\n"
            f"y_center range: [{np.nanmin(ys):.3f}, {np.nanmax(ys):.3f}]\n"
            "Tried EPSG:4326 (identity), 3857, 32640/32641, 32740/32741, and inferred UTM from MGRS.\n"
            "This strongly suggests your x/y are pixel coordinates (or another CRS)."
        )

    return best_lon, best_lat, best_label

# ─────────────────────────────────────────────────────────────
# MODIS: DOWNLOAD + PERIOD OCI COMPOSITE
# ─────────────────────────────────────────────────────────────

def load_modis_var_period(start: date, end: date, tag: str, var_name: str, filelist: Path) -> xr.DataArray:
    obpg_key = os.environ.get("OBPG_APPKEY")
    if not obpg_key:
        raise RuntimeError("OBPG_APPKEY not set")

    tmp_dir = TMP_MODIS_ROOT / tag / var_name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    subset = filter_filelist(filelist, start, end)
    if not subset:
        raise RuntimeError(f"No MODIS files found for {var_name} in period {start} → {end}")

    subset_file = tmp_dir / "filelist_subset.txt"
    subset_file.write_text("\n".join(subset) + "\n")

    run([
        "python", "scripts/download/obdaac_download.py",
        "--filelist", subset_file,
        "--odir", tmp_dir,
        "--appkey", obpg_key
    ])

    nc_files = sorted(tmp_dir.glob("*.nc"))
    if not nc_files:
        raise RuntimeError(f"No MODIS NetCDF files downloaded for {var_name}")

    stack = []
    da_template = None

    for f in nc_files:
        ds = xr.open_dataset(f)

        # try exact name first, then common alternates
        if var_name in ds:
            da = ds[var_name]
        else:
            # common SST alternates in OBPG-style files
            candidates = ["sst", "sea_surface_temperature", "SST", "sst4"]
            hit = next((c for c in candidates if c in ds), None)
            if hit is None:
                continue
            da = ds[hit]

        # basic cleaning (keep NaNs)
        da = da.where(np.isfinite(da))

        lon_sl = _slice_in_coord_order(da["lon"], ZOOM_LON_MIN, ZOOM_LON_MAX)
        lat_sl = _slice_in_coord_order(da["lat"], ZOOM_LAT_MIN, ZOOM_LAT_MAX)
        da = da.sel(lon=lon_sl, lat=lat_sl)

        if da_template is None:
            da_template = da

        if da.sizes.get("lat", 0) == 0 or da.sizes.get("lon", 0) == 0:
            continue

        stack.append(da.values)

    if not stack or da_template is None:
        raise RuntimeError(f"No valid {var_name} data in MODIS files after AOI slicing.")

    stack = np.stack(stack, axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(stack, axis=0)

    out = xr.DataArray(
        med,
        coords={"lat": da_template.lat, "lon": da_template.lon},
        dims=("lat", "lon"),
        name=var_name
    )
    return out


# ─────────────────────────────────────────────────────────────
# SENTINEL-2: index.csv bbox centers -> (AUTO CRS OR MGRS FALLBACK) -> lon/lat -> grid
# ─────────────────────────────────────────────────────────────

def _load_one_training_folder(folder: Path) -> pd.DataFrame:
    chips = folder / "chip_indices_clean_hab.csv"
    idx   = folder / "index.csv"
    if not chips.exists() or not idx.exists():
        return pd.DataFrame()

    dfc = pd.read_csv(chips)
    dfi = pd.read_csv(idx)

    if "tile" not in dfc.columns or "tile" not in dfi.columns:
        return pd.DataFrame()

    for c in ["xmin", "ymin", "xmax", "ymax"]:
        if c not in dfi.columns:
            return pd.DataFrame()

    dfc = dfc.copy()
    dfi = dfi.copy()

    # normalize tile join keys (path-safe)
    dfc["tile_key"] = dfc["tile"].map(_norm_tile_name)
    dfi["tile_key"] = dfi["tile"].map(_norm_tile_name)

    dfi = _safe_numeric(dfi, ["xmin", "ymin", "xmax", "ymax"])
    dfi["x_center"] = (dfi["xmin"] + dfi["xmax"]) / 2.0
    dfi["y_center"] = (dfi["ymin"] + dfi["ymax"]) / 2.0

    idx_time_col = _pick_datetime_col(dfi)
    cols = ["tile_key", "scene_id", "x_center", "y_center"]
    if idx_time_col is not None:
        cols.insert(2, idx_time_col)

    merged = dfc.merge(
        dfi[cols],
        on="tile_key",
        how="left",
        suffixes=("", "_idx")
    )

    # Ensure a unified "datetime" column
    chip_time_col = _pick_datetime_col(dfc)
    if "datetime" not in merged.columns:
        if chip_time_col and chip_time_col in merged.columns:
            merged = merged.rename(columns={chip_time_col: "datetime"})
        elif idx_time_col and idx_time_col in merged.columns:
            merged = merged.rename(columns={idx_time_col: "datetime"})
        else:
            return pd.DataFrame()

    if "datetime_idx" in merged.columns:
        merged["datetime"] = merged["datetime"].fillna(merged["datetime_idx"])

    return merged

def load_s2_grids_period(start: date, end: date) -> dict[str, xr.DataArray]:
    folders = sorted(Path("training").glob("*"))
    dfs = []
    for f in folders:
        if f.is_dir():
            d = _load_one_training_folder(f)
            if not d.empty:
                dfs.append(d)

    if not dfs:
        raise RuntimeError(
            "No training folders produced usable merged S2 tables.\n"
            "Likely: your index.csv has no time column AND your chip csv has no time column."
        )

    df = pd.concat(dfs, ignore_index=True)

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"])

    start_ts = pd.Timestamp(start.isoformat(), tz="UTC")
    end_ts   = pd.Timestamp((end + timedelta(days=1)).isoformat(), tz="UTC")
    df = df[(df["datetime"] >= start_ts) & (df["datetime"] < end_ts)]

    needed = ["scene_id", "valid_px", "chlor_a", "x_center", "y_center"]
    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column in merged S2 table: {c}")

    df = _safe_numeric(df, ["valid_px", "chlor_a", "kd490", "flh", "nflh", "sst", "x_center", "y_center"])
    df = df[(df["valid_px"] > 0) & (df["chlor_a"] > 0)]
    df = df.dropna(subset=["x_center", "y_center"])
    if df.empty:
        raise RuntimeError("No Sentinel-2 rows after filtering (valid_px/chlor_a/bbox centers).")

    # Needed for UTM-from-MGRS candidate and for MGRS fallback
    df["mgrs_tile"] = df["scene_id"].astype(str).str.extract(MGRS_TILE_RE, expand=False)
    df = df.dropna(subset=["mgrs_tile"])
    if df.empty:
        raise RuntimeError("Could not extract MGRS tile from scene_id.")

    df = df.reset_index(drop=True)

    # ✅ CRITICAL FIX: if x/y look broken (constant / pixel / bad merge), DO NOT CRS-GUESS.
    # Use MGRS tile-center lon/lat fallback deterministically.
    if _xy_looks_broken(df):
        tile_ll = {t: mgrs_tile_center_lonlat(t) for t in df["mgrs_tile"].unique()}
        df["lon"] = df["mgrs_tile"].map(lambda t: tile_ll[t][0])
        df["lat"] = df["mgrs_tile"].map(lambda t: tile_ll[t][1])
        chosen = "MGRS tile-center fallback (x/y look broken)"
    else:
        try:
            lon, lat, chosen = _auto_project_to_lonlat(df)
            df["lon"] = lon
            df["lat"] = lat
        except RuntimeError as e:
            # if CRS inference fails, fallback once (no loops)
            print(f"[S2] CRS inference failed, falling back to MGRS tile centers.\n  Reason: {e}")
            tile_ll = {t: mgrs_tile_center_lonlat(t) for t in df["mgrs_tile"].unique()}
            df["lon"] = df["mgrs_tile"].map(lambda t: tile_ll[t][0])
            df["lat"] = df["mgrs_tile"].map(lambda t: tile_ll[t][1])
            chosen = "MGRS tile-center fallback (CRS inference failed)"

    df = df.dropna(subset=["lon", "lat"])

    # Zoom AOI filter
    df = df[
        (df["lon"] >= ZOOM_LON_MIN) & (df["lon"] <= ZOOM_LON_MAX) &
        (df["lat"] >= ZOOM_LAT_MIN) & (df["lat"] <= ZOOM_LAT_MAX)
    ]
    if df.empty:
        raise RuntimeError(
            "S2 became empty after lon/lat conversion + zoom AOI filter.\n"
            f"Method: {chosen}\n"
            "This usually means AOI bounds don't match your S2 region, or your scene_id tiles are elsewhere."
        )

    print(f"[S2] Method: {chosen} | points in AOI: {len(df)}")

    # Build grid bins
    lon_edges = np.arange(ZOOM_LON_MIN, ZOOM_LON_MAX + GRID_RES_DEG, GRID_RES_DEG)
    lat_edges = np.arange(ZOOM_LAT_MIN, ZOOM_LAT_MAX + GRID_RES_DEG, GRID_RES_DEG)
    lon_centers = lon_edges[:-1]
    lat_centers = lat_edges[:-1]

    df["lon_bin"] = pd.cut(df["lon"], lon_edges, labels=lon_centers, include_lowest=True, right=False)
    df["lat_bin"] = pd.cut(df["lat"], lat_edges, labels=lat_centers, include_lowest=True, right=False)
    df = df.dropna(subset=["lon_bin", "lat_bin"])
    if df.empty:
        raise RuntimeError("All S2 points dropped during binning. Increase GRID_RES_DEG or widen zoom bounds.")

    vars_to_try = ["chlor_a", "kd490", "nflh", "flh", "sst"]
    vars_present = [v for v in vars_to_try if v in df.columns and df[v].notna().any()]
    if "chlor_a" not in vars_present:
        raise RuntimeError("No usable chlor_a values after merging/binnning.")

    grids: dict[str, xr.DataArray] = {}
    for v in vars_present:
        g = (
            df.groupby(["lat_bin", "lon_bin"], observed=True)[v]
              .median()
              .reset_index()
        )
        g["lon"] = g["lon_bin"].astype(float)
        g["lat"] = g["lat_bin"].astype(float)
        da = g.set_index(["lat", "lon"]).to_xarray()[v]

        # Force full rectangular grid so pcolormesh never errors
        da = da.reindex(lat=lat_centers, lon=lon_centers)

        # Light smoothing
        da = da.rolling(lat=2, lon=2, center=True, min_periods=1).mean()
        grids[v] = da

    return grids

# ─────────────────────────────────────────────────────────────
# PLOT (MODIS vs S2 chlor_a)
# ─────────────────────────────────────────────────────────────

def plot_modis_chl_vs_modis_sst(chl_da: xr.DataArray, sst_da: xr.DataArray, tag: str):
    out_fig = OUT_DIR / f"oci_modis_chl_vs_sst_{tag}.png"

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # left panel settings (chlor_a)
    chl_levels = np.arange(0.5, 4.6, 0.5)

    # right panel settings (sst) — you can tune these
    # If your SST is in Kelvin, convert to C
    sst = sst_da.copy()
    if np.nanmedian(sst.values) > 100:  # very safe Kelvin heuristic
        sst = sst - 273.15
        sst_name = "SST (°C)"
    else:
        sst_name = "SST (°C)"  # assume already °C

    sst_vmin = float(np.nanpercentile(sst.values, 5))
    sst_vmax = float(np.nanpercentile(sst.values, 95))

    for ax in axes:
        ax.set_extent([ZOOM_LON_MIN, ZOOM_LON_MAX, ZOOM_LAT_MIN, ZOOM_LAT_MAX])
        ax.coastlines(resolution="10m")
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=3)
        gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

    # Left: MODIS chlor_a
    im0 = axes[0].pcolormesh(
        chl_da.lon, chl_da.lat, chl_da,
        cmap=CMAP, vmin=0, vmax=MODIS_VMAX, shading="auto"
    )
    axes[0].contour(chl_da.lon, chl_da.lat, chl_da, levels=chl_levels, colors="k", linewidths=0.4)
    axes[0].set_title("MODIS-Aqua Chlorophyll-a (OCI)\nComposite")

    # Right: MODIS SST
    im1 = axes[1].pcolormesh(
        sst.lon, sst.lat, sst,
        cmap=CMAP, vmin=sst_vmin, vmax=sst_vmax, shading="auto"
    )
    axes[1].set_title("MODIS-Aqua SST\nComposite")

    # Two separate colorbars (clean + avoids misleading scales)
    cax0 = fig.add_axes([0.92, 0.56, 0.02, 0.30])
    cb0 = fig.colorbar(im0, cax=cax0, orientation="vertical")
    cb0.set_label("Chlorophyll-a (mg m$^{-3}$)")

    cax1 = fig.add_axes([0.92, 0.14, 0.02, 0.30])
    cb1 = fig.colorbar(im1, cax=cax1, orientation="vertical")
    cb1.set_label(sst_name)

    fig.suptitle(f"OCI-Context Composites — {tag}", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 0.90, 0.95])
    plt.savefig(out_fig, dpi=300)
    print(f"Saved → {out_fig}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    start, end, tag = period_start_end()

    chl = load_modis_var_period(start, end, tag, "chlor_a", FILELIST_CHL)
    sst = load_modis_var_period(start, end, tag, "sst", FILELIST_SST)

    plot_modis_chl_vs_modis_sst(chl, sst, tag)


if __name__ == "__main__":
    main()
