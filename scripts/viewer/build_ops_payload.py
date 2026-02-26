#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WATCH_THRESHOLD = 0.55
ACTION_THRESHOLD = 0.6238688594003279
LEGACY_BEST_F1 = 0.3926301481609915
LEGACY_DEFAULT = 0.5327723842346281

TILE_RE = re.compile(r"_T([0-9]{2})([A-Z])([A-Z]{2})_")

DRIVER_WEIGHTS = {
    "chlor_a_norm": 0.35,
    "nflh_norm": 0.35,
    "kd490_norm": 0.2,
    "sst_norm": 0.1,
}

RISK_COMPONENT_WEIGHTS = {
    "hab_prob": 0.4,
    "det_mean": 0.25,
    "oci_proxy_adj": 0.2,
    "seasonality_proxy_adj": 0.15,
}

DISAGREEMENT_DAMPING = 0.18
MIN_THRESHOLD_GAP = 0.04

OCI_LON_MIN = 49.5
OCI_LON_MAX = 60.8
OCI_LAT_MIN = 16.0
OCI_LAT_MAX = 27.0
OCI_SURFACE_RES_DEG = 0.2

DATE_RANGE_RE = re.compile(r"(\d{8})[_-](\d{8})")
DATE_ONE_RE = re.compile(r"(\d{8})")

MODIS_FILELISTS = {
    "chlor_a": Path("data/filelists/8d/filelist_8d_chlor_a_filtered.txt"),
    "sst": Path("data/filelists/8d/filelist_8d_sst.txt"),
}

MODIS_VAR_CANDIDATES = {
    "chlor_a": ["chlor_a", "chlora", "chla"],
    "sst": ["sst", "sea_surface_temperature", "SST", "sst4"],
}

MODIS_UNITS = {
    "chlor_a": "mg m^-3",
    "sst": "degC",
}


def safe_float(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def classify(
    risk: float | None,
    watch_threshold: float = WATCH_THRESHOLD,
    action_threshold: float = ACTION_THRESHOLD,
) -> str:
    x = safe_float(risk)
    if x is None:
        return "unknown"
    if x >= action_threshold:
        return "action"
    if x >= watch_threshold:
        return "watch"
    return "normal"


def _month_num(dt_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dt_series, errors="coerce", utc=True)
    return dt.dt.month.astype("Int64")


def _squash01(z: pd.Series, slope: float = 1.15) -> pd.Series:
    x = pd.to_numeric(z, errors="coerce")
    out = 1.0 / (1.0 + np.exp(-slope * x))
    out = pd.Series(out, index=x.index, dtype=float)
    out.loc[x.isna()] = np.nan
    return out


def _monthly_anomaly_z(series: pd.Series, month_num: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    m = pd.to_numeric(month_num, errors="coerce")
    df = pd.DataFrame({"v": s, "month_num": m})

    g = (
        df.dropna(subset=["v", "month_num"])
        .groupby("month_num", as_index=False)["v"]
        .agg(
            med="median",
            q25=lambda x: float(x.quantile(0.25)),
            q75=lambda x: float(x.quantile(0.75)),
        )
    )
    if g.empty:
        return pd.Series(np.nan, index=s.index, dtype=float)

    g["iqr"] = (g["q75"] - g["q25"]).replace(0.0, np.nan)
    global_med = float(df["v"].median()) if np.isfinite(df["v"].median()) else 0.0
    global_iqr = float(df["v"].quantile(0.75) - df["v"].quantile(0.25))
    if not np.isfinite(global_iqr) or global_iqr <= 0:
        global_iqr = float(df["v"].std()) if np.isfinite(df["v"].std()) and df["v"].std() > 0 else 1.0

    g["med"] = g["med"].fillna(global_med)
    g["iqr"] = g["iqr"].fillna(global_iqr)
    out = df.merge(g[["month_num", "med", "iqr"]], on="month_num", how="left")
    out["med"] = out["med"].fillna(global_med)
    out["iqr"] = out["iqr"].replace(0.0, np.nan).fillna(global_iqr)
    z = (out["v"] - out["med"]) / out["iqr"]
    z = z.clip(lower=-3.0, upper=3.0)
    return pd.Series(z.values, index=s.index, dtype=float)


def _threshold_from_exceedance(scores: pd.Series, exceedance_rate: float, fallback: float) -> float:
    s = pd.to_numeric(scores, errors="coerce").dropna()
    if s.empty:
        return float(fallback)
    r = float(np.clip(exceedance_rate, 1e-4, 0.9999))
    q = 1.0 - r
    x = s.quantile(q)
    return float(x) if np.isfinite(x) else float(fallback)


def derive_ops_thresholds(
    data_frames: list[pd.DataFrame],
    base_watch: float = WATCH_THRESHOLD,
    base_action: float = ACTION_THRESHOLD,
) -> dict[str, Any]:
    if not data_frames:
        return {
            "watch": float(base_watch),
            "action": float(base_action),
            "target_watch_rate": None,
            "target_action_rate": None,
            "actual_watch_rate": None,
            "actual_action_rate": None,
            "method": "fallback_static",
        }

    merged = pd.concat(data_frames, ignore_index=True)
    hab = pd.to_numeric(merged.get("hab_prob"), errors="coerce")
    ops = pd.to_numeric(merged.get("ops_risk"), errors="coerce")

    target_watch_rate = float((hab >= base_watch).mean()) if hab.notna().any() else None
    target_action_rate = float((hab >= base_action).mean()) if hab.notna().any() else None

    if target_watch_rate is None or target_action_rate is None:
        return {
            "watch": float(base_watch),
            "action": float(base_action),
            "target_watch_rate": target_watch_rate,
            "target_action_rate": target_action_rate,
            "actual_watch_rate": None,
            "actual_action_rate": None,
            "method": "fallback_static_missing_hab_prob",
        }

    watch = _threshold_from_exceedance(ops, target_watch_rate, base_watch)
    action = _threshold_from_exceedance(ops, target_action_rate, base_action)

    watch = float(np.clip(watch, 0.05, 0.95))
    action = float(np.clip(action, 0.05, 0.99))
    if action <= watch + MIN_THRESHOLD_GAP:
        action = float(min(0.99, watch + MIN_THRESHOLD_GAP))
    if watch >= action:
        watch = float(max(0.01, action - MIN_THRESHOLD_GAP))

    actual_watch = float((ops >= watch).mean()) if ops.notna().any() else None
    actual_action = float((ops >= action).mean()) if ops.notna().any() else None

    return {
        "watch": watch,
        "action": action,
        "target_watch_rate": target_watch_rate,
        "target_action_rate": target_action_rate,
        "actual_watch_rate": actual_watch,
        "actual_action_rate": actual_action,
        "method": "quantile_match_to_legacy_alert_load",
    }


def robust_scale(series: pd.Series, q_low: float = 0.1, q_high: float = 0.9) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    finite = s.dropna()
    if finite.empty:
        return pd.Series(np.nan, index=s.index, dtype=float)

    lo = float(finite.quantile(q_low))
    hi = float(finite.quantile(q_high))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out = pd.Series(np.nan, index=s.index, dtype=float)
        out.loc[s.notna()] = 0.5
        return out

    out = ((s - lo) / (hi - lo)).clip(0.0, 1.0)
    out.loc[s.isna()] = np.nan
    return out


def weighted_average(df: pd.DataFrame, weights: dict[str, float]) -> tuple[pd.Series, pd.Series]:
    num = pd.Series(0.0, index=df.index, dtype=float)
    den = pd.Series(0.0, index=df.index, dtype=float)
    total_weight = float(sum(weights.values())) if weights else 1.0

    for col, w in weights.items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        mask = vals.notna()
        if not mask.any():
            continue
        num.loc[mask] += vals.loc[mask] * float(w)
        den.loc[mask] += float(w)

    out = num / den.replace(0.0, np.nan)
    coverage = den / (total_weight if total_weight > 0 else 1.0)
    return out, coverage


def compute_ops_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in ["hab_prob", "p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med", "sst", "chlor_a", "kd490", "nflh"]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["det_mean"] = out[["p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med"]].mean(axis=1)
    out["det_spread"] = out[["p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med"]].std(axis=1)
    out["disagreement"] = (out["hab_prob"] - out["det_mean"]).abs()
    out["month_num"] = _month_num(out["datetime"])

    # Ocean-color zeros are usually no-data placeholders in this pipeline.
    out["chlor_a_norm"] = robust_scale(out["chlor_a"].where(out["chlor_a"] > 0))
    out["kd490_norm"] = robust_scale(out["kd490"].where(out["kd490"] > 0))
    out["nflh_norm"] = robust_scale(out["nflh"].where(out["nflh"] > 0))
    out["sst_norm"] = robust_scale(out["sst"])

    out["oci_proxy"], out["oci_coverage"] = weighted_average(out, DRIVER_WEIGHTS)
    out["oci_proxy_adj"] = (
        out["oci_proxy"] * out["oci_coverage"] + 0.5 * (1.0 - out["oci_coverage"])
    ).clip(0.0, 1.0)

    # Seasonal pressure: month-of-year anomaly transformed to risk-like [0,1].
    out["chlor_a_season_z"] = _monthly_anomaly_z(out["chlor_a"].where(out["chlor_a"] > 0), out["month_num"])
    out["kd490_season_z"] = _monthly_anomaly_z(out["kd490"].where(out["kd490"] > 0), out["month_num"])
    out["nflh_season_z"] = _monthly_anomaly_z(out["nflh"].where(out["nflh"] > 0), out["month_num"])
    out["sst_season_z"] = _monthly_anomaly_z(out["sst"], out["month_num"])

    out["chlor_a_season"] = _squash01(out["chlor_a_season_z"])
    out["kd490_season"] = _squash01(out["kd490_season_z"])
    out["nflh_season"] = _squash01(out["nflh_season_z"])
    out["sst_season"] = _squash01(out["sst_season_z"].clip(lower=0.0))

    season_weights = {
        "chlor_a_season": DRIVER_WEIGHTS["chlor_a_norm"],
        "nflh_season": DRIVER_WEIGHTS["nflh_norm"],
        "kd490_season": DRIVER_WEIGHTS["kd490_norm"],
        "sst_season": DRIVER_WEIGHTS["sst_norm"],
    }
    out["seasonality_proxy"], out["seasonality_coverage"] = weighted_average(out, season_weights)
    out["seasonality_proxy_adj"] = (
        out["seasonality_proxy"] * out["seasonality_coverage"] + 0.5 * (1.0 - out["seasonality_coverage"])
    ).clip(0.0, 1.0)

    blend, _ = weighted_average(out, RISK_COMPONENT_WEIGHTS)
    disagreement = out["disagreement"].clip(lower=0.0).fillna(0.0)
    damp = (1.0 - DISAGREEMENT_DAMPING * disagreement).clip(lower=0.75, upper=1.0)

    out["ops_risk"] = (blend * damp).clip(0.0, 1.0)
    out["ops_risk"] = out["ops_risk"].where(out["ops_risk"].notna(), out["hab_prob"])
    out["status"] = out["ops_risk"].map(lambda x: classify(x, WATCH_THRESHOLD, ACTION_THRESHOLD))

    return out


def to_iso_utc(ts: pd.Timestamp | None) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def parse_scene_utm(scene_id: str) -> tuple[int, bool] | None:
    m = TILE_RE.search(scene_id or "")
    if not m:
        return None
    zone = int(m.group(1))
    lat_band = m.group(2)
    is_north = lat_band >= "N"
    return zone, is_north


def utm_to_latlon(easting: float, northing: float, zone: int, northern: bool = True) -> tuple[float, float]:
    # WGS84 constants
    a = 6378137.0
    e = 0.0818191908
    e1sq = 0.006739497
    k0 = 0.9996

    x = float(easting) - 500000.0
    y = float(northing)
    if not northern:
        y -= 10000000.0

    m = y / k0
    mu = m / (a * (1 - (e**2) / 4.0 - 3.0 * (e**4) / 64.0 - 5.0 * (e**6) / 256.0))

    e1 = (1.0 - math.sqrt(1.0 - e * e)) / (1.0 + math.sqrt(1.0 - e * e))
    j1 = 3 * e1 / 2 - 27 * (e1**3) / 32
    j2 = 21 * (e1**2) / 16 - 55 * (e1**4) / 32
    j3 = 151 * (e1**3) / 96
    j4 = 1097 * (e1**4) / 512

    fp = mu + j1 * math.sin(2 * mu) + j2 * math.sin(4 * mu) + j3 * math.sin(6 * mu) + j4 * math.sin(8 * mu)

    c1 = e1sq * (math.cos(fp) ** 2)
    t1 = math.tan(fp) ** 2
    r1 = a * (1 - e * e) / ((1 - (e * math.sin(fp)) ** 2) ** 1.5)
    n1 = a / math.sqrt(1 - (e * math.sin(fp)) ** 2)
    d = x / (n1 * k0)

    q1 = n1 * math.tan(fp) / r1
    q2 = (d**2) / 2
    q3 = (5 + 3 * t1 + 10 * c1 - 4 * (c1**2) - 9 * e1sq) * (d**4) / 24
    q4 = (61 + 90 * t1 + 298 * c1 + 45 * (t1**2) - 252 * e1sq - 3 * (c1**2)) * (d**6) / 720
    lat = fp - q1 * (q2 - q3 + q4)

    q5 = d
    q6 = (1 + 2 * t1 + c1) * (d**3) / 6
    q7 = (5 - 2 * c1 + 28 * t1 - 3 * (c1**2) + 8 * e1sq + 24 * (t1**2)) * (d**5) / 120
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    lon = lon0 + (q5 - q6 + q7) / math.cos(fp)

    return math.degrees(lat), math.degrees(lon)


def write_geojson(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True)


def build_monthly_overlay(plant_id: str, plant_root: Path, month: str, out_root: Path) -> bool:
    month_dir = plant_root / month
    idx_path = month_dir / "chips" / "index.csv"
    det_path = month_dir / "detector_scores.csv"
    inf_path = month_dir / "inference.csv"

    if not idx_path.exists() or not det_path.exists():
        return False

    idx = pd.read_csv(idx_path)
    if idx.empty:
        return False

    idx = idx.copy()
    if "tile" not in idx.columns:
        return False
    idx["chip_id"] = idx["tile"].astype(str).map(lambda s: Path(s).stem)

    det = pd.read_csv(det_path)
    if "chip_id" not in det.columns:
        return False
    det["chip_id"] = det["chip_id"].astype(str).map(lambda s: Path(s).stem)
    for c in ["p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med"]:
        if c in det.columns:
            det[c] = pd.to_numeric(det[c], errors="coerce")

    merged = idx.merge(det, on="chip_id", how="left")

    if inf_path.exists():
        inf = pd.read_csv(inf_path)
        if "tile" in inf.columns:
            inf["chip_id"] = inf["tile"].astype(str).map(lambda s: Path(s).stem)
            cols = ["chip_id"] + [c for c in ["hab_prob", "p_fused", "prob"] if c in inf.columns]
            inf = inf[cols].drop_duplicates(subset=["chip_id"])
            merged = merged.merge(inf, on="chip_id", how="left")
        elif "chip_id" in inf.columns:
            inf["chip_id"] = inf["chip_id"].astype(str).map(lambda s: Path(s).stem)
            cols = ["chip_id"] + [c for c in ["hab_prob", "p_fused", "prob"] if c in inf.columns]
            inf = inf[cols].drop_duplicates(subset=["chip_id"])
            merged = merged.merge(inf, on="chip_id", how="left")

    for c in ["xmin", "xmax", "ymin", "ymax"]:
        if c not in merged.columns:
            return False
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    merged = merged.dropna(subset=["xmin", "xmax", "ymin", "ymax"])
    if merged.empty:
        return False

    features: list[dict[str, Any]] = []
    for _, r in merged.iterrows():
        scene_id = str(r.get("scene_id") or "")
        utm_info = parse_scene_utm(scene_id)
        if not utm_info:
            continue
        zone, is_north = utm_info

        # match existing pipeline correction
        xmin = -10.0 * float(r["xmin"])
        xmax = -10.0 * float(r["xmax"])
        ymin = 10.0 * float(r["ymin"])
        ymax = 10.0 * float(r["ymax"])

        try:
            lat1, lon1 = utm_to_latlon(xmin, ymin, zone, northern=is_north)
            lat2, lon2 = utm_to_latlon(xmax, ymax, zone, northern=is_north)
        except Exception:
            continue

        lon_min, lon_max = min(lon1, lon2), max(lon1, lon2)
        lat_min, lat_max = min(lat1, lat2), max(lat1, lat2)

        props = {
            "chip_id": str(r.get("chip_id")),
            "month_key": month,
            "datetime": str(r.get("datetime")) if pd.notna(r.get("datetime")) else None,
            "tile": str(r.get("tile")) if pd.notna(r.get("tile")) else None,
            "scene_id": scene_id,
        }
        for c in ["hab_prob", "p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med", "p_fused", "prob"]:
            if c in merged.columns:
                props[c] = safe_float(r.get(c))

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [round(lon_min, 6), round(lat_min, 6)],
                        [round(lon_max, 6), round(lat_min, 6)],
                        [round(lon_max, 6), round(lat_max, 6)],
                        [round(lon_min, 6), round(lat_max, 6)],
                        [round(lon_min, 6), round(lat_min, 6)],
                    ]],
                },
                "properties": props,
            }
        )

    if not features:
        return False

    out_dir = out_root / f"osm_way_{plant_id}"
    out_path = out_dir / f"{month}_tile_overlay.geojson"
    write_geojson(out_path, {"type": "FeatureCollection", "features": features})
    return True


def load_csv_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "datetime" not in df.columns:
        raise ValueError(f"Missing datetime in {path}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    # Ensure required score fields exist
    for c in ["hab_prob", "p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Optional driver fields
    for c in ["sst", "chlor_a", "kd490", "nflh", "fai_mean", "ndwi_mean", "rednir_mean"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # month key fallback
    if "month" in df.columns:
        m = df["month"].astype(str)
    elif "month_key" in df.columns:
        m = df["month_key"].astype(str)
    else:
        m = df["datetime"].dt.strftime("%Y-%m")
    df["month"] = m

    return df


def monthly_table(
    df: pd.DataFrame,
    risk_col: str = "ops_risk",
    watch_threshold: float = WATCH_THRESHOLD,
    action_threshold: float = ACTION_THRESHOLD,
) -> list[dict[str, Any]]:
    rc = risk_col if risk_col in df.columns else "hab_prob"
    g = (
        df.groupby("month", as_index=False)
        .agg(
            n=(rc, "size"),
            mean=(rc, "mean"),
            p95=(rc, lambda s: float(s.quantile(0.95))),
            max=(rc, "max"),
            model_mean=("hab_prob", "mean"),
            model_p95=("hab_prob", lambda s: float(s.quantile(0.95))),
            oci_mean=("oci_proxy", "mean"),
            oci_adj_mean=("oci_proxy_adj", "mean"),
            seasonality_mean=("seasonality_proxy_adj", "mean"),
            sst_mean=("sst", "mean"),
            chlor_a_mean=("chlor_a", "mean"),
            kd490_mean=("kd490", "mean"),
            nflh_mean=("nflh", "mean"),
            watch_rate=(rc, lambda s: float((s >= watch_threshold).mean())),
            action_rate=(rc, lambda s: float((s >= action_threshold).mean())),
        )
        .sort_values("month")
    )

    out: list[dict[str, Any]] = []
    for _, r in g.iterrows():
        p95 = safe_float(r["p95"])
        out.append(
            {
                "month": str(r["month"]),
                "n": int(r["n"]),
                "mean": safe_float(r["mean"]),
                "p95": p95,
                "max": safe_float(r["max"]),
                "model_mean": safe_float(r["model_mean"]),
                "model_p95": safe_float(r["model_p95"]),
                "oci_mean": safe_float(r["oci_mean"]),
                "oci_adj_mean": safe_float(r["oci_adj_mean"]),
                "seasonality_mean": safe_float(r["seasonality_mean"]),
                "sst_mean": safe_float(r["sst_mean"]),
                "chlor_a_mean": safe_float(r["chlor_a_mean"]),
                "kd490_mean": safe_float(r["kd490_mean"]),
                "nflh_mean": safe_float(r["nflh_mean"]),
                "watch_rate": safe_float(r["watch_rate"]),
                "action_rate": safe_float(r["action_rate"]),
                "status": classify(p95, watch_threshold, action_threshold),
            }
        )
    return out


def top_events(df: pd.DataFrame, n: int = 12, risk_col: str = "ops_risk") -> list[dict[str, Any]]:
    rc = risk_col if risk_col in df.columns else "hab_prob"
    d = df.copy()
    cols = [
        "tile",
        "scene_id",
        "chip_id",
        "datetime",
        "month",
        "ops_risk",
        "hab_prob",
        "p_frcnn_r50_med",
        "p_frcnn_mb_med",
        "p_ssd_mb_med",
        "sst",
        "chlor_a",
        "kd490",
        "nflh",
    ]
    for c in cols:
        if c not in d.columns:
            d[c] = np.nan

    d = d.sort_values(rc, ascending=False).copy()
    subset = [c for c in ["scene_id", "datetime", "month"] if c in d.columns]
    if subset:
        d = d.drop_duplicates(subset=subset, keep="first")
    d = d.head(n)
    out: list[dict[str, Any]] = []
    for _, r in d.iterrows():
        out.append(
            {
                "tile": str(r["tile"]) if pd.notna(r["tile"]) else None,
                "scene_id": str(r["scene_id"]) if pd.notna(r["scene_id"]) else None,
                "chip_id": str(r["chip_id"]) if pd.notna(r["chip_id"]) else None,
                "datetime": to_iso_utc(r["datetime"]),
                "month": str(r["month"]) if pd.notna(r["month"]) else None,
                "ops_risk": safe_float(r.get("ops_risk")),
                "hab_prob": safe_float(r["hab_prob"]),
                "p_frcnn_r50_med": safe_float(r["p_frcnn_r50_med"]),
                "p_frcnn_mb_med": safe_float(r["p_frcnn_mb_med"]),
                "p_ssd_mb_med": safe_float(r["p_ssd_mb_med"]),
                "sst": safe_float(r["sst"]),
                "chlor_a": safe_float(r["chlor_a"]),
                "kd490": safe_float(r["kd490"]),
                "nflh": safe_float(r["nflh"]),
            }
        )
    return out


def compact_timeseries(df: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        out.append(
            {
                "tile": str(r.get("tile")) if pd.notna(r.get("tile")) else None,
                "scene_id": str(r.get("scene_id")) if pd.notna(r.get("scene_id")) else None,
                "chip_id": str(r.get("chip_id")) if pd.notna(r.get("chip_id")) else None,
                "datetime": to_iso_utc(r["datetime"]),
                "month": str(r["month"]),
                "ops_risk": safe_float(r.get("ops_risk")),
                "hab_prob": safe_float(r["hab_prob"]),
                "p_frcnn_r50_med": safe_float(r["p_frcnn_r50_med"]),
                "p_frcnn_mb_med": safe_float(r["p_frcnn_mb_med"]),
                "p_ssd_mb_med": safe_float(r["p_ssd_mb_med"]),
                "det_mean": safe_float(r.get("det_mean")),
                "disagreement": safe_float(r.get("disagreement")),
                "sst": safe_float(r["sst"]),
                "chlor_a": safe_float(r["chlor_a"]),
                "kd490": safe_float(r["kd490"]),
                "nflh": safe_float(r["nflh"]),
                "sst_norm": safe_float(r.get("sst_norm")),
                "chlor_a_norm": safe_float(r.get("chlor_a_norm")),
                "kd490_norm": safe_float(r.get("kd490_norm")),
                "nflh_norm": safe_float(r.get("nflh_norm")),
                "oci_proxy": safe_float(r.get("oci_proxy")),
                "oci_proxy_adj": safe_float(r.get("oci_proxy_adj")),
                "oci_coverage": safe_float(r.get("oci_coverage")),
                "chlor_a_season_z": safe_float(r.get("chlor_a_season_z")),
                "kd490_season_z": safe_float(r.get("kd490_season_z")),
                "nflh_season_z": safe_float(r.get("nflh_season_z")),
                "sst_season_z": safe_float(r.get("sst_season_z")),
                "seasonality_proxy": safe_float(r.get("seasonality_proxy")),
                "seasonality_proxy_adj": safe_float(r.get("seasonality_proxy_adj")),
                "seasonality_coverage": safe_float(r.get("seasonality_coverage")),
                "fai_mean": safe_float(r["fai_mean"]),
                "ndwi_mean": safe_float(r["ndwi_mean"]),
                "rednir_mean": safe_float(r["rednir_mean"]),
            }
        )
    return out


def cadence_hours(df: pd.DataFrame) -> dict[str, Any]:
    x = df["datetime"].dropna().drop_duplicates().sort_values()
    if len(x) < 2:
        return {"median": None, "p90": None, "max": None}
    d = x.diff().dropna().dt.total_seconds() / 3600.0
    return {
        "median": safe_float(d.median()),
        "p90": safe_float(d.quantile(0.9)),
        "max": safe_float(d.max()),
    }


def load_drift(drift_csv: Path, drift_overall_json: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    drift_by_plant: dict[str, Any] = {}
    if drift_csv.exists():
        d = pd.read_csv(drift_csv)
        for _, r in d.iterrows():
            pid = str(r["plant"]).replace("osm_way_", "")
            drift_by_plant[pid] = {
                "psi": safe_float(r.get("psi")),
                "ks_D": safe_float(r.get("ks_D")),
                "ks_pvalue": safe_float(r.get("ks_pvalue")),
                "train_rows": int(r.get("train_rows")) if pd.notna(r.get("train_rows")) else None,
                "test_rows": int(r.get("test_rows")) if pd.notna(r.get("test_rows")) else None,
            }

    overall = {
        "psi": None,
        "ks_D": None,
        "ks_pvalue": None,
        "train_rows": None,
        "test_rows": None,
    }
    if drift_overall_json.exists():
        with open(drift_overall_json, "r", encoding="utf-8") as f:
            o = json.load(f)
        overall.update(
            {
                "psi": safe_float(o.get("psi")),
                "ks_D": safe_float(o.get("ks_D")),
                "ks_pvalue": safe_float(o.get("ks_pvalue")),
                "train_rows": int(o.get("train_rows")) if o.get("train_rows") is not None else None,
                "test_rows": int(o.get("test_rows")) if o.get("test_rows") is not None else None,
            }
        )

    return drift_by_plant, overall


def copy_optional(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _month_start_end(month_key: str) -> tuple[date, date] | None:
    try:
        start = datetime.strptime(f"{month_key}-01", "%Y-%m-%d").date()
    except Exception:
        return None
    end = (start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    return start, end


def _date_midpoint_from_name(name: str) -> date | None:
    src = Path(str(name).strip()).name
    m = DATE_RANGE_RE.search(src)
    if m:
        d0 = datetime.strptime(m.group(1), "%Y%m%d").date()
        d1 = datetime.strptime(m.group(2), "%Y%m%d").date()
        return d0 + (d1 - d0) // 2
    m = DATE_ONE_RE.search(src)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    return None


def _slice_in_coord_order(values: np.ndarray, vmin: float, vmax: float) -> slice:
    if values.size < 2:
        return slice(vmin, vmax)
    ascending = bool(float(values[0]) < float(values[-1]))
    return slice(vmin, vmax) if ascending else slice(vmax, vmin)


def _find_lon_lat_names(ds: Any) -> tuple[str | None, str | None]:
    lon_candidates = ["lon", "longitude", "LONGITUDE", "x"]
    lat_candidates = ["lat", "latitude", "LATITUDE", "y"]
    lon_name = next((c for c in lon_candidates if c in ds.coords or c in ds.variables), None)
    lat_name = next((c for c in lat_candidates if c in ds.coords or c in ds.variables), None)
    return lon_name, lat_name


def _filter_modis_filelist_month(filelist_path: Path, month_key: str) -> list[str]:
    if not filelist_path.exists():
        return []
    bounds = _month_start_end(month_key)
    if bounds is None:
        return []
    start, end = bounds

    out: list[str] = []
    with open(filelist_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            mid = _date_midpoint_from_name(name)
            if mid is None:
                continue
            if start <= mid <= end:
                out.append(name)
    return out


def _ensure_modis_files(
    *,
    repo: Path,
    month_key: str,
    variable: str,
    out_dir: Path,
    appkey: str | None,
    allow_download: bool = True,
) -> tuple[list[Path], bool]:
    filelist_rel = MODIS_FILELISTS.get(variable)
    if filelist_rel is None:
        return [], False
    filelist = repo / filelist_rel
    entries = _filter_modis_filelist_month(filelist, month_key)
    if not entries:
        print(f"[warn] no filelist entries for {variable} @ {month_key}")
        return [], False

    out_dir.mkdir(parents=True, exist_ok=True)
    expected = [out_dir / Path(e.split("?")[0]).name for e in entries]

    missing_entries = [
        e for e, p in zip(entries, expected)
        if not (p.exists() and p.stat().st_size > 0)
    ]
    hard_download_failure = False

    if missing_entries:
        if not appkey:
            print(f"[warn] OBPG_APPKEY missing; cannot download {len(missing_entries)} {variable} files for {month_key}")
        elif not allow_download:
            pass
        else:
            subset_file = out_dir / f"filelist_subset_{month_key}_{variable}.txt"
            with open(subset_file, "w", encoding="utf-8") as f:
                f.write("\n".join(missing_entries))
                f.write("\n")

            dl_script = repo / "scripts" / "download" / "obdaac_download.py"
            cmd = [
                sys.executable,
                str(dl_script),
                "--filelist",
                str(subset_file),
                "--odir",
                str(out_dir),
                "--appkey",
                appkey,
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                hard_download_failure = True
                raw_msg = (proc.stderr or proc.stdout or "").strip()
                if raw_msg:
                    lines = [ln.strip() for ln in raw_msg.splitlines() if ln.strip()]
                    msg = lines[-1] if lines else "unknown download error"
                else:
                    msg = "unknown download error"
                if len(msg) > 280:
                    msg = msg[:277] + "..."
                print(f"[warn] OBPG download failed for {variable} @ {month_key}: {msg}")
            else:
                print(f"[ok] downloaded missing {variable} files for {month_key}")

    existing = [p for p in expected if p.exists() and p.stat().st_size > 0]
    return sorted(existing), hard_download_failure


def _load_modis_month_composite(
    nc_files: list[Path],
    variable: str,
) -> dict[str, Any] | None:
    if not nc_files:
        return None

    try:
        import xarray as xr  # type: ignore
    except Exception as exc:
        print(f"[warn] xarray unavailable; cannot build MODIS composites: {exc}")
        return None

    var_candidates = MODIS_VAR_CANDIDATES.get(variable, [variable])
    stack: list[np.ndarray] = []
    template_lon: np.ndarray | None = None
    template_lat: np.ndarray | None = None

    for nc in nc_files:
        try:
            with xr.open_dataset(nc, decode_cf=True, mask_and_scale=True) as ds:
                var_name = next((name for name in var_candidates if name in ds.data_vars), None)
                if var_name is None:
                    continue
                da = ds[var_name]

                lon_name, lat_name = _find_lon_lat_names(ds)
                if lon_name is None or lat_name is None:
                    continue

                if lon_name != "lon" or lat_name != "lat":
                    da = da.rename({lon_name: "lon", lat_name: "lat"})

                # Keep only the mapped 2D surface.
                extra_dims = [d for d in da.dims if d not in {"lat", "lon"}]
                for dim in extra_dims:
                    da = da.isel({dim: 0}, drop=True)

                if "lat" not in da.dims or "lon" not in da.dims:
                    continue

                lon_vals = np.asarray(da["lon"].values, dtype=float)
                lat_vals = np.asarray(da["lat"].values, dtype=float)
                lon_slice = _slice_in_coord_order(lon_vals, OCI_LON_MIN, OCI_LON_MAX)
                lat_slice = _slice_in_coord_order(lat_vals, OCI_LAT_MIN, OCI_LAT_MAX)
                da = da.sel(lon=lon_slice, lat=lat_slice)

                if da.sizes.get("lat", 0) == 0 or da.sizes.get("lon", 0) == 0:
                    continue

                arr = np.asarray(da.values, dtype=float)
                arr[~np.isfinite(arr)] = np.nan

                if variable == "chlor_a":
                    arr = np.where(arr > 0, arr, np.nan)
                elif variable == "sst":
                    med = np.nanmedian(arr)
                    if np.isfinite(med) and med > 100:
                        arr = arr - 273.15

                if not np.isfinite(arr).any():
                    continue

                stack.append(arr)
                template_lon = np.asarray(da["lon"].values, dtype=float)
                template_lat = np.asarray(da["lat"].values, dtype=float)
        except Exception as exc:
            print(f"[warn] failed to read {nc.name}: {exc}")

    if not stack or template_lon is None or template_lat is None:
        return None

    with np.errstate(invalid="ignore"):
        composite = np.nanmedian(np.stack(stack, axis=0), axis=0)

    if not np.isfinite(composite).any():
        return None

    return {
        "array": composite,
        "lon": template_lon,
        "lat": template_lat,
        "n_files": len(stack),
    }


def _apply_surface_orientation(
    rgba: np.ndarray,
    lon_vals: np.ndarray,
    lat_vals: np.ndarray,
) -> np.ndarray:
    out = rgba
    if lat_vals.size >= 2 and float(lat_vals[0]) < float(lat_vals[-1]):
        out = np.flipud(out)
    if lon_vals.size >= 2 and float(lon_vals[0]) > float(lon_vals[-1]):
        out = np.fliplr(out)
    return out


def _modis_limits(series: list[np.ndarray], variable: str) -> tuple[float, float]:
    if variable == "chlor_a":
        return 0.0, 4.5

    finite_chunks = [a[np.isfinite(a)] for a in series if np.isfinite(a).any()]
    finite = np.concatenate(finite_chunks) if finite_chunks else np.array([])
    if finite.size == 0:
        return 24.0, 33.0
    lo = float(np.nanpercentile(finite, 5))
    hi = float(np.nanpercentile(finite, 95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 24.0, 33.0
    return lo, hi


def build_monthly_modis_surfaces(
    *,
    repo: Path,
    months: list[str],
    out_root: Path,
    appkey: str | None,
) -> dict[str, dict[str, Any]]:
    if not months:
        return {}

    mpl_cfg = out_root / "_mpl_cache"
    os_cache = out_root / "_font_cache"
    mpl_cfg.mkdir(parents=True, exist_ok=True)
    os_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cfg))
    os.environ.setdefault("XDG_CACHE_HOME", str(os_cache))

    from matplotlib import colormaps, colors
    import matplotlib.pyplot as plt

    cache_root = repo / "data" / "l3" / "modis_8d"
    surface_dir = out_root / "modis_surfaces"
    if surface_dir.exists():
        for old in surface_dir.glob("*_surface.png"):
            old.unlink()
    surface_dir.mkdir(parents=True, exist_ok=True)

    composites: dict[str, dict[str, dict[str, Any]]] = {var: {} for var in MODIS_FILELISTS}
    missing_counts = {var: 0 for var in MODIS_FILELISTS}
    downloads_enabled = True

    for month in sorted(set(months)):
        for var in MODIS_FILELISTS:
            nc_files, download_failed = _ensure_modis_files(
                repo=repo,
                month_key=month,
                variable=var,
                out_dir=cache_root / var,
                appkey=appkey,
                allow_download=downloads_enabled,
            )
            if download_failed and downloads_enabled:
                downloads_enabled = False
                print("[warn] disabling further OBPG download attempts after first failure; using only local MODIS files")
            comp = _load_modis_month_composite(nc_files, var)
            if comp is not None:
                comp["array"] = _box_smooth(np.asarray(comp["array"], dtype=float), k=3)
                composites[var][month] = comp
            else:
                missing_counts[var] += 1

    for var, miss_n in missing_counts.items():
        if miss_n:
            print(f"[warn] no MODIS composite for {var} in {miss_n} month(s)")

    limits_by_var = {
        var: _modis_limits([np.asarray(v["array"], dtype=float) for v in month_map.values()], var)
        for var, month_map in composites.items()
    }

    cmap = colormaps["turbo"]
    manifest: dict[str, dict[str, Any]] = {}

    for month in sorted(set(months)):
        month_manifest: dict[str, Any] = {}

        for var, month_map in composites.items():
            comp = month_map.get(month)
            if comp is None:
                continue

            arr = np.asarray(comp["array"], dtype=float)
            lon_vals = np.asarray(comp["lon"], dtype=float)
            lat_vals = np.asarray(comp["lat"], dtype=float)

            lo, hi = limits_by_var.get(var, (0.0, 1.0))
            norm = colors.Normalize(vmin=lo, vmax=hi, clip=True)
            rgba = cmap(norm(arr))
            rgba[..., 3] = np.where(np.isfinite(arr), 0.44, 0.0)
            rgba = _apply_surface_orientation(rgba, lon_vals, lat_vals)

            out_name = f"{month}_{var}_surface.png"
            out_path = surface_dir / out_name
            plt.imsave(out_path, rgba)

            month_manifest[var] = {
                "image": "/" + str(Path("ops") / "modis_surfaces" / out_name).replace("\\", "/"),
                "bounds": [
                    float(np.nanmin(lon_vals)),
                    float(np.nanmin(lat_vals)),
                    float(np.nanmax(lon_vals)),
                    float(np.nanmax(lat_vals)),
                ],
                "variable": var,
                "units": MODIS_UNITS.get(var),
                "n_files": int(comp["n_files"]),
                "value_min": safe_float(np.nanmin(arr)),
                "value_max": safe_float(np.nanmax(arr)),
                "render_vmin": safe_float(lo),
                "render_vmax": safe_float(hi),
                "source": "MODIS-Aqua L3m 8D monthly median composite",
            }

        if month_manifest:
            manifest[month] = month_manifest

    return manifest


def _box_smooth(a: np.ndarray, k: int = 3) -> np.ndarray:
    if a.ndim != 2 or k <= 1:
        return a
    pad = k // 2
    ap = np.pad(a, ((pad, pad), (pad, pad)), mode="edge")
    out = np.zeros_like(a, dtype=float)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[i, j] = float(np.nanmean(ap[i : i + k, j : j + k]))
    return out


def build_monthly_oci_surfaces(
    points_by_month: dict[str, list[dict[str, float]]],
    out_root: Path,
) -> dict[str, dict[str, Any]]:
    mpl_cfg = out_root / "_mpl_cache"
    os_cache = out_root / "_font_cache"
    mpl_cfg.mkdir(parents=True, exist_ok=True)
    os_cache.mkdir(parents=True, exist_ok=True)

    import os

    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cfg))
    os.environ.setdefault("XDG_CACHE_HOME", str(os_cache))

    from matplotlib import colormaps
    import matplotlib.pyplot as plt

    surface_dir = out_root / "oci_surfaces"
    if surface_dir.exists():
        for old in surface_dir.glob("*_oci_surface.png"):
            old.unlink()
    surface_dir.mkdir(parents=True, exist_ok=True)

    lons = np.arange(OCI_LON_MIN, OCI_LON_MAX + OCI_SURFACE_RES_DEG, OCI_SURFACE_RES_DEG)
    lats = np.arange(OCI_LAT_MIN, OCI_LAT_MAX + OCI_SURFACE_RES_DEG, OCI_SURFACE_RES_DEG)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    cmap = colormaps["turbo"]
    manifest: dict[str, dict[str, Any]] = {}

    for month, points in sorted(points_by_month.items()):
        pts = [
            p for p in points
            if safe_float(p.get("lon")) is not None
            and safe_float(p.get("lat")) is not None
            and safe_float(p.get("oci")) is not None
        ]
        if len(pts) < 2:
            continue

        num = np.zeros_like(lon_grid, dtype=float)
        den = np.zeros_like(lon_grid, dtype=float)

        for p in pts:
            lon0 = float(p["lon"])
            lat0 = float(p["lat"])
            val = float(p["oci"])

            dx = (lon_grid - lon0) * np.cos(np.deg2rad(lat_grid))
            dy = lat_grid - lat0
            dist = np.sqrt(dx * dx + dy * dy)
            w = 1.0 / np.maximum(dist, 1e-3) ** 2
            num += w * val
            den += w

        z = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        z = _box_smooth(z, k=5)
        z = np.clip(z, 0.0, 1.0)

        rgba = cmap(z)
        alpha = np.where(np.isfinite(z), 0.42, 0.0)
        rgba[..., 3] = alpha
        rgba = np.flipud(rgba)

        out_name = f"{month}_oci_surface.png"
        out_path = surface_dir / out_name

        plt.imsave(out_path, rgba)

        manifest[month] = {
            "image": "/" + str(Path("ops") / "oci_surfaces" / out_name).replace("\\", "/"),
            "bounds": [OCI_LON_MIN, OCI_LAT_MIN, OCI_LON_MAX, OCI_LAT_MAX],
            "n_points": len(pts),
            "oci_min": safe_float(np.nanmin(z)),
            "oci_max": safe_float(np.nanmax(z)),
        }

    return manifest


def main() -> None:
    ap = argparse.ArgumentParser("Build operations payload for rednet-risk-viewer")
    ap.add_argument("--repo_root", default=".")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()

    viewer_public = repo / "rednet-risk-viewer" / "public"
    plants_json_path = viewer_public / "data" / "plants.json"
    out_root = viewer_public / "ops"
    out_root.mkdir(parents=True, exist_ok=True)
    overlays_root = viewer_public / "overlays"
    overlays_root.mkdir(parents=True, exist_ok=True)

    # regenerate clean artifacts each run
    for sub in ["transport", "aoi"]:
        subdir = out_root / sub
        if subdir.exists():
            shutil.rmtree(subdir)

    deployment_root = repo / "deployment" / "outputs" / "by_plant"
    aoi_root = repo / "deployment" / "aoi" / "plant_aois"

    drift_csv = repo / "runs" / "eval" / "generalization" / "generalization_2025_watch055" / "drift_2017-2024_vs_2025_by_plant.csv"
    drift_overall_json = repo / "runs" / "eval" / "generalization" / "generalization_2025_watch055" / "drift_overall_2017-2024_vs_2025.json"

    drift_by_plant, drift_overall = load_drift(drift_csv, drift_overall_json)

    with open(plants_json_path, "r", encoding="utf-8") as f:
        plants_raw = json.load(f)

    plants_out: list[dict[str, Any]] = []
    plants_stage: list[dict[str, Any]] = []

    transport_manifest: dict[str, dict[str, str]] = {}
    aoi_manifest: dict[str, str] = {}

    for p in plants_raw:
        pid = str(p["id"])
        osm = f"osm_way_{pid}"

        deployment_csv = deployment_root / osm / "inference_all_months.csv"
        fallback_csv = viewer_public / "data" / f"plant_{pid}.csv"
        source_csv = deployment_csv if deployment_csv.exists() else fallback_csv

        if not source_csv.exists():
            print(f"[skip] no timeseries for plant {pid}")
            continue

        df = load_csv_timeseries(source_csv)
        if df.empty:
            print(f"[skip] empty timeseries for plant {pid}")
            continue

        # build month overlays directly from deployment monthly chips + scores
        plant_deploy_root = deployment_root / osm
        overlay_plant_dir = overlays_root / osm
        if overlay_plant_dir.exists():
            for old in overlay_plant_dir.glob("*_tile_overlay.geojson"):
                old.unlink()
        for month in sorted(df["month"].dropna().astype(str).unique()):
            build_monthly_overlay(pid, plant_deploy_root, month, overlays_root)

        df = compute_ops_signals(df)

        try:
            source_csv_rel = str(source_csv.relative_to(repo)).replace("\\", "/")
        except ValueError:
            source_csv_rel = str(source_csv)

        plants_stage.append(
            {
                "id": pid,
                "osm_id": osm,
                "name": p.get("name"),
                "lat": safe_float(p.get("lat")),
                "lon": safe_float(p.get("lon")),
                "source_csv": source_csv_rel,
                "df": df,
                "drift": drift_by_plant.get(pid),
            }
        )

        # transport layers intentionally disabled for the current UI iteration

        # AOI
        aoi_src = aoi_root / f"plant_osm_way_{pid}.geojson"
        if aoi_src.exists():
            rel = Path("aoi") / f"plant_osm_way_{pid}.geojson"
            dst = out_root / rel
            copy_optional(aoi_src, dst)
            aoi_manifest[pid] = "/" + str(Path("ops") / rel).replace("\\", "/")

    threshold_fit = derive_ops_thresholds([p["df"] for p in plants_stage], WATCH_THRESHOLD, ACTION_THRESHOLD)
    watch_threshold = float(threshold_fit["watch"])
    action_threshold = float(threshold_fit["action"])

    oci_points_by_month: dict[str, list[dict[str, float]]] = {}

    for rec in plants_stage:
        df = rec["df"]
        latest = df.iloc[-1]
        last_30 = df[df["datetime"] >= (latest["datetime"] - pd.Timedelta(days=30))]

        latest_risk = safe_float(latest.get("ops_risk"))
        status = classify(latest_risk, watch_threshold, action_threshold)

        monthly = monthly_table(
            df,
            risk_col="ops_risk",
            watch_threshold=watch_threshold,
            action_threshold=action_threshold,
        )
        for m in monthly:
            if safe_float(rec.get("lon")) is None or safe_float(rec.get("lat")) is None:
                continue
            if safe_float(m.get("oci_adj_mean")) is None:
                continue
            oci_points_by_month.setdefault(str(m["month"]), []).append(
                {
                    "lon": float(rec["lon"]),
                    "lat": float(rec["lat"]),
                    "oci": float(m["oci_adj_mean"]),
                }
            )

        plants_out.append(
            {
                "id": rec["id"],
                "osm_id": rec["osm_id"],
                "name": rec["name"],
                "lat": rec["lat"],
                "lon": rec["lon"],
                "source_csv": rec["source_csv"],
                "latest": {
                    "datetime": to_iso_utc(latest["datetime"]),
                    "month": str(latest["month"]),
                    "ops_risk": latest_risk,
                    "hab_prob": safe_float(latest["hab_prob"]),
                    "p_frcnn_r50_med": safe_float(latest["p_frcnn_r50_med"]),
                    "p_frcnn_mb_med": safe_float(latest["p_frcnn_mb_med"]),
                    "p_ssd_mb_med": safe_float(latest["p_ssd_mb_med"]),
                    "det_mean": safe_float(latest.get("det_mean")),
                    "disagreement": safe_float(latest.get("disagreement")),
                    "sst": safe_float(latest["sst"]),
                    "chlor_a": safe_float(latest["chlor_a"]),
                    "kd490": safe_float(latest["kd490"]),
                    "nflh": safe_float(latest["nflh"]),
                    "oci_proxy": safe_float(latest.get("oci_proxy")),
                    "oci_proxy_adj": safe_float(latest.get("oci_proxy_adj")),
                    "oci_coverage": safe_float(latest.get("oci_coverage")),
                    "seasonality_proxy": safe_float(latest.get("seasonality_proxy")),
                    "seasonality_proxy_adj": safe_float(latest.get("seasonality_proxy_adj")),
                    "seasonality_coverage": safe_float(latest.get("seasonality_coverage")),
                    "status": status,
                },
                "summary": {
                    "n_obs": int(len(df)),
                    "n_months": int(df["month"].nunique()),
                    "n_scenes": int(df["scene_id"].nunique()) if "scene_id" in df.columns else None,
                    "start": to_iso_utc(df["datetime"].min()),
                    "end": to_iso_utc(df["datetime"].max()),
                    "hab_mean": safe_float(df["hab_prob"].mean()),
                    "hab_p95": safe_float(df["hab_prob"].quantile(0.95)),
                    "hab_max": safe_float(df["hab_prob"].max()),
                    "watch_rate": safe_float((df["hab_prob"] >= WATCH_THRESHOLD).mean()),
                    "action_rate": safe_float((df["hab_prob"] >= ACTION_THRESHOLD).mean()),
                    "ops_mean": safe_float(df["ops_risk"].mean()),
                    "ops_p95": safe_float(df["ops_risk"].quantile(0.95)),
                    "ops_max": safe_float(df["ops_risk"].max()),
                    "ops_watch_rate": safe_float((df["ops_risk"] >= watch_threshold).mean()),
                    "ops_action_rate": safe_float((df["ops_risk"] >= action_threshold).mean()),
                    "disagreement_mean": safe_float(df["disagreement"].mean()),
                    "disagreement_p95": safe_float(df["disagreement"].quantile(0.95)),
                    "oci_mean": safe_float(df["oci_proxy"].mean()),
                    "oci_adj_mean": safe_float(df["oci_proxy_adj"].mean()),
                    "oci_p95": safe_float(df["oci_proxy"].quantile(0.95)),
                    "oci_coverage_mean": safe_float(df["oci_coverage"].mean()),
                    "seasonality_mean": safe_float(df["seasonality_proxy_adj"].mean()),
                    "seasonality_p95": safe_float(df["seasonality_proxy_adj"].quantile(0.95)),
                    "cadence_hours": cadence_hours(df),
                    "last_30d_mean": safe_float(last_30["ops_risk"].mean()) if len(last_30) else None,
                    "last_30d_max": safe_float(last_30["ops_risk"].max()) if len(last_30) else None,
                },
                "monthly": monthly,
                "top_events": top_events(df, n=16, risk_col="ops_risk"),
                "timeseries": compact_timeseries(df),
                "drift": rec["drift"],
            }
        )

    plants_out.sort(key=lambda x: x.get("name") or "")
    months_for_surfaces = sorted(
        {
            str(m)
            for rec in plants_stage
            for m in rec["df"]["month"].dropna().astype(str).unique()
        }
    )

    obpg_appkey = os.environ.get("OBPG_APPKEY")
    modis_surface_manifest = build_monthly_modis_surfaces(
        repo=repo,
        months=months_for_surfaces,
        out_root=out_root,
        appkey=obpg_appkey,
    )

    # Keep fallback synthetic OCI surfaces only when MODIS composites are unavailable.
    oci_surface_manifest = (
        {}
        if modis_surface_manifest
        else build_monthly_oci_surfaces(oci_points_by_month, out_root)
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "watch": watch_threshold,
            "action": action_threshold,
            "watch_base": WATCH_THRESHOLD,
            "action_base": ACTION_THRESHOLD,
            "legacy_best_f1": LEGACY_BEST_F1,
            "legacy_default": LEGACY_DEFAULT,
            "calibration": threshold_fit,
        },
        "risk_index": {
            "name": "ops_risk_v2_seasonal",
            "description": "Operational HAB risk blending model probability, detector consensus, OCI context, and month-of-year seasonal anomaly pressure with disagreement damping.",
            "component_weights": {
                "hab_prob": RISK_COMPONENT_WEIGHTS["hab_prob"],
                "det_mean": RISK_COMPONENT_WEIGHTS["det_mean"],
                "oci_proxy_adj": RISK_COMPONENT_WEIGHTS["oci_proxy_adj"],
                "seasonality_proxy_adj": RISK_COMPONENT_WEIGHTS["seasonality_proxy_adj"],
            },
            "oci_driver_weights": {
                "chlor_a_norm": DRIVER_WEIGHTS["chlor_a_norm"],
                "nflh_norm": DRIVER_WEIGHTS["nflh_norm"],
                "kd490_norm": DRIVER_WEIGHTS["kd490_norm"],
                "sst_norm": DRIVER_WEIGHTS["sst_norm"],
            },
            "seasonality_definition": "Per-feature monthly anomaly z-score ((x - month_median) / month_IQR), squashed to [0,1].",
            "disagreement_damping": DISAGREEMENT_DAMPING,
        },
        "drift_overall": drift_overall,
        "transport_manifest": transport_manifest,
        "aoi_manifest": aoi_manifest,
        "modis_surface_manifest": modis_surface_manifest,
        "oci_surface_manifest": oci_surface_manifest,
        "plants": plants_out,
    }

    out_file = out_root / "ops_payload.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)

    print(f"[ok] wrote {out_file}")
    print(f"[ok] plants: {len(plants_out)}")


if __name__ == "__main__":
    main()
