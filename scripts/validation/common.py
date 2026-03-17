#!/usr/bin/env python3
from __future__ import annotations

import csv
import glob
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_WATCH_THRESHOLD = 0.55
DEFAULT_ACTION_THRESHOLD = 0.6238688594
DEFAULT_PRIMARY_WINDOW_DAYS = 3
DEFAULT_SENSITIVITY_WINDOW_DAYS = 7
DEFAULT_MAX_PLANT_DISTANCE_KM = 150.0

EVENT_REQUIRED_COLUMNS = [
    "event_id",
    "source_type",
    "source_url",
    "event_date_start",
    "event_date_end",
    "location_name",
    "plant_id",
    "lat",
    "lon",
    "event_class",
    "severity",
    "evidence_text",
    "confidence",
    "external_positive",
]

INSITU_REQUIRED_COLUMNS = [
    "sample_id",
    "datetime_utc",
    "lat",
    "lon",
    "plant_id",
    "hab_event",
    "evidence_type",
    "species",
    "cell_count",
    "toxin_value",
    "notes",
    "source_url",
]

DISALLOWED_PRIMARY_EVENT_CLASSES = {
    "chlorophyll_only",
    "seasonal_bloom_commentary",
    "undated_review",
}

_OPS_MODULE: Any | None = None


def ensure_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns for {label}: {missing}")


def normalize_string(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_nullable_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_plant_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    m = re.search(r"(\d+)", text)
    return m.group(1) if m else text


def parse_datetime_utc(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return pd.NaT
    return ts


def parse_date_start(value: Any) -> pd.Timestamp:
    ts = parse_datetime_utc(value)
    if pd.isna(ts):
        return pd.NaT
    return ts.normalize()


def parse_date_end(value: Any) -> pd.Timestamp:
    ts = parse_datetime_utc(value)
    if pd.isna(ts):
        return pd.NaT
    return ts.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)


def midpoint_timestamp(start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(start) or pd.isna(end):
        return pd.NaT
    return start + (end - start) / 2


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def load_plants(plants_json_path: Path) -> pd.DataFrame:
    plants = json.loads(plants_json_path.read_text(encoding="utf-8"))
    df = pd.DataFrame(plants)
    if df.empty:
        raise SystemExit(f"No plants found in {plants_json_path}")
    df["plant_id"] = df["id"].map(normalize_plant_id)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df[["plant_id", "name", "lat", "lon"]].copy()


def load_location_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            alias = normalize_string(row.get("location_alias"))
            plant_id = normalize_plant_id(row.get("plant_id"))
            if alias and plant_id:
                mapping[alias] = plant_id
    return mapping


def assign_plant(
    record: dict[str, Any],
    plants: pd.DataFrame,
    location_map: dict[str, str],
    max_distance_km: float = DEFAULT_MAX_PLANT_DISTANCE_KM,
) -> tuple[str | None, str | None, float | None]:
    explicit = normalize_plant_id(record.get("plant_id"))
    if explicit and explicit in set(plants["plant_id"]):
        return explicit, "plant_id", 0.0

    location_name = normalize_string(record.get("location_name"))
    if location_name and location_name in location_map:
        plant_id = location_map[location_name]
        if plant_id in set(plants["plant_id"]):
            return plant_id, "location_map", None

    lat = pd.to_numeric(pd.Series([record.get("lat")]), errors="coerce").iloc[0]
    lon = pd.to_numeric(pd.Series([record.get("lon")]), errors="coerce").iloc[0]
    if pd.notna(lat) and pd.notna(lon):
        dists = plants.apply(lambda r: haversine_km(lat, lon, r["lat"], r["lon"]), axis=1)
        idx = dists.idxmin()
        best_dist = float(dists.loc[idx])
        if best_dist <= max_distance_km:
            plant_id = str(plants.loc[idx, "plant_id"])
            return plant_id, "nearest_plant", best_dist
    return None, None, None


def load_prediction_series(prediction_glob: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for path_str in sorted(glob.glob(prediction_glob)):
        path = Path(path_str)
        m = re.search(r"plant_(\d+)", path.name)
        if not m:
            continue
        plant_id = m.group(1)
        df = pd.read_csv(path)
        if "datetime" not in df.columns:
            continue
        df = df.copy()
        df["plant_id"] = plant_id
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df = df.dropna(subset=["datetime"]).reset_index(drop=True)
        df = maybe_compute_ops_signals(df)
        sort_cols = ["datetime"]
        if "ops_risk" in df.columns:
            df["ops_risk"] = pd.to_numeric(df["ops_risk"], errors="coerce")
            sort_cols.append("ops_risk")
            ascending = [True, False]
        elif "hab_prob" in df.columns:
            df["hab_prob"] = pd.to_numeric(df["hab_prob"], errors="coerce")
            sort_cols.append("hab_prob")
            ascending = [True, False]
        else:
            ascending = [True]
        sizes = df.groupby("datetime").size().rename("n_rows_aggregated").reset_index()
        agg = (
            df.sort_values(sort_cols, ascending=ascending)
            .groupby("datetime", as_index=False)
            .first()
            .sort_values("datetime")
            .reset_index(drop=True)
        )
        agg["plant_id"] = plant_id
        agg = agg.merge(sizes, on="datetime", how="left")
        out[plant_id] = agg
    if not out:
        raise SystemExit(f"No plant prediction files matched: {prediction_glob}")
    return out


def maybe_compute_ops_signals(df: pd.DataFrame) -> pd.DataFrame:
    if "ops_risk" in df.columns:
        return df
    required = {"datetime", "hab_prob", "p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med", "sst", "chlor_a", "kd490", "nflh"}
    if not required.issubset(df.columns):
        return df
    module = _load_ops_module()
    if module is None or not hasattr(module, "compute_ops_signals"):
        return df
    try:
        out = module.compute_ops_signals(df.copy())
        return out
    except Exception:
        return df


def _load_ops_module() -> Any | None:
    global _OPS_MODULE
    if _OPS_MODULE is not None:
        return _OPS_MODULE
    module_path = Path(__file__).resolve().parents[1] / "viewer" / "build_ops_payload.py"
    if not module_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("validation_ops_payload", module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _OPS_MODULE = module
    return module


def nearest_prediction_row(df: pd.DataFrame, target_ts: pd.Timestamp) -> pd.Series | None:
    if df.empty or pd.isna(target_ts):
        return None
    deltas = (df["datetime"] - target_ts).abs()
    if deltas.empty:
        return None
    idx = deltas.idxmin()
    return df.loc[idx]


def day_delta(value_a: pd.Timestamp, value_b: pd.Timestamp) -> float | None:
    if pd.isna(value_a) or pd.isna(value_b):
        return None
    return abs((value_a - value_b).total_seconds()) / 86400.0


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isfinite(value):
            return float(value)
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_default)
        f.write("\n")


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"auroc": None, "auprc": None, "n": int(len(y_true))}
    if len(y_true) == 0:
        return out
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    finite = np.isfinite(y_score)
    y_true = y_true[finite]
    y_score = y_score[finite]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return out
    counts = np.bincount(y_true, minlength=2)
    if len(y_true) < 5 or counts.min() < 2:
        return out
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        out["auroc"] = float(roc_auc_score(y_true, y_score))
        out["auprc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        pass
    return out


def confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    finite = np.isfinite(y_score)
    y_true = y_true[finite]
    y_score = y_score[finite]
    pred = (y_score >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
    }
