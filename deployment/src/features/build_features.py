# deployment/src/build_features.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    features_ordered: List[str]
    dtypes: Dict[str, str]
    default_float: float = 0.0
    default_int: int = 0


def _to_month_num(month_key: str) -> int:
    # "YYYY-MM" -> 1..12 (month)
    if not isinstance(month_key, str) or "-" not in month_key:
        return 0
    try:
        _, m = month_key.split("-")
        m = int(m)
        return m if 1 <= m <= 12 else 0
    except Exception:
        return 0


def _safe_log(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.log(np.clip(x.astype(float), eps, None))


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return a.astype(float) / np.clip(b.astype(float), eps, None)


def build_features(raw_df: pd.DataFrame, cfg: FeatureConfig, month_key_col: str = "month_key") -> pd.DataFrame:
    df = raw_df.copy()

    # --- ensure month_key exists ---
    if month_key_col not in df.columns:
        df[month_key_col] = "0000-00"

    # --- base numeric columns that may be used ---
    base_cols = [
        "nflh", "chlor_a", "kd490", "sst", "sst_anom", "sst_anom_z", "sst_clim_rm",
        "fai_mean", "ndwi_mean", "ndwi_std", "rednir_mean", "rednir_std",
        "p_frcnn_r50_med", "p_frcnn_mb_med", "p_ssd_mb_med", "p_tab",
    ]
    for c in base_cols:
        if c not in df.columns:
            df[c] = np.nan

    # coerce to numeric
    for c in base_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- month features ---
    df["month_num"] = df[month_key_col].astype(str).map(_to_month_num).astype(int)
    # cyclical encoding
    m = df["month_num"].astype(float).values
    # if month_num=0 => sin/cos become 0/1-ish; we’ll just set both to 0 for month_num=0
    month_sin = np.sin(2 * np.pi * (m - 1) / 12.0)
    month_cos = np.cos(2 * np.pi * (m - 1) / 12.0)
    month_sin = np.where(df["month_num"].values == 0, 0.0, month_sin)
    month_cos = np.where(df["month_num"].values == 0, 0.0, month_cos)
    df["month_sin"] = month_sin
    df["month_cos"] = month_cos

    # --- logs ---
    df["log_nflh"] = _safe_log(df["nflh"].values)
    df["log_chlor_a"] = _safe_log(df["chlor_a"].values)
    df["log_kd490"] = _safe_log(df["kd490"].values)

    # --- ratios ---
    df["ratio_nflh_kd"] = _safe_div(df["nflh"].values, df["kd490"].values)
    df["ratio_chl_kd"] = _safe_div(df["chlor_a"].values, df["kd490"].values)

    # --- interactions/products ---
    df["chl_times_nflh"] = df["chlor_a"].values * df["nflh"].values
    df["sst_anom_x_kd490"] = df["sst_anom"].values * df["kd490"].values
    df["sst_anom_x_chlor_a"] = df["sst_anom"].values * df["chlor_a"].values
    df["sst_anom_x_fai_mean"] = df["sst_anom"].values * df["fai_mean"].values
    df["sst_anom_x_nflh"] = df["sst_anom"].values * df["nflh"].values
    df["sst_anom_x_month_sin"] = df["sst_anom"].values * df["month_sin"].values
    df["sst_anom_x_month_cos"] = df["sst_anom"].values * df["month_cos"].values

    # --- fill NaNs according to cfg ---
    out = pd.DataFrame(index=df.index)
    for feat in cfg.features_ordered:
        if feat not in df.columns:
            out[feat] = np.nan
        else:
            out[feat] = df[feat]

    # apply dtype + fillna
    for feat in cfg.features_ordered:
        dtype = cfg.dtypes.get(feat, "float")
        if dtype in ("float", "float64"):
            out[feat] = pd.to_numeric(out[feat], errors="coerce").fillna(cfg.default_float).astype(float)
        else:
            out[feat] = pd.to_numeric(out[feat], errors="coerce").fillna(cfg.default_int).astype(int)

    return out
