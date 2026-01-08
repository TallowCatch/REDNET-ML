#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import hashlib
import re
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr

# ─────────────────────────────────────────────────────────────
# utilities
# ─────────────────────────────────────────────────────────────
def stable_rng(key: str):
    h = hashlib.sha1(key.encode()).digest()
    seed = int.from_bytes(h[:4], "little")
    return np.random.default_rng(seed)

def meters_to_deg(lat_deg, dx_m, dy_m):
    # dy -> latitude, dx -> longitude
    dlat = dy_m / 111_000.0
    dlon = dx_m / (111_000.0 * math.cos(math.radians(lat_deg)))
    return dlat, dlon

def clean_float(x):
    if x is None:
        return None
    try:
        if not np.isfinite(x):
            return None
    except Exception:
        return None
    return float(x)

def wrap_lon_180(lon):
    """Convert lon to [-180, 180) for mapping tools that prefer it."""
    if lon is None:
        return None
    lon = float(lon)
    while lon >= 180.0:
        lon -= 360.0
    while lon < -180.0:
        lon += 360.0
    return lon

def maybe_to_360(lon, ds_lon):
    """
    HYCOM often uses lon in [0, 360). If dataset lon max > 180, we operate in 0..360.
    """
    lon = float(lon)
    try:
        lon_max = float(np.nanmax(ds_lon.values))
    except Exception:
        lon_max = 180.0

    if lon_max > 180.0:
        # dataset expects 0..360
        if lon < 0.0:
            lon = lon + 360.0
    return lon

# ─────────────────────────────────────────────────────────────
# HYCOM access
# ─────────────────────────────────────────────────────────────
HYCOM_OPENDAP = "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/uv3z"

def open_hycom():
    """
    Important: decode_times=True can fail because some variables (e.g. tau) have
    non-CF units like 'hours since analysis'. So we open with decode_times=False
    and decode the 'time' coord ourselves.
    """
    print("🌊 Opening HYCOM dataset (decode_times=False)")
    ds = xr.open_dataset(HYCOM_OPENDAP, decode_times=False)

    # Some HYCOM servers return lon/lat as 'lon'/'lat'. We assume that.
    if "time" not in ds:
        raise ValueError("HYCOM dataset missing 'time' coordinate")

    return ds

_TIME_UNITS_RE = re.compile(
    r"^\s*(seconds|minutes|hours|days)\s+since\s+(.+?)\s*$",
    re.IGNORECASE
)

def decode_hycom_time_to_unix_seconds(ds: xr.Dataset) -> np.ndarray:
    """
    Convert HYCOM ds['time'] numeric coord into Unix seconds using its CF-ish units.

    Example units: "hours since 2000-01-01 00:00:00"
    """
    t = ds["time"].values.astype("float64")
    units = ds["time"].attrs.get("units", "")

    m = _TIME_UNITS_RE.match(units or "")
    if not m:
        raise ValueError(
            f"Unrecognized HYCOM time units: {units!r}. "
            "Expected like 'hours since YYYY-MM-DD hh:mm:ss'."
        )

    unit = m.group(1).lower()
    base_str = m.group(2).strip()

    # Robust parse (assume UTC if no TZ)
    base = pd.to_datetime(base_str, utc=True, errors="raise").to_pydatetime()
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)

    # multipliers to seconds
    if unit == "seconds":
        mult = 1.0
    elif unit == "minutes":
        mult = 60.0
    elif unit == "hours":
        mult = 3600.0
    elif unit == "days":
        mult = 86400.0
    else:
        raise ValueError(f"Unsupported time unit: {unit}")

    base_unix = base.timestamp()
    unix_seconds = base_unix + t * mult

    print(f"🕒 HYCOM time units: {units}")
    print(f"✅ HYCOM time epoch: {datetime.fromtimestamp(base_unix, tz=timezone.utc).isoformat()}")

    return unix_seconds

def nearest_time_index_from_unix(unix_seconds_array: np.ndarray, target_datetime_utc: pd.Timestamp) -> int:
    target_unix = target_datetime_utc.to_pydatetime().timestamp()
    return int(np.argmin(np.abs(unix_seconds_array - target_unix)))

# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser("HAB risk plume advection (HYCOM + diffusion)")
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--plant_lat", type=float, required=True)
    ap.add_argument("--plant_lon", type=float, required=True)
    ap.add_argument("--out_geojson", required=True)
    ap.add_argument("--out_risk_csv", required=True)

    ap.add_argument("--n_particles", type=int, default=400)
    ap.add_argument("--base_diffusion_m", type=float, default=400)
    ap.add_argument("--max_diffusion_m", type=float, default=4000)
    ap.add_argument("--max_drift_m", type=float, default=12000)  # CLAMP

    args = ap.parse_args()

    # ── Load inference
    df = pd.read_csv(args.in_csv)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    # ── HYCOM
    ds = open_hycom()
    hycom_unix = decode_hycom_time_to_unix_seconds(ds)

    # normalize plant lon to HYCOM lon convention if needed
    plant_lon_hycom = maybe_to_360(args.plant_lon, ds["lon"])
    plant_lat = float(args.plant_lat)

    print(f"🏭 Plant anchor used: lat={plant_lat:.6f}, lon={args.plant_lon:.6f} (HYCOM lon={plant_lon_hycom:.6f})")

    # ── Initialize particles AT PLANT
    particles = [
        {"id": i, "lat": plant_lat, "lon": plant_lon_hycom}
        for i in range(args.n_particles)
    ]

    features = []
    risk_rows = []

    # 8-day timestep
    dt = 8 * 24 * 3600

    for t, row in df.iterrows():
        prob = clean_float(row.hab_prob)
        if prob is None:
            continue

        # ── Diffusion scales with HAB risk
        diff = args.base_diffusion_m + prob * (
            args.max_diffusion_m - args.base_diffusion_m
        )

        # ── HYCOM current near the plant at the correct time
        ti = nearest_time_index_from_unix(hycom_unix, row.datetime)

        # Pick surface / first depth if present (more stable than nanmean across depth)
        # water_u/v dims commonly: time, depth, lat, lon
        u_da = ds["water_u"].isel(time=ti)
        v_da = ds["water_v"].isel(time=ti)

        if "depth" in u_da.dims:
            u_da = u_da.isel(depth=0)
        if "depth" in v_da.dims:
            v_da = v_da.isel(depth=0)

        u_val = u_da.sel(lat=plant_lat, lon=plant_lon_hycom, method="nearest").values
        v_val = v_da.sel(lat=plant_lat, lon=plant_lon_hycom, method="nearest").values

        u_val = clean_float(u_val) or 0.0
        v_val = clean_float(v_val) or 0.0

        cur_dx = float(np.clip(u_val * dt, -args.max_drift_m, args.max_drift_m))
        cur_dy = float(np.clip(v_val * dt, -args.max_drift_m, args.max_drift_m))

        # ── Advect particles (now actually advecting forward)
        for p in particles:
            rng = stable_rng(f"{p['id']}_{t}")

            dx = rng.normal(0, diff) + cur_dx
            dy = rng.normal(0, diff) + cur_dy

            # IMPORTANT: use particle's current latitude for lon scaling
            dlat, dlon = meters_to_deg(p["lat"], dx, dy)

            lat = p["lat"] + dlat
            lon = p["lon"] + dlon

            # Update particle state (this was missing before)
            p["lat"] = lat
            p["lon"] = lon

            # Convert lon back to [-180,180) for GeoJSON consumers like Kepler (safe even if already)
            lon_out = wrap_lon_180(lon)

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [clean_float(lon_out), clean_float(lat)],
                },
                "properties": {
                    "time": row.datetime.isoformat(),
                    "hab_prob": prob,
                    "month": str(row.month),
                    "particle_id": int(p["id"]),
                },
            })

        # ── Plant-level risk index
        risk_rows.append({
            "datetime": row.datetime.isoformat(),
            "hab_prob": prob,
            "risk_level": (
                "high" if prob >= 0.53 else
                "medium" if prob >= 0.39 else
                "low"
            ),
        })

    # ── Write outputs (VALID JSON)
    Path(args.out_geojson).write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2)
    )

    pd.DataFrame(risk_rows).to_csv(args.out_risk_csv, index=False)

    print("✅ HAB plume + plant risk complete")
    print(f"   → {args.out_geojson}")
    print(f"   → {args.out_risk_csv}")

if __name__ == "__main__":
    main()
