#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ============================================================
# CONFIG
# ============================================================
HYCOM_OPENDAP = "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/uv3z"
GRID_DEG = 0.1   # ← coarse grid resolution (degrees)

_TIME_UNITS_RE = re.compile(
    r"^\s*(seconds|minutes|hours|days)\s+since\s+(.+?)\s*$",
    re.IGNORECASE
)

# ============================================================
# UTILITIES
# ============================================================
def clean_float(x, default=0.0):
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def maybe_to_360(lon: float, ds_lon: xr.DataArray) -> float:
    lon = float(lon)
    if float(ds_lon.max()) > 180.0 and lon < 0:
        lon += 360.0
    return lon


def wrap_lon_180(lon: float) -> float:
    return ((float(lon) + 180) % 360) - 180


def meters_to_deg(lat, dx_m, dy_m):
    dlat = dy_m / 111_000.0
    dlon = dx_m / (111_000.0 * math.cos(math.radians(lat)))
    return dlat, dlon


# ============================================================
# HYCOM TIME
# ============================================================
def decode_hycom_time(ds: xr.Dataset) -> pd.DatetimeIndex:
    t = ds["time"].values.astype(float)
    units = ds["time"].attrs.get("units", "")

    m = _TIME_UNITS_RE.match(units)
    if not m:
        raise ValueError(f"Unsupported HYCOM time units: {units}")

    unit, base_str = m.groups()
    base = pd.to_datetime(base_str, utc=True)
    delta = pd.to_timedelta(t, unit=unit[0])

    dt = pd.DatetimeIndex(base + delta)
    print(f"🕒 HYCOM time steps: {len(dt)}")
    print(f"📆 HYCOM range: {dt[0]} → {dt[-1]}")
    return dt


def nearest_time_index(times: pd.DatetimeIndex, t: pd.Timestamp) -> int:
    idx = times.searchsorted(t)
    if idx <= 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    return idx if (times[idx] - t) < (t - times[idx - 1]) else idx - 1


# ============================================================
# DATA MODEL
# ============================================================
@dataclass
class Particle:
    pid: int
    lat: float
    lon: float
    weight: float
    coords: list
    times_ms: list


# ============================================================
# HYCOM ACCESS + PRE-SAMPLING
# ============================================================
def open_and_presample_hycom(lat0, lon0, tmin, tmax, window_km, step_hours, hycom_stride):
    print("🌊 Opening HYCOM (subset + presample)")
    print("   ⏬ Opening remote dataset…")

    ds = xr.open_dataset(
        HYCOM_OPENDAP,
        decode_times=False,
        engine="netcdf4",
        chunks=None
    )
    print("   ✅ Dataset opened")

    times = decode_hycom_time(ds)

    print(f"   🧾 CSV range: {tmin.isoformat()} → {tmax.isoformat()}")
    print(f"   🧾 HYCOM range: {times[0].isoformat()} → {times[-1].isoformat()}")

    # ----------------------------
    # 1) Clip to HYCOM overlap
    # ----------------------------
    tmin_clip = max(tmin, times[0])
    tmax_clip = min(tmax, times[-1])
    if tmin_clip >= tmax_clip:
        raise ValueError(
            f"No overlap between CSV times and HYCOM times.\n"
            f"CSV:  {tmin.isoformat()} → {tmax.isoformat()}\n"
            f"HYCOM:{times[0].isoformat()} → {times[-1].isoformat()}"
        )

    pad = pd.Timedelta(days=2)
    t0 = max(tmin_clip - pad, times[0])
    t1 = min(tmax_clip + pad, times[-1])

    # ----------------------------
    # 2) Build only the timestamps you will use
    # ----------------------------
    step = pd.Timedelta(hours=int(step_hours))
    sim_times = pd.date_range(t0, t1, freq=step, tz="UTC")
    # map to hycom indices (nearest)
    idx = times.searchsorted(sim_times)
    idx = np.clip(idx, 0, len(times) - 1)

    # fix nearest vs previous
    idx2 = np.clip(idx - 1, 0, len(times) - 1)
    use_idx = []
    for a, b, tt in zip(idx2, idx, sim_times):
        use_idx.append(b if (times[b] - tt) < (tt - times[a]) else a)

    use_idx = np.array(sorted(set(use_idx)), dtype=int)

    hycom_stride = max(1, int(hycom_stride))
    if hycom_stride > 1:
        before = len(use_idx)
        use_idx = use_idx[::hycom_stride]
        print(f"   🪓 Applying hycom_stride={hycom_stride}: {before} → {len(use_idx)} timesteps")


    print(f"   ⏱️ Will load only {len(use_idx)} HYCOM timesteps (instead of {len(times)})")
    print(f"      first loaded HYCOM time: {times[use_idx[0]].isoformat()}")
    print(f"      last  loaded HYCOM time: {times[use_idx[-1]].isoformat()}")

    # ----------------------------
    # 3) Spatial bounds
    # ----------------------------
    dlat = window_km * 1000 / 111_000
    dlon = window_km * 1000 / (111_000 * math.cos(math.radians(lat0)))

    lat_min, lat_max = lat0 - dlat, lat0 + dlat
    lon_min, lon_max = lon0 - dlon, lon0 + dlon

    lat_grid = np.arange(lat_min, lat_max + GRID_DEG, GRID_DEG)
    lon_grid = np.arange(lon_min, lon_max + GRID_DEG, GRID_DEG)

    print(f"🧊 Presampling grid: {len(lat_grid)} x {len(lon_grid)}")
    print(f"   🌍 Spatial subset lat={lat_min:.3f}..{lat_max:.3f} lon={lon_min:.3f}..{lon_max:.3f}")

    # ----------------------------
    # 4) Subset + load (SMALL now)
    # ----------------------------
    print("   ✂️ Subsetting time+space…")
    ds_sub = ds[["water_u", "water_v"]].isel(time=use_idx).sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max)
    )

    if "depth" in ds_sub["water_u"].dims:
        ds_sub = ds_sub.isel(depth=0)

    print("   ⏬ Forcing HYCOM subset into memory NOW…")
    print(f"      expected load arrays ~ (t={len(use_idx)}, lat≈{ds_sub.sizes.get('lat','?')}, lon≈{ds_sub.sizes.get('lon','?')})")
    ds_sub = ds_sub.load()
    print("   ✅ HYCOM subset loaded")

    lat_src = ds_sub["lat"].values
    lon_src = ds_sub["lon"].values

    print("   🧮 Building nearest-neighbour index maps…")
    lat_idx = np.abs(lat_src[:, None] - lat_grid[None, :]).argmin(axis=0)
    lon_idx = np.abs(lon_src[:, None] - lon_grid[None, :]).argmin(axis=0)

    print("   📦 Extracting u/v arrays…")
    u_src = ds_sub["water_u"].values
    v_src = ds_sub["water_v"].values

    print("   🔄 Mapping to coarse grid (pure NumPy)…")
    u_grid = u_src[:, lat_idx[:, None], lon_idx[None, :]]
    v_grid = v_src[:, lat_idx[:, None], lon_idx[None, :]]

    print("   ✅ Presampling complete")
    print(f"      u_grid shape = {u_grid.shape}")

    hycom_times_loaded = times[use_idx]

    return u_grid, v_grid, lat_grid, lon_grid, hycom_times_loaded




def sample_uv_grid(u_grid, v_grid, lat_grid, lon_grid, ti, lat, lon):
    i = np.clip(np.searchsorted(lat_grid, lat) - 1, 0, len(lat_grid) - 1)
    j = np.clip(np.searchsorted(lon_grid, lon) - 1, 0, len(lon_grid) - 1)
    return clean_float(u_grid[ti, i, j]), clean_float(v_grid[ti, i, j])


# ============================================================
# OUTPUTS
# ============================================================
def write_geojson(path, features):
    Path(path).write_text(json.dumps({
        "type": "FeatureCollection",
        "features": features
    }))
    print(f"✅ wrote {path} ({len(features)} features)")


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser("HAB HYCOM streaklines (cached grid)")
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--plant_lat", type=float, required=True)
    ap.add_argument("--plant_lon", type=float, required=True)
    ap.add_argument("--out_trips_geojson", required=True)
    ap.add_argument("--out_points_geojson", required=True)

    ap.add_argument("--window_km", type=float, default=100)
    ap.add_argument("--step_hours", type=int, default=12)
    ap.add_argument("--max_particles", type=int, default=500)
    ap.add_argument("--seed_per_obs", type=int, default=120)
    ap.add_argument("--hycom_stride", type=int, default=3,
                help="Presample every Nth HYCOM timestep (1=all, 2=every 2nd, 3=every 3rd)")

    ap.add_argument("--base_diff_m", type=float, default=200)
    ap.add_argument("--risk_diff_m", type=float, default=1200)
    ap.add_argument("--half_life_days", type=float, default=5)
    ap.add_argument("--max_drift_m", type=float, default=15000)

    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime")

    ds0 = xr.open_dataset(HYCOM_OPENDAP, decode_times=False)
    plant_lon_hy = maybe_to_360(args.plant_lon, ds0["lon"])
    ds0.close()

    u_grid, v_grid, lat_grid, lon_grid, hycom_times = open_and_presample_hycom(
        args.plant_lat,
        plant_lon_hy,
        df["datetime"].min(),
        df["datetime"].max(),
        args.window_km,
        args.step_hours,
        args.hycom_stride
    )



    step_s = args.step_hours * 3600
    decay = math.log(2) / (args.half_life_days * 86400)

    rng = np.random.default_rng(42)
    particles, finished = [], []
    pid = 0

    for i in range(len(df) - 1):
        tA, tB = df.iloc[i]["datetime"], df.iloc[i + 1]["datetime"]
        hab = clean_float(df.iloc[i]["hab_prob"])

        for _ in range(int(hab * args.seed_per_obs)):
            if len(particles) >= args.max_particles:
                break
            particles.append(Particle(
                pid=pid,
                lat=args.plant_lat,
                lon=plant_lon_hy,
                weight=hab,
                coords=[(wrap_lon_180(plant_lon_hy), args.plant_lat)],
                times_ms=[int(tA.timestamp() * 1000)]
            ))
            pid += 1

        t = tA
        while t < tB and particles:
            t2 = min(t + pd.Timedelta(seconds=step_s), tB)
            dt = (t2 - t).total_seconds()
            ti = min(ti + 1, len(hycom_times) - 1) if 'ti' in locals() else nearest_time_index(hycom_times, t)

            new = []
            for p in particles:
                p.weight *= math.exp(-decay * dt)
                if p.weight < 0.05:
                    finished.append(p)
                    continue

                u, v = sample_uv_grid(u_grid, v_grid, lat_grid, lon_grid, ti, p.lat, p.lon)

                dx = np.clip(u * dt, -args.max_drift_m, args.max_drift_m)
                dy = np.clip(v * dt, -args.max_drift_m, args.max_drift_m)

                dx += rng.normal(0, args.base_diff_m + hab * args.risk_diff_m)
                dy += rng.normal(0, args.base_diff_m + hab * args.risk_diff_m)

                dlat, dlon = meters_to_deg(p.lat, dx, dy)
                p.lat += dlat
                p.lon += dlon

                # crude land guard: if velocity collapses, kill particle
                if abs(u) < 1e-6 and abs(v) < 1e-6:
                    finished.append(p)
                    continue

                p.coords.append((wrap_lon_180(p.lon), p.lat))
                p.times_ms.append(int(t2.timestamp() * 1000))
                new.append(p)

            particles = new
            t = t2

    finished.extend(particles)

    trips = [{
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": p.coords},
        "properties": {"timestamps": p.times_ms}
    } for p in finished if len(p.coords) > 1]

    points = [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": c},
        "properties": {"time": t, "pid": p.pid}
    } for p in finished for c, t in zip(p.coords, p.times_ms)]


    write_geojson(args.out_trips_geojson, trips)
    write_geojson(args.out_points_geojson, points)

    print("✅ DONE — fast, cached, and scientifically clean")


if __name__ == "__main__":
    main()
