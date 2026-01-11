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
from shapely.geometry import Point, Polygon, MultiPolygon, mapping
from shapely.ops import unary_union

# =========================
# HYCOM CONFIG
# =========================
HYCOM_OPENDAP = "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/uv3z"
GRID_DEG = 0.1  # coarse res for cached lookup grid

_TIME_UNITS_RE = re.compile(
    r"^\s*(seconds|minutes|hours|days)\s+since\s+(.+?)\s*$",
    re.IGNORECASE
)

# =========================
# Small utils
# =========================
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
    dlon = dx_m / (111_000.0 * max(1e-6, math.cos(math.radians(lat))))
    return dlat, dlon

def write_geojson(path: Path, features):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    print(f"✅ wrote {path} ({len(features)} features)")

# =========================
# HYCOM time decode
# =========================
def decode_hycom_time(ds: xr.Dataset) -> pd.DatetimeIndex:
    t = ds["time"].values.astype(float)
    units = ds["time"].attrs.get("units", "")

    m = _TIME_UNITS_RE.match(units)
    if not m:
        raise ValueError(f"Unsupported HYCOM time units: {units}")

    unit, base_str = m.groups()
    base = pd.to_datetime(base_str, utc=True)
    delta = pd.to_timedelta(t, unit=unit[0])  # s/m/h/d
    dt = pd.DatetimeIndex(base + delta)
    return dt

def nearest_time_index(times: pd.DatetimeIndex, t: pd.Timestamp) -> int:
    idx = times.searchsorted(t)
    if idx <= 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    return idx if (times[idx] - t) < (t - times[idx - 1]) else idx - 1

# =========================
# HYCOM cached presampling
# =========================
def presample_hycom(
    lat0: float,
    lon0: float,
    tmin: pd.Timestamp,
    tmax: pd.Timestamp,
    window_km: float,
    step_hours: int,
    hycom_stride: int,
    cache_npz: Path | None,
):
    """
    Loads a small HYCOM subset into memory and maps it to a coarse grid for fast sampling.
    Optionally caches to NPZ (so repeat runs don't re-download the same subset).
    """
    if cache_npz and cache_npz.exists():
        print(f"📦 Loading HYCOM cache: {cache_npz}")
        npz = np.load(cache_npz, allow_pickle=True)
        u_grid = npz["u_grid"]
        v_grid = npz["v_grid"]
        lat_grid = npz["lat_grid"]
        lon_grid = npz["lon_grid"]
        hycom_times = pd.to_datetime(npz["hycom_times"].astype(str), utc=True)
        print(f"✅ Loaded cached HYCOM: u_grid shape={u_grid.shape}")
        return u_grid, v_grid, lat_grid, lon_grid, hycom_times

    print("🌊 Opening HYCOM remote dataset (subset + presample)…")
    ds = xr.open_dataset(HYCOM_OPENDAP, decode_times=False, engine="pydap", chunks=None)
    times = decode_hycom_time(ds)
    approx_steps_per_year = 365 * 24 // step_hours
    tmax_steps = min(len(ds.time), int(approx_steps_per_year * 10))
    ds = ds.isel(time=slice(0, tmax_steps))
    times = times[:tmax_steps]

    # clip range to overlap
    tmin_clip = max(tmin, times[0])
    tmax_clip = min(tmax, times[-1])
    if tmin_clip >= tmax_clip:
        raise ValueError(
            f"No overlap between CSV and HYCOM times.\nCSV: {tmin} → {tmax}\nHYCOM:{times[0]} → {times[-1]}"
        )

    pad = pd.Timedelta(days=2)
    t0 = max(tmin_clip - pad, times[0])
    t1 = min(tmax_clip + pad, times[-1])

    step = pd.Timedelta(hours=int(step_hours))
    sim_times = pd.date_range(t0, t1, freq=step, tz="UTC")

    idx = times.searchsorted(sim_times)
    idx = np.clip(idx, 0, len(times) - 1)
    idx2 = np.clip(idx - 1, 0, len(times) - 1)

    use_idx = []
    for a, b, tt in zip(idx2, idx, sim_times):
        use_idx.append(b if (times[b] - tt) < (tt - times[a]) else a)

    use_idx = np.array(sorted(set(use_idx)), dtype=int)
    hycom_stride = max(1, int(hycom_stride))
    if hycom_stride > 1:
        use_idx = use_idx[::hycom_stride]

    # spatial bounds
    dlat = window_km * 1000 / 111_000
    dlon = window_km * 1000 / (111_000 * max(1e-6, math.cos(math.radians(lat0))))
    lat_min, lat_max = lat0 - dlat, lat0 + dlat
    lon_min, lon_max = lon0 - dlon, lon0 + dlon

    lat_grid = np.arange(lat_min, lat_max + GRID_DEG, GRID_DEG)
    lon_grid = np.arange(lon_min, lon_max + GRID_DEG, GRID_DEG)

    ds_sub = ds[["water_u", "water_v"]].isel(time=use_idx).sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max),
    )
    if "depth" in ds_sub["water_u"].dims:
        ds_sub = ds_sub.isel(depth=0)

    ds_sub = ds_sub.load()
    ds.close()

    lat_src = ds_sub["lat"].values
    lon_src = ds_sub["lon"].values

    lat_idx = np.abs(lat_src[:, None] - lat_grid[None, :]).argmin(axis=0)
    lon_idx = np.abs(lon_src[:, None] - lon_grid[None, :]).argmin(axis=0)

    u_src = ds_sub["water_u"].values
    v_src = ds_sub["water_v"].values

    u_grid = u_src[:, lat_idx[:, None], lon_idx[None, :]]
    v_grid = v_src[:, lat_idx[:, None], lon_idx[None, :]]

    hycom_times = times[use_idx]

    if cache_npz:
        cache_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_npz,
            u_grid=u_grid,
            v_grid=v_grid,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            hycom_times=np.array(hycom_times.astype(str), dtype=object),
        )
        print(f"📦 Saved HYCOM cache: {cache_npz}")

    return u_grid, v_grid, lat_grid, lon_grid, hycom_times

def sample_uv_grid(u_grid, v_grid, lat_grid, lon_grid, ti, lat, lon):
    i = np.clip(np.searchsorted(lat_grid, lat) - 1, 0, len(lat_grid) - 1)
    j = np.clip(np.searchsorted(lon_grid, lon) - 1, 0, len(lon_grid) - 1)
    u = clean_float(u_grid[ti, i, j])
    v = clean_float(v_grid[ti, i, j])
    return u, v

# =========================
# Particle model
# =========================
@dataclass
class Particle:
    pid: int
    lat: float
    lon: float
    source_hab: float
    survival: float
    coords: list
    times_ms: list

# =========================
# Density + envelope
# =========================
def km_to_deg_lat(km: float) -> float:
    return km / 111.0

def km_to_deg_lon(km: float, lat_deg: float) -> float:
    return km / (111.0 * max(1e-6, math.cos(math.radians(lat_deg))))

def largest_polygon(geom):
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(list(geom.geoms), key=lambda g: g.area, default=None)
    try:
        polys = [g for g in geom.geoms if isinstance(g, Polygon)]
        return max(polys, key=lambda g: g.area, default=None)
    except Exception:
        return None

def make_density_grid(points_lonlat: np.ndarray, grid_km: float, cap_points: int | None):
    pts = points_lonlat
    if cap_points and len(pts) > cap_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), size=cap_points, replace=False)
        pts = pts[idx]
        print(f"🔻 Density subsample: {len(points_lonlat)} → {len(pts)}")

    lons = pts[:, 0]
    lats = pts[:, 1]
    lat0 = float(np.median(lats))

    dlat = km_to_deg_lat(grid_km)
    dlon = km_to_deg_lon(grid_km, lat0)

    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())

    ix = np.floor((lons - lon_min) / dlon).astype(int)
    iy = np.floor((lats - lat_min) / dlat).astype(int)

    counts = {}
    for a, b in zip(iy, ix):
        counts[(int(a), int(b))] = counts.get((int(a), int(b)), 0) + 1

    total = float(len(pts))
    maxc = max(counts.values()) if counts else 1

    features = []
    for (cy, cx), c in counts.items():
        x0 = lon_min + cx * dlon
        x1 = x0 + dlon
        y0 = lat_min + cy * dlat
        y1 = y0 + dlat
        poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])

        features.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {
                "kind": "density_cell",
                "count": int(c),
                "p": float(c / total),
                "p_norm": float(c / float(maxc)),
                "grid_km": float(grid_km),
            }
        })
    return features

def make_envelope(points_lonlat: np.ndarray, buffer_m: float, simplify_m: float):
    lons = points_lonlat[:, 0]
    lats = points_lonlat[:, 1]
    lat0 = float(np.median(lats))

    buf_km = buffer_m / 1000.0
    simp_km = simplify_m / 1000.0

    dlat_buf = km_to_deg_lat(buf_km)
    dlon_buf = km_to_deg_lon(buf_km, lat0)
    buf_deg = float((dlat_buf + dlon_buf) / 2.0)

    dlat_s = km_to_deg_lat(simp_km)
    dlon_s = km_to_deg_lon(simp_km, lat0)
    simp_deg = float((dlat_s + dlon_s) / 2.0)

    pts_geom = [Point(float(x), float(y)) for x, y in points_lonlat]
    hull = unary_union(pts_geom).convex_hull

    blob = unary_union([p.buffer(buf_deg) for p in pts_geom])
    blob = largest_polygon(blob) or hull
    blob_s = blob.simplify(simp_deg, preserve_topology=True)
    blob_s = largest_polygon(blob_s) or blob

    return [
        {"type": "Feature", "geometry": mapping(hull),
         "properties": {"kind": "envelope", "type": "convex_hull"}},
        {"type": "Feature", "geometry": mapping(blob_s),
         "properties": {"kind": "envelope", "type": "buffer_union_envelope",
                        "buffer_m": float(buffer_m), "simplify_m": float(simplify_m)}}
    ]

# =========================
# Main pipeline
# =========================
def main():
    ap = argparse.ArgumentParser("Make a single Kepler-ready GeoJSON: trips + points + density + envelope")

    ap.add_argument("--in_csv", required=True, help="inference_all_months.csv")
    ap.add_argument("--plant_lat", type=float, required=True)
    ap.add_argument("--plant_lon", type=float, required=True)

    ap.add_argument("--out_geojson", required=True, help="single merged output GeoJSON")

    # Transport controls
    ap.add_argument("--window_km", type=float, default=100)
    ap.add_argument("--step_hours", type=int, default=24)
    ap.add_argument("--hycom_stride", type=int, default=3)

    ap.add_argument("--max_particles", type=int, default=3000, help="hard cap total active particles")
    ap.add_argument("--seed_per_obs", type=float, default=180.0, help="expected seeds at hab=1.0 (Poisson)")
    ap.add_argument("--seed_floor", type=float, default=0.0, help="subtract baseline risk before seeding")
    ap.add_argument("--min_seeds_if_positive", type=int, default=0, help="optional minimum seeds when hab > seed_floor")

    ap.add_argument("--half_life_days", type=float, default=5.0, help="survival half-life (days)")
    ap.add_argument("--min_survival", type=float, default=0.01, help="kill particle when survival drops below this")

    ap.add_argument("--base_diff_m", type=float, default=200)
    ap.add_argument("--risk_diff_m", type=float, default=1200)
    ap.add_argument("--max_drift_m", type=float, default=15000)

    # Aggregation controls
    ap.add_argument("--grid_km", type=float, default=2.0)
    ap.add_argument("--cap_points", type=int, default=150000, help="cap points for density; 0 disables")
    ap.add_argument("--buffer_m", type=float, default=1500.0)
    ap.add_argument("--simplify_m", type=float, default=300.0)

    # Caching
    ap.add_argument("--hycom_cache_npz", default="", help="optional NPZ cache path (recommended)")

    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if "datetime" not in df.columns or "hab_prob" not in df.columns:
        raise ValueError("CSV must contain at least: datetime, hab_prob")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    # Convert plant lon to HYCOM lon domain (0..360 if needed)
    ds0 = xr.open_dataset(HYCOM_OPENDAP, decode_times=False)
    plant_lon_hy = maybe_to_360(args.plant_lon, ds0["lon"])
    ds0.close()

    cache_npz = Path(args.hycom_cache_npz) if args.hycom_cache_npz.strip() else None

    u_grid, v_grid, lat_grid, lon_grid, hycom_times = presample_hycom(
        lat0=args.plant_lat,
        lon0=plant_lon_hy,
        tmin=df["datetime"].min(),
        tmax=df["datetime"].max(),
        window_km=args.window_km,
        step_hours=args.step_hours,
        hycom_stride=args.hycom_stride,
        cache_npz=cache_npz,
    )

    rng = np.random.default_rng(42)

    step_s = int(args.step_hours) * 3600
    decay = math.log(2) / (args.half_life_days * 86400.0)

    particles: list[Particle] = []
    finished: list[Particle] = []
    pid = 0

    # helper: stable “risk used for seeding”
    def hab_for_seeding(h: float) -> float:
        h = float(np.clip(h, 0.0, 1.0))
        if args.seed_floor > 0:
            h = max(0.0, h - args.seed_floor)
            h = h / max(1e-9, (1.0 - args.seed_floor))
        return float(np.clip(h, 0.0, 1.0))

    for i in range(len(df) - 1):
        tA = df.loc[i, "datetime"]
        tB = df.loc[i + 1, "datetime"]
        hab = clean_float(df.loc[i, "hab_prob"], 0.0)
        hab_s = hab_for_seeding(hab)

        # Poisson seeding (fixes your “int() => 0” problem)
        lam = hab_s * float(args.seed_per_obs)
        n_seed = int(rng.poisson(lam=lam)) if lam > 0 else 0
        if args.min_seeds_if_positive and hab_s > 0:
            n_seed = max(n_seed, int(args.min_seeds_if_positive))

        # cap by remaining capacity
        cap_left = max(0, int(args.max_particles) - len(particles))
        n_seed = min(n_seed, cap_left)

        for _ in range(n_seed):
            particles.append(Particle(
                pid=pid,
                lat=float(args.plant_lat),
                lon=float(plant_lon_hy),
                source_hab=float(hab),
                survival=1.0,  # survival is separate from hab_prob
                coords=[(wrap_lon_180(plant_lon_hy), float(args.plant_lat))],
                times_ms=[int(tA.timestamp() * 1000)]
            ))
            pid += 1

        # march forward between tA and tB
        t = tA
        ti = nearest_time_index(hycom_times, t)
        while t < tB and particles:
            t2 = min(t + pd.Timedelta(seconds=step_s), tB)
            dt = float((t2 - t).total_seconds())

            # advance ti (bounded)
            ti = min(ti + 1, len(hycom_times) - 1)

            new_particles = []
            for p in particles:
                # decay survival
                p.survival *= math.exp(-decay * dt)
                if p.survival < float(args.min_survival):
                    finished.append(p)
                    continue

                u, v = sample_uv_grid(u_grid, v_grid, lat_grid, lon_grid, ti, p.lat, p.lon)

                if p.pid == 0:
                    print(
                        "DEBUG:",
                        "ti=", ti,
                        "u=", u,
                        "v=", v,
                        "lon=", p.lon,
                        "lat=", p.lat
                    )

                # if HYCOM gives dead velocities, end trajectory
                # if abs(u) < 1e-8 and abs(v) < 1e-8:
                #     finished.append(p)
                #     continue

                dx = float(np.clip(u * dt, -args.max_drift_m, args.max_drift_m))
                dy = float(np.clip(v * dt, -args.max_drift_m, args.max_drift_m))

                # diffusion scaled by source risk (your scientific assumption)
                sig = float(args.base_diff_m + p.source_hab * args.risk_diff_m)
                dx += float(rng.normal(0.0, sig))
                dy += float(rng.normal(0.0, sig))

                dlat, dlon = meters_to_deg(p.lat, dx, dy)
                p.lat += dlat
                p.lon += dlon

                p.coords.append((wrap_lon_180(p.lon), float(p.lat)))
                p.times_ms.append(int(t2.timestamp() * 1000))

                new_particles.append(p)

            particles = new_particles
            t = t2

    finished.extend(particles)

    # Build Kepler-friendly features
    trip_features = []
    point_features = []

    for p in finished:
        if len(p.coords) > 1:
            trip_features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": p.coords},
                "properties": {
                    "kind": "trip",
                    "pid": int(p.pid),
                    "source_hab": float(p.source_hab),
                    "survival_end": float(p.survival),
                    "timestamps": p.times_ms,   # Kepler Trip layer uses this
                }
            })

        for (lon, lat), tms in zip(p.coords, p.times_ms):
            point_features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "kind": "particle_point",
                    "pid": int(p.pid),
                    "time": int(tms),           # Kepler time filter field
                    "source_hab": float(p.source_hab),
                    "survival": float(p.survival),
                }
            })

    print(f"🧾 trips={len(trip_features)} | points={len(point_features)}")

    if len(point_features) == 0:
        print("⚠️ No particle points were generated. Try:")
        print("   - increase --seed_per_obs (e.g. 400)")
        print("   - decrease --min_survival (e.g. 0.005)")
        print("   - reduce --seed_floor or set it to 0")
        out = trip_features  # still write whatever exists
        write_geojson(Path(args.out_geojson), out)
        return

    pts_arr = np.array([f["geometry"]["coordinates"] for f in point_features], dtype=float)

    cap = None if int(args.cap_points) == 0 else int(args.cap_points)
    density = make_density_grid(pts_arr, grid_km=float(args.grid_km), cap_points=cap)
    envelope = make_envelope(pts_arr, buffer_m=float(args.buffer_m), simplify_m=float(args.simplify_m))

    # Merge everything into ONE FeatureCollection (Kepler loads this cleanly)
    merged = []
    merged.extend(trip_features)
    merged.extend(point_features)
    merged.extend(density)
    merged.extend(envelope)

    write_geojson(Path(args.out_geojson), merged)
    print("✅ DONE — load this single GeoJSON into Kepler and build layers by filtering `kind`.")

if __name__ == "__main__":
    main()
