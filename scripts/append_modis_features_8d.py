#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, re
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import xarray as xr
from pyproj import Transformer

# ────────────────────────────────────────────────────────────────────────────────
# Filename → date
DATE_ONE   = re.compile(r"(\d{8})")
DATE_RANGE = re.compile(r"(\d{8})[_\-](\d{8})")

def mid_date_from_name(p: Path) -> datetime | None:
    """
    Accepts names like:
      AQUA_MODIS.20190218_20190225.L3m.8D.CHL.chlor_a.4km.nc
      TERRA_MODIS.20190906_20190913.L3m.8D.CHL.chlor_a.4km.nc
    Returns midpoint date (naive). Falls back to first YYYYMMDD if no range.
    """
    m2 = DATE_RANGE.search(p.name)
    if m2:
        d0 = datetime.strptime(m2.group(1), "%Y%m%d")
        d1 = datetime.strptime(m2.group(2), "%Y%m%d")
        return d0 + (d1 - d0)/2
    m1 = DATE_ONE.search(p.name)
    if m1:
        return datetime.strptime(m1.group(1), "%Y%m%d")
    return None

# ────────────────────────────────────────────────────────────────────────────────
# Parse chip datetime
def parse_chip_dt(s: str) -> datetime | None:
    s = (s or "").replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        try:
            return datetime.fromisoformat(s[:10])  # YYYY-MM-DD
        except Exception:
            return None

# ────────────────────────────────────────────────────────────────────────────────
# UTM → lon/lat (auto zone from Sentinel tile/scene_id; Oman default EPSG:32639)
UTM_TILE_PAT = re.compile(r"T(?P<zone>\d{2})(?P<band>[C-HJ-NP-X])", re.I)

def infer_epsg_from_id(id_text: str | None) -> int | None:
    if not id_text:
        return None
    m = UTM_TILE_PAT.search(id_text)
    if not m:
        return None
    zone = int(m.group("zone"))
    band = m.group("band").upper()
    north = band >= "N"  # N..X are northern hemisphere
    return (32600 if north else 32700) + zone

def centroid_lonlat(bounds, tile: str | None, scene_id: str | None):
    xmin, ymin, xmax, ymax = bounds
    x = (xmin + xmax) * 0.5
    y = (ymin + ymax) * 0.5

    looks_like_degrees = (-180.0 <= x <= 180.0) and (-90.0 <= y <= 90.0)
    looks_like_meters  = (abs(x) > 1_000.0) or (abs(y) > 1_000.0)

    if looks_like_degrees and not looks_like_meters:
        return float(x), float(y)

    epsg = infer_epsg_from_id(tile) or infer_epsg_from_id(scene_id) or 32639  # Oman fallback
    tfm = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon, lat = tfm.transform(x, y)
    return float(lon), float(lat)

# ────────────────────────────────────────────────────────────────────────────────
# Find MODIS product files (works with mixed AQUA/TERRA roots)
def list_product_files(root: Path, product_key: str) -> list[Path]:
    out = []
    for p in root.rglob("*.nc"):
        s = p.as_posix().lower()
        if product_key == "chlor_a":
            if ("8d" in s or ".l3m.8d." in s) and ("chl" in s and "chlor_a" in s):
                out.append(p)
        elif product_key == "kd_490" or product_key == "Kd_490":
            if ("8d" in s or ".l3m.8d." in s) and ("kd" in s and ("kd_490" in s or "kd-490" in s)):
                out.append(p)
        elif product_key == "nflh":
            if ("8d" in s or ".l3m.8d." in s) and ("flh" in s and "nflh" in s):
                out.append(p)
    return sorted(out)

# ────────────────────────────────────────────────────────────────────────────────
# NetCDF sampling (monotonic-safe + NaN-tolerant)
def _find_lonlat(ds: xr.Dataset):
    lon_candidates = ["lon", "longitude", "x"]
    lat_candidates = ["lat", "latitude", "y"]
    lon = next((c for c in lon_candidates if c in ds.coords or c in ds.data_vars), None)
    lat = next((c for c in lat_candidates if c in ds.coords or c in ds.data_vars), None)
    is2d = False
    if lon and lat:
        a = ds[lon]; b = ds[lat]
        is2d = (a.ndim == 2 or b.ndim == 2)
    return lon, lat, is2d

def _index_1d_monotonic(vals: np.ndarray, x: float) -> int:
    """Nearest index for x in a 1D monotonic array (ascending or descending)."""
    vals = np.asarray(vals)
    n = vals.size
    if n <= 1:
        return 0
    asc = bool(vals[-1] > vals[0])
    if asc:
        idx = np.searchsorted(vals, x)
        if idx <= 0: return 0
        if idx >= n: return n - 1
        return int(idx if abs(vals[idx]-x) < abs(vals[idx-1]-x) else idx-1)
    else:
        nv, nx = -vals, -x
        idx = np.searchsorted(nv, nx)
        if idx <= 0: return 0
        if idx >= n: return n - 1
        return int(idx if abs(nv[idx]-nx) < abs(nv[idx-1]-nx) else idx-1)

def _finite_nearby(a: np.ndarray, i_lat: int, j_lon: int, max_px: int, mode: str):
    """Search/average finite values around (i_lat, j_lon) in a growing window."""
    H, W = a.shape[-2], a.shape[-1]
    if mode == "nearest":
        for r in range(0, max_px+1):
            i0 = max(i_lat-r, 0); i1 = min(i_lat+r, H-1)
            j0 = max(j_lon-r, 0); j1 = min(j_lon+r, W-1)
            # top/bottom rows
            for j in range(j0, j1+1):
                v = a[..., i0, j];  # top
                if np.isfinite(v): return float(v)
                v = a[..., i1, j];  # bottom
                if np.isfinite(v): return float(v)
            # left/right cols
            for i in range(i0+1, i1):
                v = a[..., i, j0]
                if np.isfinite(v): return float(v)
                v = a[..., i, j1]
                if np.isfinite(v): return float(v)
        return None
    else:  # mean
        for r in range(0, max_px+1):
            i0 = max(i_lat-r, 0); i1 = min(i_lat+r, H-1)
            j0 = max(j_lon-r, 0); j1 = min(j_lon+r, W-1)
            box = a[..., i0:i1+1, j0:j1+1]
            m = np.nanmean(box)
            if np.isfinite(m): return float(m)
        return None

def sample_nc(path: Path, varname: str, lon_q: float, lat_q: float,
              px_radius: int = 5, mode: str = "mean") -> float | None:
    """
    Return value at (lon_q,lat_q) from L3m file (NaN-tolerant).
    mode: 'nearest' or 'mean' (nanmean in growing window up to px_radius).
    """
    try:
        ds = xr.open_dataset(path, mask_and_scale=True, decode_cf=True)
    except Exception:
        return None
    try:
        if varname not in ds:
            return None
        v = ds[varname]
        lon_name, lat_name, is2d = _find_lonlat(ds)
        if not lon_name or not lat_name:
            return None

        lon_vals = np.asarray(ds[lon_name].values)
        lat_vals = np.asarray(ds[lat_name].values)

        # normalize query lon into dataset range
        if np.nanmin(lon_vals) >= 0 and lon_q < 0:
            lon_q = (lon_q + 360.0) % 360.0
        if np.nanmax(lon_vals) <= 180 and lon_q > 180:
            lon_q = ((lon_q + 180.0) % 360.0) - 180.0

        # 1D fast path (typical for L3m mapped SMI)
        if not is2d and ds[lon_name].ndim == 1 and ds[lat_name].ndim == 1:
            i_lat = _index_1d_monotonic(lat_vals, lat_q)  # rows
            j_lon = _index_1d_monotonic(lon_vals, lon_q)  # cols
            arr = np.asarray(v.values)  # (lat, lon)
            val = arr[i_lat, j_lon]
            if np.isfinite(val):
                return float(val)
            return _finite_nearby(arr, i_lat, j_lon, px_radius, mode)
        else:
            # 2D curvilinear fallback
            LON = np.asarray(ds[lon_name].values)
            LAT = np.asarray(ds[lat_name].values)
            stride = max(1, int(round(max(LON.shape)/256)))
            ii = np.arange(0, LON.shape[0], stride)  # lat idx
            jj = np.arange(0, LON.shape[1], stride)  # lon idx
            subd = np.hypot(LON[np.ix_(ii, jj)] - lon_q, LAT[np.ix_(ii, jj)] - lat_q)
            k = int(np.argmin(subd))
            i0, j0 = ii[k // len(jj)], jj[k % len(jj)]  # (lat,lon)
            arr = np.asarray(v.values)
            val = arr[i0, j0]
            if np.isfinite(val):
                return float(val)
            return _finite_nearby(arr, i0, j0, px_radius, mode)
    except Exception:
        return None
    finally:
        try: ds.close()
        except Exception: pass

# ────────────────────────────────────────────────────────────────────────────────
# CSV processing
def process_csv(chips_csv: Path, modis_root: Path, max_days: int,
                products_to_use: list[str], debug: bool=False,
                px_radius: int = 5, sample_mode: str = "mean"):
    rows = list(csv.DictReader(open(chips_csv)))
    if not rows:
        if debug: print(f"[skip] {chips_csv}: no rows")
        return

    # Do we already have bounds in the file? If not, read them from sibling index.csv
    has_bounds = all(k in rows[0] for k in ("xmin","ymin","xmax","ymax"))
    bounds_lut = {}
    if not has_bounds:
        idx = chips_csv.parent / "index.csv"
        if not idx.exists():
            raise SystemExit(f"Missing bounds and index.csv at {idx}")
        with open(idx) as f:
            for r in csv.DictReader(f):
                try:
                    bounds_lut[r["tile"]] = (float(r["xmin"]), float(r["ymin"]),
                                             float(r["xmax"]), float(r["ymax"]))
                except: pass

    # Gather files per product (supports AQUA and TERRA mixed)
    files_by_prod = {p: list_product_files(modis_root, p) for p in products_to_use}
    if debug:
        for p, lst in files_by_prod.items():
            print(f"[debug] product={p} files={len(lst)} root={modis_root}")

    # Cache mid dates
    mid_cache = {p: [(mid_date_from_name(f), f) for f in files]
                 for p, files in files_by_prod.items()}

    def nearest_file(prod: str, target: datetime):
        best = None; bestd = 10**9
        for d,f in mid_cache.get(prod, []):
            if not d: continue
            dd = abs((d.date() - target.date()).days)
            if dd < bestd:
                bestd, best = dd, f
        return (best, bestd) if best is not None else (None, None)

    # NetCDF var names and CSV output column names
    var_of = {"chlor_a":"chlor_a", "Kd_490":"Kd_490", "kd_490":"Kd_490", "nflh":"nflh"}
    col_of = {"chlor_a":"chlor_a", "Kd_490":"kd490", "kd_490":"kd490", "nflh":"flh"}

    # Build header (keep old cols, ensure our output cols exist)
    base_fields = list(rows[0].keys())
    for p in products_to_use:
        if col_of[p] not in base_fields:
            base_fields.append(col_of[p])

    out_rows = []
    for r in rows:
        # bounds
        if has_bounds:
            try:
                bounds = (float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"]))
            except Exception:
                bounds = bounds_lut.get(r.get("tile"))
        else:
            bounds = bounds_lut.get(r.get("tile"))
        if bounds is None:
            out_rows.append(r); continue

        dt = parse_chip_dt(r.get("datetime",""))
        if dt is None:
            out_rows.append(r); continue

        lon, lat = centroid_lonlat(bounds, r.get("tile"), r.get("scene_id"))

        if debug:
            print(f"[debug] CSV={chips_csv}")
            print(f"        lon,lat={lon:.6f},{lat:.6f}  dt={dt}")

        rr = dict(r)
        for prod in products_to_use:
            nc, dgap = nearest_file(prod, dt)
            if debug:
                print(f"        {prod}: nearest={nc.name if nc else None} (Δ={dgap}d)")
            if nc is None or (dgap is not None and dgap > max_days):
                if debug:
                    print(f"        {prod}: skip (Δ={dgap}d > max_days={max_days})")
                continue

            varname = var_of[prod]
            val = sample_nc(nc, varname, lon, lat,
                            px_radius=px_radius, mode=sample_mode)
            if debug:
                print(f"        {prod}: sample={val}")
            if val is not None:
                rr[col_of[prod]] = f"{val:.6g}"
        out_rows.append(rr)

    with open(chips_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base_fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"✓ updated {chips_csv} ({len(out_rows)} rows)")

# ────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips_csv_glob", required=True,
        help='e.g. "data/aerial_*_2019/chip_indices_clean.csv" or ".../index.csv"')
    ap.add_argument("--modis_root", required=True,
        help="Folder with 8-day L3m files (AQUA and/or TERRA).")
    ap.add_argument("--max_days", type=int, default=30,
        help="Max temporal gap allowed between chip date and 8-day composite mid-date.")
    ap.add_argument("--products", nargs="+", default=["chlor_a"],
        choices=["chlor_a","Kd_490","nflh"])
    ap.add_argument("--px_radius", type=int, default=7,
        help="Search/average radius (in MODIS pixels).")
    ap.add_argument("--sample_mode", choices=["nearest","mean"], default="mean",
        help="Pick nearest finite value or nanmean in window.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    files = sorted(Path().glob(args.chips_csv_glob))
    if not files:
        raise SystemExit("No chips CSVs matched.")

    root = Path(args.modis_root)
    for c in files:
        process_csv(c, root, args.max_days, args.products,
                    debug=args.debug, px_radius=args.px_radius,
                    sample_mode=args.sample_mode)

if __name__ == "__main__":
    main()
