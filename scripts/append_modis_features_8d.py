#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, re
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import xarray as xr
import rasterio
from pyproj import Transformer

# ── filename date helpers ───────────────────────────────────────────────────────
DATE_ONE   = re.compile(r"(\d{8})")
DATE_RANGE = re.compile(r"(\d{8})[_\-](\d{8})")

def mid_date_from_name(p: Path) -> datetime | None:
    m2 = DATE_RANGE.search(p.name)
    if m2:
        d0 = datetime.strptime(m2.group(1), "%Y%m%d")
        d1 = datetime.strptime(m2.group(2), "%Y%m%d")
        return d0 + (d1 - d0)/2
    m1 = DATE_ONE.search(p.name)
    if m1:
        return datetime.strptime(m1.group(1), "%Y%m%d")
    return None

# ── product file finder (works with per-product temp dirs) ─────────────────────
PRODUCTS = {
    "chlor_a": ("CHL", "chlor_a"),
    "Kd_490":  ("KD",  "Kd_490"),
    "nflh":    ("FLH", "nflh"),
}

def list_product_files(root: Path, product_key: str) -> list[Path]:
    # allow either flat dir with only that product, or shared root/aqua_8d/<prod>/
    candidates = []
    for p in root.rglob("*.nc"):
        low = p.as_posix().lower()
        if product_key == "chlor_a" and ("chl" in low and "chlor_a" in low):
            candidates.append(p)
        elif product_key == "Kd_490" and ("kd" in low and ("kd_490" in low or "kd-490" in low)):
            candidates.append(p)
        elif product_key == "nflh" and ("flh" in low and "nflh" in low):
            candidates.append(p)
    return sorted(candidates)

# ── chip helpers ───────────────────────────────────────────────────────────────
def parse_chip_dt(s: str) -> datetime | None:
    s = (s or "").replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        try:
            return datetime.fromisoformat(s[:10])
        except Exception:
            return None

def chip_centroid(bounds):
    xmin, ymin, xmax, ymax = bounds
    return ((xmin + xmax)*0.5, (ymin + ymax)*0.5)

# ── NetCDF sampler (fast for 1-D lon/lat; works for 2-D too) ───────────────────
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

def _nearest_index_1d(vals: np.ndarray, x: float) -> int:
    idx = np.searchsorted(vals, x)
    if idx <= 0: return 0
    if idx >= vals.size: return vals.size - 1
    return int(idx if abs(vals[idx]-x) < abs(vals[idx-1]-x) else idx-1)

def sample_nc(path: Path, varname: str, lon_q: float, lat_q: float) -> float | None:
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
        # normalize longitude into dataset range
        if np.nanmin(lon_vals) >= 0 and lon_q < 0:
            lon_q = (lon_q + 360.0) % 360.0

        if not is2d and ds[lon_name].ndim == 1 and ds[lat_name].ndim == 1:
            i = _nearest_index_1d(np.asarray(ds[lon_name]), lon_q)
            j = _nearest_index_1d(np.asarray(ds[lat_name]), lat_q)
            out = v.isel({v.dims[-2]: j, v.dims[-1]: i}).values  # dims usually (lat,lon)
        else:
            LON = np.asarray(ds[lon_name])
            LAT = np.asarray(ds[lat_name])
            stride = max(1, int(round(max(LON.shape)/256)))
            ii = np.arange(0, LON.shape[0], stride)
            jj = np.arange(0, LON.shape[1], stride)
            sub = np.hypot(LON[np.ix_(ii, jj)]-lon_q, LAT[np.ix_(ii, jj)]-lat_q)
            k = int(np.argmin(sub))
            i0, j0 = ii[k // len(jj)], jj[k % len(jj)]
            i1 = slice(max(i0-4,0), min(i0+5, LON.shape[0]))
            j1 = slice(max(j0-4,0), min(j0+5, LON.shape[1]))
            d = np.hypot(LON[i1, j1]-lon_q, LAT[i1, j1]-lat_q)
            di, dj = np.unravel_index(np.argmin(d), d.shape)
            out = v.isel({v.dims[-2]: (i1.start or 0)+di,
                          v.dims[-1]: (j1.start or 0)+dj}).values

        val = float(np.asarray(out).squeeze())
        return val if np.isfinite(val) else None
    except Exception:
        return None
    finally:
        try: ds.close()
        except Exception: pass

# ── main per-CSV pass ──────────────────────────────────────────────────────────
def process_csv(chips_csv: Path, modis_root: Path, max_days: int, products_to_use: list[str]):
    rows = list(csv.DictReader(open(chips_csv)))
    if not rows:
        print(f"[skip] {chips_csv}: no rows")
        return

    # bounds lookup
    has_bounds = all(k in rows[0] for k in ("xmin","ymin","xmax","ymax"))
    bounds_lut = {}
    if not has_bounds:
        idx = chips_csv.parent/"index.csv"
        if not idx.exists():
            raise SystemExit(f"Missing bounds and index.csv at {idx}")
        with open(idx) as f:
            for r in csv.DictReader(f):
                try:
                    bounds_lut[r["tile"]] = (float(r["xmin"]), float(r["ymin"]),
                                             float(r["xmax"]), float(r["ymax"]))
                except: pass

    # collect available files by product (works even if only one product exists)
    files_by_prod = {p: list_product_files(modis_root, p) for p in products_to_use}
    for p,k in list(files_by_prod.items()):
        if not k: print(f"[warn] no files for {p} under {modis_root}")

    # cache file mid-dates for quick nearest lookup
    mid_cache = {p: [(mid_date_from_name(f), f) for f in files]
                 for p, files in files_by_prod.items()}

    def nearest_file(prod: str, target: datetime):
        best = None; bestd = 10**9
        for d,f in mid_cache.get(prod, []):
            if not d: continue
            dd = abs((d.date()-target.date()).days)
            if dd < bestd:
                bestd, best = dd, f
        return best if (best is not None and bestd <= max_days) else None

    # variable names per product
    var_of = {"chlor_a":"chlor_a", "Kd_490":"Kd_490", "nflh":"nflh"}

    # overwrite in place
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

        lon, lat = chip_centroid(bounds)
        dt = parse_chip_dt(r.get("datetime",""))
        if dt is None:
            out_rows.append(r); continue

        rr = dict(r)
        for prod in products_to_use:
            nc = nearest_file(prod, dt)
            if nc is None:
                continue
            val = sample_nc(nc, var_of[prod], lon, lat)
            if val is not None:
                # ensure the column exists and is overwritten
                rr[prod] = f"{val:.6g}"
        out_rows.append(rr)

    with open(chips_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys() | set(products_to_use)))
        w.writeheader()
        w.writerows(out_rows)
    print(f"✓ updated {chips_csv} ({len(out_rows)} rows)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips_csv_glob", required=True,
                    help='e.g. "data/aerial_*_20*/chip_indices_clean.csv"')
    ap.add_argument("--modis_root", required=True,
                    help="Path with downloaded 8D files (can be a temp folder).")
    ap.add_argument("--max_days", type=int, default=30)
    ap.add_argument("--products", nargs="+", default=["chlor_a","Kd_490","nflh"],
                    choices=["chlor_a","Kd_490","nflh"])
    args = ap.parse_args()

    files = sorted(Path().glob(args.chips_csv_glob))
    if not files:
        raise SystemExit("No chips CSVs matched.")
    for c in files:
        process_csv(c, Path(args.modis_root), args.max_days, args.products)

if __name__ == "__main__":
    main()
