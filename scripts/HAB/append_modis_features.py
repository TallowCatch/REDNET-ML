#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, re, os, tempfile, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import rasterio
from pyproj import Transformer
import xarray as xr

# ── Filenames & dates ───────────────────────────────────────────────────────────
DATE_ONE   = re.compile(r"(\d{8})")                 # 20170401
DATE_RANGE = re.compile(r"(\d{8})[_-](\d{8})")      # 20170401_20170408

def parse_date_or_midpoint(p: Path) -> datetime | None:
    n = p.name
    m2 = DATE_RANGE.search(n)
    if m2:
        d0 = datetime.strptime(m2.group(1), "%Y%m%d")
        d1 = datetime.strptime(m2.group(2), "%Y%m%d")
        mid = d0 + (d1 - d0)/2
        return mid.replace(tzinfo=None)
    m1 = DATE_ONE.search(n)
    if m1:
        return datetime.strptime(m1.group(1), "%Y%m%d")
    return None

def list_product_files(root: Path, product_key: str) -> list[Path]:
    key = product_key.lower()
    # accept both MO and 8D names; CHL/FLH/KD490 variations
    variants = {
        "chlor": ("chlor_a", "chlora", ".chl.", "chla"),
        "kd490": ("kd_490", "kd490", "kd-490"),
        "flh":   ("nflh", "flh"),
    }[key]
    out = []
    for p in root.rglob("*.nc"):
        lp = p.as_posix().lower()
        if ".l3m." in lp and any(v in lp for v in variants):
            out.append(p)
    # GeoTIFFs if you ever add them:
    for p in root.rglob("*.tif*"):
        lp = p.as_posix().lower()
        if any(v in lp for v in variants):
            out.append(p)
    return sorted(out)

def nearest_by_date(files: list[Path], target_dt: datetime, max_days: int) -> Path | None:
    best, best_delta = None, 10**9
    for p in files:
        d = parse_date_or_midpoint(p)
        if not d: 
            continue
        delta = abs((d.date() - target_dt.date()).days)
        if delta < best_delta:
            best_delta, best = delta, p
    if best is None or best_delta > max_days:
        return None
    return best

# ── Sampling helpers ────────────────────────────────────────────────────────────
def sample_geotiff_at_lonlat(path: Path, lon: float, lat: float) -> float | None:
    with rasterio.open(path) as ds:
        if ds.crs is None:
            return None
        if str(ds.crs).endswith("4326"):
            x, y = lon, lat
        else:
            tfm = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
            x, y = tfm.transform(lon, lat)
        row, col = ds.index(x, y)
        if not (0 <= row < ds.height and 0 <= col < ds.width):
            return None
        val = ds.read(1, window=rasterio.windows.Window(col, row, 1, 1), boundless=True)
        msk = ds.read_masks(1, window=rasterio.windows.Window(col, row, 1, 1), boundless=True)
        if msk.size == 0 or msk[0, 0] == 0:
            return None
        v = float(val[0, 0])
        return v if np.isfinite(v) else None

def _find_lonlat_names(ds: xr.Dataset) -> tuple[str | None, str | None, bool]:
    lon_candidates = ["lon", "longitude", "x"]
    lat_candidates = ["lat", "latitude", "y"]
    lon_name = next((c for c in lon_candidates if c in ds.coords), None)
    lat_name = next((c for c in lat_candidates if c in ds.coords), None)
    is_2d = False
    if lon_name is None or lat_name is None:
        for c in lon_candidates:
            if c in ds and ds[c].ndim == 2:
                lon_name, is_2d = c, True
                break
        for c in lat_candidates:
            if c in ds and ds[c].ndim == 2:
                lat_name, is_2d = c, True
                break
    else:
        is_2d = (ds[lon_name].ndim == 2 or ds[lat_name].ndim == 2)
    return lon_name, lat_name, is_2d

def _find_product_var(ds: xr.Dataset, product_label: str) -> str | None:
    pl = product_label.lower()
    for k in ds.data_vars:
        lk = k.lower()
        if pl == "chlor" and ("chlor_a" in lk or lk == "chlor"):
            return k
        if pl == "flh" and ("nflh" in lk or lk == "flh" or "fluorescence" in lk):
            return k
        if pl == "kd490" and ("kd_490" in lk or "kd490" in lk or lk == "kd"):
            return k
    for k in ds.data_vars:
        if "palette" not in k.lower():
            return k
    return None

def _nearest_index_1d(vals: np.ndarray, x: float) -> int:
    idx = np.searchsorted(vals, x)
    if idx == 0: return 0
    if idx >= vals.size: return vals.size - 1
    return int(idx if abs(vals[idx] - x) < abs(vals[idx-1] - x) else idx-1)

def sample_netcdf_at_lonlat(path: Path, lon: float, lat: float, product_label: str) -> float | None:
    try:
        ds = xr.open_dataset(path, mask_and_scale=True, decode_cf=True)
    except Exception:
        return None
    try:
        var = _find_product_var(ds, product_label)
        if var is None:
            return None
        lon_name, lat_name, is_2d = _find_lonlat_names(ds)
        if lon_name is None or lat_name is None:
            return None

        lon_vals = ds[lon_name].values
        if np.nanmin(lon_vals) >= 0 and lon < 0:
            lon_q = (lon + 360.0) % 360.0
        else:
            lon_q = lon

        da = ds[var]
        if not is_2d and ds[lon_name].ndim == 1 and ds[lat_name].ndim == 1:
            lons = np.asarray(ds[lon_name].values)
            lats = np.asarray(ds[lat_name].values)
            if (np.diff(lons) < 0).any():
                v = da.sel({lon_name: lon_q, lat_name: lat}, method="nearest").values
            else:
                i = _nearest_index_1d(lons, lon_q)
                j = _nearest_index_1d(lats, lat)
                v = da.isel({lon_name: i, lat_name: j}).values
            v = float(np.asarray(v).squeeze())
            return v if np.isfinite(v) else None

        # 2-D curvilinear fallback
        LON2 = np.asarray(ds[lon_name].values)
        LAT2 = np.asarray(ds[lat_name].values)
        stride = max(1, int(round(max(LON2.shape)/256)))
        ii = np.arange(0, LON2.shape[0], stride)
        jj = np.arange(0, LON2.shape[1], stride)
        sub_lon, sub_lat = LON2[np.ix_(ii, jj)], LAT2[np.ix_(ii, jj)]
        dsub = np.hypot(sub_lon - lon_q, sub_lat - lat)
        k = np.argmin(dsub)
        i0, j0 = divmod(int(k), len(jj))
        i1 = slice(max(ii[i0]-4, 0), min(ii[i0]+5, LON2.shape[0]))
        j1 = slice(max(jj[j0]-4, 0), min(jj[j0]+5, LON2.shape[1]))
        dwin = np.hypot(LON2[i1, j1] - lon_q, LAT2[i1, j1] - lat)
        ki, kj = np.unravel_index(np.argmin(dwin), dwin.shape)
        vi = (i1.start or 0) + ki
        vj = (j1.start or 0) + kj
        v = da.isel({da.dims[-2]: vi, da.dims[-1]: vj}).values
        v = float(np.asarray(v).squeeze())
        return v if np.isfinite(v) else None
    except Exception:
        return None
    finally:
        try: ds.close()
        except Exception: pass

def sample_any(path: Path, lon: float, lat: float, label: str) -> float | None:
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        return sample_geotiff_at_lonlat(path, lon, lat)
    if ext == ".nc":
        return sample_netcdf_at_lonlat(path, lon, lat, label)
    return None

# ── Chip utils ─────────────────────────────────────────────────────────────────
def chip_centroid(b): 
    xmin, ymin, xmax, ymax = b
    return ((xmin + xmax)*0.5, (ymin + ymax)*0.5)

def load_bounds_lookup(index_csv: Path) -> dict[str, tuple[float,float,float,float]]:
    lut = {}
    with open(index_csv, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                lut[r["tile"]] = (float(r["xmin"]), float(r["ymin"]),
                                  float(r["xmax"]), float(r["ymax"]))
            except Exception:
                continue
    return lut

def parse_chip_dt(s: str) -> datetime | None:
    s = (s or "").replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        try: return datetime.fromisoformat(s[:10])
        except Exception: return None

# ── Main ───────────────────────────────────────────────────────────────────────
def process_csv(chips_csv: Path, modis_root: Path, max_days: int, out_csv: str | None):
    rows = list(csv.DictReader(open(chips_csv)))
    if not rows:
        print(f"[skip] No rows in {chips_csv}")
        return

    has_bounds = all(k in rows[0] for k in ("xmin","ymin","xmax","ymax"))
    bounds_lut = {}
    if not has_bounds:
        idx = chips_csv.parent / "index.csv"
        if not idx.exists():
            raise SystemExit(f"Missing bounds; index.csv not found at {idx}")
        bounds_lut = load_bounds_lookup(idx)

    chlor_files = list_product_files(modis_root, "chlor")
    kd_files    = list_product_files(modis_root, "kd490")
    flh_files   = list_product_files(modis_root, "flh")
    if not (chlor_files or kd_files or flh_files):
        raise SystemExit(f"No L3m files under {modis_root} (expect *.nc with 8D/MO chlor/kd490/flh).")

    out_rows = []
    for r in rows:
        b = (float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"])) if has_bounds else bounds_lut.get(r.get("tile"))
        if b is None: 
            continue
        lon, lat = chip_centroid(b)
        dt = parse_chip_dt(r.get("datetime", ""))
        if dt is None:
            continue

        def sample_nearest(files, label):
            p = nearest_by_date(files, dt, max_days)
            if p is None:
                return ""
            v = sample_any(p, lon, lat, label)
            return "" if (v is None) else f"{v:.6g}"

        out = dict(r)
        out["chlor_a"] = sample_nearest(chlor_files, "chlor")
        out["kd490"]   = sample_nearest(kd_files, "kd490")
        out["flh"]     = sample_nearest(flh_files, "flh")
        out_rows.append(out)

    if not out_rows:
        print(f"[warn] No rows written for {chips_csv}")
        return

    # overwrite the input file unless a different path is provided
    target = out_csv or str(chips_csv)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv", prefix="l3tmp_")
    os.close(tmp_fd)
    fieldnames = list(out_rows[0].keys())
    with open(tmp_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    shutil.move(tmp_path, target)
    print(f"✓ Updated {target} ({len(out_rows)} rows)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips_csv", default=None, help="One chips csv to update in-place")
    ap.add_argument("--chips_csv_glob", default=None, help="Glob over seasons (e.g., 'data/aerial_*_20*/chip_indices_clean.csv')")
    ap.add_argument("--modis_root", required=True, help="Folder with 8D L3m NetCDFs (e.g., data/l3/aqua_8d)")
    ap.add_argument("--max_days", type=int, default=30, help="Max gap (days) between chip date and L3m date")
    ap.add_argument("--out_csv", default=None, help="Optional explicit output (single-file mode only)")
    args = ap.parse_args()

    inputs = []
    if args.chips_csv_glob:
        inputs = [Path(p) for p in sorted(Path().glob(args.chips_csv_glob))]
    elif args.chips_csv:
        inputs = [Path(args.chips_csv)]
    else:
        ap.error("Provide --chips_csv or --chips_csv_glob")

    root = Path(args.modis_root)
    for p in inputs:
        process_csv(p, root, args.max_days, args.out_csv if len(inputs)==1 else None)

if __name__ == "__main__":
    main()
