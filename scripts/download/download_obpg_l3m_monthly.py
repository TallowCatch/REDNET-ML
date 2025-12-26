#!/usr/bin/env python3
from __future__ import annotations
import argparse, datetime as dt, time, os, calendar
from pathlib import Path
import urllib.request

# Monthly L3m names commonly seen at OBPG:
#  A) AQUA_MODIS.YYYYMM01_YYYYMMDD.L3m.MO.CHL.x_chlor_a.nc
#  B) AQUA_MODIS.YYYYMMDD.L3m.MO.CHL.chlor_a.4km.nc  (older alt)
#
# We will try (A) first (your style), then (B) as a fallback.

PRODUCTS = {
    "chlor_a": ("CHL",  "chlor_a"),
    "KD490":   ("KD490","KD490"),
    "FLH":     ("FLH",  "FLH"),
}

BASE = "https://oceandata.sci.gsfc.nasa.gov/ob/getfile/"

def month_first_last(y:int, m:int) -> tuple[str,str]:
    first = f"{y:04d}{m:02d}01"
    last_day = calendar.monthrange(y, m)[1]
    last  = f"{y:04d}{m:02d}{last_day:02d}"
    return first, last

def month_dates(y0:int, y1:int):
    d = dt.date(y0, 1, 1)
    end = dt.date(y1, 12, 31)
    while d <= end:
        yield d.year, d.month
        # jump to first of next month
        d = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1)

def fetch(url: str, out: Path, tries=3, sleep=2.0):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return "skip"
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(out, "wb") as f:
                data = r.read()
                # crude guard: make sure not HTML
                if b"<html" in data[:200].lower():
                    raise RuntimeError("received HTML (likely auth required)")
                f.write(data)
            return "ok"
        except Exception as e:
            last = e
            time.sleep(sleep * (k + 1))
    raise RuntimeError(f"download failed: {url}\n{last}")

def main():
    ap = argparse.ArgumentParser(description="Download OBPG monthly L3m NetCDF (mapped)")
    ap.add_argument("--var", choices=PRODUCTS.keys(), required=True)
    ap.add_argument("--years", nargs=2, type=int, required=True, metavar=("Y0","Y1"))
    ap.add_argument("--out", default="data/chl_tiles/requested_files")
    args = ap.parse_args()

    appkey = os.environ.get("OBPG_APPKEY")
    if not appkey:
        raise SystemExit("Missing OBPG_APPKEY env var. Run: export OBPG_APPKEY=your_key")

    product, varname = PRODUCTS[args.var]
    y0, y1 = args.years
    root = Path(args.out) / args.var

    ok = sk = er = 0
    for y, m in month_dates(y0, y1):
        first, last = month_first_last(y, m)

        # Style A (your style)
        fnameA = f"AQUA_MODIS.{first}_{last}.L3m.MO.{product}.x_{varname}.nc"
        urlA   = f"{BASE}{fnameA}?appkey={appkey}"

        # Style B (fallback)
        fnameB = f"AQUA_MODIS.{first}.L3m.MO.{product}.{varname}.4km.nc"
        urlB   = f"{BASE}{fnameB}?appkey={appkey}"

        outA = (root / f"{y}"/ fnameA)
        outB = (root / f"{y}"/ fnameB)

        try:
            try:
                status = fetch(urlA, outA)
                if status == "ok": ok += 1
                else: sk += 1
                print(f"✓ {fnameA}")
            except Exception:
                status = fetch(urlB, outB)
                if status == "ok": ok += 1
                else: sk += 1
                print(f"✓ {fnameB} (fallback)")
        except Exception as e:
            er += 1
            print(f"✗ {y}-{m:02d} :: {e}")

    print(f"\nDone. ok={ok} skip={sk} err={er} → {root}")

if __name__ == "__main__":
    main()
