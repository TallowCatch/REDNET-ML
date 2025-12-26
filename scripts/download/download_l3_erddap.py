#!/usr/bin/env python3
from __future__ import annotations
import argparse, datetime as dt, os, sys, time
from pathlib import Path
from urllib.parse import quote
import urllib.request

def month_iter(y0: int, y1: int):
    d = dt.date(y0, 1, 1)
    stop = dt.date(y1, 12, 31)
    while d <= stop:
        yield d.replace(day=15)  # use 15th as monthly tag
        # advance one month
        d = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1)

def fetch(url: str, out_path: Path, tries=3, sleep=1.5):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return "skip"
    last_err = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r, open(out_path, "wb") as f:
                f.write(r.read())
            return "ok"
        except Exception as e:
            last_err = e
            time.sleep(sleep * (k + 1))
    raise RuntimeError(f"download failed after {tries} tries: {url}\n{last_err}")

def main():
    ap = argparse.ArgumentParser(description="Download monthly L3 rasters from an ERDDAP griddap endpoint as GeoTIFFs.")
    ap.add_argument("--erddap", default="https://coastwatch.pfeg.noaa.gov/erddap",
                    help="Base ERDDAP URL (no trailing /griddap).")
    ap.add_argument("--dataset", required=True,
                    help="ERDDAP dataset id, e.g. 'AQUA_MODIS_L3m_CHL_monthly_4km' (varies by server).")
    ap.add_argument("--var", required=True, help="Variable name inside the dataset, e.g. 'chlor_a', 'KD490', 'FLH'.")
    ap.add_argument("--years", nargs=2, type=int, required=True, metavar=("Y0","Y1"),
                    help="Inclusive year span, e.g. 2017 2024")
    ap.add_argument("--bbox", default="51.5,60.8,15.5,26.5",
                    help="minlon,maxlon,minlat,maxlat (default = Oman coastal box).")
    ap.add_argument("--out", default="data/modis_l3", help="Output root folder.")
    ap.add_argument("--lon360", action="store_true",
                    help="If dataset uses 0..360 longitudes, convert bbox longitudes by adding 360.")
    args = ap.parse_args()

    minlon, maxlon, minlat, maxlat = map(float, args.bbox.split(","))
    if args.lon360:
        minlon += 360.0
        maxlon += 360.0

    root = Path(args.out) / args.var
    base = args.erddap.rstrip("/") + "/griddap/" + args.dataset

    n_ok = n_sk = n_err = 0
    for d in month_iter(args.years[0], args.years[1]):
        # ERDDAP expects ISO8601 with Z; monthly datasets will snap to the month
        tstr = d.strftime("%Y-%m-15T00:00:00Z")
        # Order in ERDDAP is usually: time][lat][lon] with [min:max] ranges
        # Use geotiff writer:
        query = (f"{quote(args.var)}[{tstr}][({minlat}:{maxlat})][({minlon}:{maxlon})]")
        url = f"{base}.geotiff?{query}"
        # nice filename
        out_name = f"{args.dataset}_{args.var}_{d.strftime('%Y%m')}.tif"
        out_path = root / str(d.year) / out_name
        try:
            status = fetch(url, out_path)
            if status == "ok":
                n_ok += 1
                print(f"✓ {out_name}")
            else:
                n_sk += 1
        except Exception as e:
            n_err += 1
            print(f"✗ {out_name} :: {e}", file=sys.stderr)

    print(f"\nDone. ok={n_ok} skipped={n_sk} errors={n_err} → {root}")

if __name__ == "__main__":
    main()
