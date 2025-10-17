#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import date, timedelta

# Aqua-MODIS L3m 8-day, 4km, correct product/variable pairs
PRODUCTS = {
    "chlor_a": ("CHL", "chlor_a"),
    "Kd_490":  ("KD",  "Kd_490"),
    "nflh":    ("FLH", "nflh"),
}

def eight_day_bins(year: int):
    """Yield (start_date, end_date) for OBPG 8-day bins within a year.
       OBPG bins are anchored on Jan 1; last bin may be 5–7 days."""
    d0 = date(year, 1, 1)
    d1 = date(year, 12, 31)
    s = d0
    while s <= d1:
        e = min(s + timedelta(days=7), d1)
        yield s, e
        s = e + timedelta(days=1)

def file_names_8d(y0: int, y1: int, product: str, var: str):
    for y in range(y0, y1 + 1):
        for s, e in eight_day_bins(y):
            d1 = s.strftime("%Y%m%d")
            d2 = e.strftime("%Y%m%d")
            # AQUA_MODIS.YYYYMMDD_YYYYMMDD.L3m.8D.<PROD>.<VAR>.4km.nc
            yield f"AQUA_MODIS.{d1}_{d2}.L3m.8D.{product}.{var}.4km.nc"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, required=True, metavar=("Y0","Y1"))
    ap.add_argument("--outdir", default="filelists_8d")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    y0, y1 = args.years

    written = []
    for key, (prod, var) in PRODUCTS.items():
        lst = outdir / f"filelist_8d_{key}.txt"
        with open(lst, "w") as f:
            for name in file_names_8d(y0, y1, prod, var):
                f.write(name + "\n")
        written.append(lst)
        print(f"{key}: wrote {lst} ({sum(1 for _ in open(lst))} entries)")
    print("✔ wrote:\n" + "\n".join(str(p) for p in written))

if __name__ == "__main__":
    main()
