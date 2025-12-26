#!/usr/bin/env python3
from __future__ import annotations
import argparse, calendar
from pathlib import Path

# Aqua-MODIS L3m monthly, 4km, correct product/variable pairs
PRODUCTS = {
    "chlor_a": ("CHL", "chlor_a"),
    "Kd_490":  ("KD",  "Kd_490"),
    "nflh":    ("FLH", "nflh"),
}

def file_names(year0:int, year1:int, product:str, var:str):
    for y in range(year0, year1+1):
        for m in range(1, 12+1):
            d1 = f"{y}{m:02d}01"
            last = calendar.monthrange(y, m)[1]
            d2 = f"{y}{m:02d}{last:02d}"
            # AQUA_MODIS.YYYYMMDD_YYYYMMDD.L3m.MO.<PROD>.<VAR>.4km.nc
            yield f"AQUA_MODIS.{d1}_{d2}.L3m.MO.{product}.{var}.4km.nc"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, required=True, metavar=("Y0","Y1"))
    ap.add_argument("--outdir", default="filelists")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    y0, y1 = args.years

    paths = []
    for key, (prod, var) in PRODUCTS.items():
        lst = outdir / f"filelist_{key}.txt"
        with open(lst, "w") as f:
            for name in file_names(y0, y1, prod, var):
                # just the filename; downloader will prepend /ob/getfile/
                f.write(name + "\n")
        paths.append(lst)
        print(f"{key}: wrote {lst} ({sum(1 for _ in open(lst))} entries)")
    print("✔ wrote:\n" + "\n".join(str(p) for p in paths))

if __name__ == "__main__":
    main()
