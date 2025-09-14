# src/oman_clip.py
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import os, random

# Oman bbox
LON_MIN, LON_MAX = 52.0, 60.5
LAT_MIN, LAT_MAX = 16.0, 26.5

IN_DIR  = Path("data/chl_tiles")
OUT_DIR = Path("data/chl_tiles/oman_clipped")
CSV_PATH = Path("data/labels/regression.csv")

random.seed(42)

def open_nc(path):
    for eng in ("netcdf4", "h5netcdf"):
        try:
            return xr.open_dataset(path, engine=eng)
        except Exception:
            pass
    raise RuntimeError(f"Failed to open {path} with netcdf4/h5netcdf")

def find_coords(ds):
    lon_name = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
    lat_name = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
    if not lon_name or not lat_name:
        raise ValueError(f"No lon/lat coords in {list(ds.coords)}")
    return lon_name, lat_name

def find_chl_var(ds):
    for k in ds.data_vars:
        lk = k.lower()
        if "chlor_a" in lk or lk.startswith("chlor"):
            return k
    raise ValueError(f"No chlorophyll variable among {list(ds.data_vars)}")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(IN_DIR.glob("*.nc"))
    if not files:
        print("No .nc files found in data/chl_tiles/")
        return

    rows = []
    for p in files:
        ds = open_nc(p)
        lon_name, lat_name = find_coords(ds)
        chl_name = find_chl_var(ds)

        # Handle ascending/descending latitude
        lat_vals = ds[lat_name].values
        lat_asc = (lat_vals[-1] > lat_vals[0])

        if lat_asc:
            clipped = ds.sel({lat_name: (slice(LAT_MIN, LAT_MAX)),
                              lon_name: (slice(LON_MIN, LON_MAX))})
        else:
            clipped = ds.sel({lat_name: (slice(LAT_MAX, LAT_MIN)),
                              lon_name: (slice(LON_MIN, LON_MAX))})

        if clipped[chl_name].size == 0:
            print(f"Skip {p.name}: no data in bbox")
            continue

        out_nc = OUT_DIR / p.name
        clipped.to_netcdf(out_nc)

        chl_mean = float(np.nanmean(clipped[chl_name].values))
        rows.append((str(out_nc), chl_mean))

    # 70/15/15 split
    random.shuffle(rows)
    n = len(rows)
    n_tr = int(0.7*n)
    n_va = int(0.15*n)
    splits = (["train"]*n_tr) + (["val"]*n_va) + (["test"]*(n-n_tr-n_va))

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w") as f:
        f.write("filepath,split,chl\n")
        for (fp, chl), sp in zip(rows, splits):
            f.write(f"{fp},{sp},{chl:.4f}\n")

    print(f"Wrote {len(rows)} clipped files to {OUT_DIR}")
    print(f"Wrote labels to {CSV_PATH}")

if __name__ == "__main__":
    main()