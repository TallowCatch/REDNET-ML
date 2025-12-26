# scripts/s2_compute_chip_indices.py
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from pystac_client import Client
import planetary_computer as pc

S2_COLLECTION = "sentinel-2-l2a"

def fetch_item(stac: Client, scene_id: str):
    """Return a signed pystac.Item for a given scene_id, or None if not found."""
    coll = stac.search(collections=[S2_COLLECTION], ids=[scene_id]).item_collection()
    items = list(coll)  # ItemCollection is iterable
    if not items:
        return None
    return pc.sign(items[0])

def read_band_window(item, key: str, bounds, out_h=128, out_w=128):
    """Safe band read over chip bounds; clamps to raster extent."""
    if key not in item.assets:
        return None, None
    href = item.assets[key].href
    with rasterio.open(href) as ds:
        T = ds.transform
        H, W = ds.height, ds.width
        xmin, ymin, xmax, ymax = bounds
        if xmin > xmax: xmin, xmax = xmax, xmin
        if ymin > ymax: ymin, ymax = ymax, ymin
        r0, c0 = rasterio.transform.rowcol(T, xmin, ymax)  # UL (note y=max)
        r1, c1 = rasterio.transform.rowcol(T, xmax, ymin)  # LR (note y=min)
        r0 = max(0, min(H - 1, r0)); r1 = max(0, min(H - 1, r1))
        c0 = max(0, min(W - 1, c0)); c1 = max(0, min(W - 1, c1))
        rmin, rmax = sorted([r0, r1]); cmin, cmax = sorted([c0, c1])
        height = rmax - rmin + 1
        width  = cmax - cmin + 1
        if height <= 0 or width <= 0:
            return None, None
        win = rasterio.windows.Window(cmin, rmin, width, height)
        arr = ds.read(1, window=win, out_shape=(out_h, out_w)).astype(np.float32)
        msk = ds.read_masks(1, window=win, out_shape=(out_h, out_w))
    return arr, msk

def read_scl_window(item, bounds, out_h=128, out_w=128):
    """Read SCL (scene classification) over bounds; returns uint8 or None."""
    if "SCL" not in item.assets:
        return None
    href = item.assets["SCL"].href
    with rasterio.open(href) as ds:
        T = ds.transform
        H, W = ds.height, ds.width
        xmin, ymin, xmax, ymax = bounds
        if xmin > xmax: xmin, xmax = xmax, xmin
        if ymin > ymax: ymin, ymax = ymax, ymin
        r0, c0 = rasterio.transform.rowcol(T, xmin, ymax)
        r1, c1 = rasterio.transform.rowcol(T, xmax, ymin)
        r0 = max(0, min(H - 1, r0)); r1 = max(0, min(H - 1, r1))
        c0 = max(0, min(W - 1, c0)); c1 = max(0, min(W - 1, c1))
        rmin, rmax = sorted([r0, r1]); cmin, cmax = sorted([c0, c1])
        height = rmax - rmin + 1
        width  = cmax - cmin + 1
        if height <= 0 or width <= 0:
            return None
        win = rasterio.windows.Window(cmin, rmin, width, height)
        scl = ds.read(1, window=win, out_shape=(out_h, out_w)).astype(np.uint8)
    return scl


def robust_stats(x, mask):
    """Mean/Std over valid (mask>0 & finite). Returns (mean,std,count_valid)."""
    valid = (mask > 0) & np.isfinite(x)
    if valid.sum() == 0:
        return np.nan, np.nan, 0
    return float(x[valid].mean()), float(x[valid].std()), int(valid.sum())

def process_folder(folder: str):
    folder = Path(folder)
    index_csv = folder / "index.csv"
    out_csv   = folder / "chip_indices.csv"
    out_csv_tmp = folder / "chip_indices.tmp.csv"

    if not index_csv.exists():
        raise SystemExit(f"Missing {index_csv}")

    stac = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", ignore_conformance=True)

    # read all chip rows
    rows = []
    with open(index_csv, "r", newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)

    # cache items to avoid repeated network calls
    cache = {}

    with open(out_csv_tmp, "w", newline="") as f:
        fieldnames = [
            "tile","scene_id","datetime",
            "ndwi_mean","ndwi_std","fai_mean","fai_std",
            "rednir_mean","rednir_std","valid_px"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for r in rows:
            scene_id = r["scene_id"]
            bounds = (float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"]))

            # get (or fetch) signed item
            item = cache.get(scene_id)
            if item is None:
                item = fetch_item(stac, scene_id)
                cache[scene_id] = item
            if item is None:
                # skip if not resolvable
                continue

            # read bands needed:
            #  - NDWI uses G (B03) & NIR (B08)
            #  - FAI uses approx 560 (B03), 665 (B04), 865 (B08)
            #  - red/NIR uses B04 / B08
            B03, M03 = read_band_window(item, "B03", bounds)
            B04, M04 = read_band_window(item, "B04", bounds)
            B08, M08 = read_band_window(item, "B08", bounds)
            SCL      = read_scl_window(item, bounds)

            if B03 is None or B04 is None or B08 is None:
                continue

            scale = 10000.0
            G   = B03 / scale
            R   = B04 / scale
            NIR = B08 / scale

            # base data mask from GDAL masks
            M_data = (M03 > 0) & (M04 > 0) & (M08 > 0)

            # SCL-based masks (Sentinel-2 L2A conventions)
            # 6 = WATER, 7/8/9 = CLOUD prob, 10 = THIN_CIRRUS, 11 = SNOW
            if SCL is not None:
                SCL_cloudy = np.isin(SCL, [7, 8, 9, 10, 11])
                SCL_water  = (SCL == 6)
                # prefer pure water; if none, use "not cloudy" as fallback
                M = M_data & SCL_water
                if M.sum() == 0:
                    M = M_data & (~SCL_cloudy)
            else:
                M = M_data

            # final emergency fallback so you don't get all-NaN rows
            if M.sum() == 0:
                M = np.isfinite(G) & np.isfinite(R) & np.isfinite(NIR)

            # ---- indices
            ndwi = (G - NIR) / (G + NIR + 1e-6)
            ndwi_mean, ndwi_std, valid_px = robust_stats(ndwi, M)

            w1, w2, w0 = 560.0, 865.0, 665.0
            baseline = G + (NIR - G) * ((w0 - w1) / (w2 - w1))
            fai = R - baseline
            fai_mean, fai_std, _ = robust_stats(fai, M)

            rednir = R / (NIR + 1e-6)
            rednir_mean, rednir_std, _ = robust_stats(rednir, M)
            # ---- write output row
            w.writerow({
                "tile": r["tile"],
                "scene_id": scene_id,
                "datetime": r.get("datetime",""),
                "ndwi_mean": ndwi_mean,
                "ndwi_std": ndwi_std,
                "fai_mean": fai_mean,
                "fai_std": fai_std,
                "rednir_mean": rednir_mean,
                "rednir_std": rednir_std,
                "valid_px": valid_px
            })

    out_csv_tmp.replace(out_csv)
    print(f"✓ Wrote {out_csv}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Folder containing index.csv and tiles_png/")
    args = ap.parse_args()
    process_folder(args.folder)
