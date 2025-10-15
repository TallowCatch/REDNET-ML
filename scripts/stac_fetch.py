#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, warnings
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import xy, Affine
from shapely.geometry import shape, box, mapping, Polygon
from shapely.ops import unary_union, transform as shp_transform
from pystac_client import Client
import planetary_computer as pc
from pyproj import CRS, Transformer
import imageio.v2 as iio

warnings.filterwarnings("ignore", category=UserWarning)
WGS84 = CRS.from_epsg(4326)

# ---------- helpers ----------
def load_aoi(path: str):
    gj = json.loads(Path(path).read_text())
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    return unary_union([shape(f["geometry"]) for f in feats])

def search_items(aoi_geom, start, end, max_cloud, collection="sentinel-2-l2a", limit=20):
    stac = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", ignore_conformance=True)
    q = stac.search(
        collections=[collection],
        datetime=f"{start}/{end}",
        intersects=mapping(aoi_geom),
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=limit,
    )
    return [pc.sign(it) for it in q.item_collection()]

def raster_poly(transform, width, height) -> Polygon:
    x0, y0 = xy(transform, 0, 0, offset="ul")
    x1, y1 = xy(transform, height - 1, width - 1, offset="lr")
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

def window_poly(transform, win: Window) -> Polygon:
    xs = [win.col_off, win.col_off + win.width, win.col_off + win.width, win.col_off]
    ys = [win.row_off, win.row_off, win.row_off + win.height, win.row_off + win.height]
    pts = [~transform * (x, y) for x, y in zip(xs, ys)]
    return box(min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))

def scaled_transform(base: Affine, in_w: int, in_h: int, out_w: int, out_h: int) -> Affine:
    sx = in_w / float(out_w); sy = in_h / float(out_h)
    return Affine(base.a * sx, base.b, base.c, base.d, base.e * sy, base.f)

# ---------- RGB + SCL readers ----------
def read_rgb_window(item, win: Window, out_size: int):
    """Read RGB window as 3xHxW array"""
    bands = []
    transform_out = None; crs = None
    for key in ("B04", "B03", "B02"):
        with rasterio.open(item.assets[key].href) as r:
            arr = r.read(1, window=win, out_shape=(out_size, out_size)).astype(np.float32)
            t_win = rasterio.windows.transform(win, r.transform)
            transform_out = scaled_transform(t_win, win.width, win.height, out_size, out_size)
            crs = r.crs
        bands.append(arr)
    stack = np.stack(bands, 0)
    v = stack.reshape(3, -1)
    p2, p98 = np.percentile(v, (2, 98))
    stack = np.clip((stack - p2) / (p98 - p2 + 1e-6), 0, 1) * 255
    return stack.astype(np.uint8), transform_out, crs

def read_scl(item):
    if "SCL" not in item.assets:
        return None
    with rasterio.open(item.assets["SCL"].href) as r:
        return r.read(1)

# ---------- chipping ----------
def chip_stream(
    item, aoi_item, size=640, stride=256,
    scl_water_min_frac=0.01, debug=False,
    jitter_px=64,            # <— NEW: up to ±64 px random shift
    extra_scales=(1.0, 0.8)  # <— NEW: also crop at 80% size
):
    with rasterio.open(item.assets["B03"].href) as r:
        H, W = r.height, r.width
        T = r.transform
        img_poly = raster_poly(T, W, H)

    clip = aoi_item.intersection(img_poly)
    if clip.is_empty:
        if debug: print("  debug: AOI∩image empty")
        return

    scl = read_scl(item)

    from rasterio.transform import rowcol
    xmin, ymin, xmax, ymax = clip.bounds
    r0, c0 = rowcol(T, xmin, ymax)
    r1, c1 = rowcol(T, xmax, ymin)
    rmin, rmax = sorted([max(0, min(H-1, r0)), max(0, min(H-1, r1))])
    cmin, cmax = sorted([max(0, min(W-1, c0)), max(0, min(W-1, c1))])

    # pad to be safe, then make a stride grid
    pad = size
    rmin = max(0, rmin - pad); rmax = min(H, rmax + pad)
    cmin = max(0, cmin - pad); cmax = min(W, cmax + pad)
    start_r = (rmin // stride) * stride
    start_c = (cmin // stride) * stride

    rng = np.random.default_rng()

    def passes_water(r, c, s):
        if scl is None: 
            return True
        sub = scl[r:r+s, c:c+s]
        return (sub.size > 0) and (float((sub == 6).mean()) >= scl_water_min_frac)

    wrote_local = 0
    for r_base in range(start_r, max(0, rmax - size + 1) + 1, stride):
        for c_base in range(start_c, max(0, cmax - size + 1) + 1, stride):
            for scale in extra_scales:
                s = int(round(size * scale))
                # jitter around the base anchor
                dr = int(rng.integers(-jitter_px, jitter_px+1))
                dc = int(rng.integers(-jitter_px, jitter_px+1))
                r = max(0, min(H - s, r_base + dr))
                c = max(0, min(W - s, c_base + dc))
                win = Window(c, r, s, s)
                wpoly = window_poly(T, win)
                if not wpoly.intersects(clip):
                    continue
                if not passes_water(r, c, s):
                    continue
                rgb, _, _ = read_rgb_window(item, win, out_size=size)  # upscale smaller crops to `size`
                if rgb.max() == 0:
                    continue
                wrote_local += 1
                yield np.transpose(rgb, (1, 2, 0)), wpoly.bounds

    # fallback: centered chip (also try scales)
    if wrote_local == 0:
        cx = (xmin + xmax) * 0.5; cy = (ymin + ymax) * 0.5
        rr, cc = rowcol(T, cx, cy)
        for scale in extra_scales:
            s = int(round(size * scale))
            r = max(0, min(H - s, rr - s // 2))
            c = max(0, min(W - s, cc - s // 2))
            if not passes_water(r, c, s): 
                continue
            win = Window(c, r, s, s)
            rgb, _, _ = read_rgb_window(item, win, out_size=size)
            if rgb.max() > 0:
                if debug: print("  debug: forced center chip")
                yield np.transpose(rgb, (1, 2, 0)), window_poly(T, win).bounds

    if debug:
        print(f"  debug chip: wrote_local={wrote_local}")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end",   default="2023-12-31")
    ap.add_argument("--cloud", type=float, default=40)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--out",   default="data/aerial")
    ap.add_argument("--size",  type=int, default=640)
    ap.add_argument("--stride",type=int, default=256)
    ap.add_argument("--scl_water_min_frac", type=float, default=0.01)
    ap.add_argument("--save_previews", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    aoi_ll = load_aoi(args.aoi)
    out_root = Path(args.out)
    tiles_dir = out_root / "tiles_png"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    index_csv = out_root / "index.csv"

    items = search_items(aoi_ll, args.start, args.end, args.cloud, "sentinel-2-l2a", args.limit)
    print(f"Found {len(items)} candidate items.")

    all_rows = []
    for it in items:
        try:
            if not shape(it.geometry).intersects(aoi_ll):
                continue
        except Exception:
            pass

        # AOI reprojection
        with rasterio.open(it.assets["B03"].href) as src:
            crs_item = src.crs
        tfm = Transformer.from_crs(WGS84, crs_item, always_xy=True)
        aoi_item = shp_transform(lambda x, y: tfm.transform(x, y), aoi_ll)

        meta = {
            "scene_id": it.id,
            "datetime": str(it.datetime),
            "collection": it.collection_id,
            "cloudcov": it.properties.get("eo:cloud_cover", -1),
        }

        wrote = 0
        for chip, (xmin, ymin, xmax, ymax) in chip_stream(
            it, aoi_item,
            size=args.size,
            stride=args.stride,
            scl_water_min_frac=args.scl_water_min_frac,
            debug=args.debug,
        ):
            fname = f"{it.id}_{wrote:04d}.jpg"
            iio.imwrite(tiles_dir / fname, chip, quality=90)
            all_rows.append({**meta, "tile": fname,
                             "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
            wrote += 1

        if wrote == 0 and args.save_previews:
            try:
                with rasterio.open(it.assets["visual"].href) as src:
                    H, W = src.height, src.width
                    r0 = max(0, H // 2 - args.size // 2)
                    c0 = max(0, W // 2 - args.size // 2)
                    arr = src.read(window=Window(c0, r0, args.size, args.size),
                                   out_shape=(3, args.size, args.size))
                    arr = np.transpose(arr, (1, 2, 0))
                    iio.imwrite(tiles_dir / f"{it.id}_preview.jpg", arr, quality=85)
                    print(f"{it.id}: wrote 0 chips → saved preview")
            except Exception:
                pass
        print(f"✅ {it.id}: wrote {wrote} tiles to {tiles_dir}")

    # write CSV
    with open(index_csv, "w", newline="") as f:
        fields = ["scene_id","datetime","collection","cloudcov","tile","xmin","ymin","xmax","ymax"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"\nDone. Total tiles: {len(all_rows)} → {index_csv}")

if __name__ == "__main__":
    main()
