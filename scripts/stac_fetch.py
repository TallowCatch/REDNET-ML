# scripts/stac_fetch.py
from __future__ import annotations
import argparse, csv, json, os, warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import xy
from shapely.geometry import shape, box, Polygon, mapping
from shapely.ops import unary_union, transform as shp_transform
from pystac_client import Client
import planetary_computer as pc
from pyproj import CRS, Transformer
import imageio

warnings.filterwarnings("ignore", category=UserWarning)
WGS84 = CRS.from_epsg(4326)

def load_aoi(path: str):
    gj = json.loads(Path(path).read_text())
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    return unary_union([shape(f["geometry"]) for f in feats])

def search_items(aoi_geom, start, end, max_cloud, collection="sentinel-2-l2a", limit=200):
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

def read_band(item, key):
    if key not in item.assets:
        return None, None, None
    with rasterio.open(item.assets[key].href) as r:
        arr = r.read(1).astype(np.float32)
        return arr, r.transform, r.crs

def read_visual_rgb(item):
    if "visual" in item.assets:
        with rasterio.open(item.assets["visual"].href) as src:
            img = src.read(out_dtype=np.uint8)
            return img, src.transform, src.crs
    need = ["B04", "B03", "B02"]
    stacks = []
    transform = crs = None
    for k in need:
        arr, transform, crs = read_band(item, k)
        if arr is None:
            raise RuntimeError("Missing RGB bands")
        stacks.append(arr)
    stack = np.stack(stacks, 0)  # R,G,B
    p2, p98 = np.percentile(stack, (2, 98))
    stack = np.clip((stack - p2) / (p98 - p2 + 1e-6), 0, 1) * 255
    return stack.astype(np.uint8), transform, crs

def window_poly(transform, win: Window) -> Polygon:
    left, top = win.col_off, win.row_off
    right, bottom = left + win.width, top + win.height
    xs = [left, right, right, left]; ys = [top, top, bottom, bottom]
    pts = [~transform * (x, y) for x, y in zip(xs, ys)]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return box(min(xs), min(ys), max(xs), max(ys))

def chip_and_save(
    img, transform, crs, aoi_clip, out_dir: Path, meta_row: dict,
    size=640, stride=320, water_only=False, water_min_frac=0.02,
    item=None, save_format="jpg", jpeg_quality=90,
    max_bytes: int | None = None, debug=False
):
    import imageio.v2 as iio
    from rasterio.transform import rowcol

    def over_quota():
        if not max_bytes: return False
        try:
            total = sum((p.stat().st_size for p in out_dir.glob("*") if p.is_file()), 0)
            return total >= max_bytes
        except Exception:
            return False

    out_dir.mkdir(parents=True, exist_ok=True)
    C, H, W = img.shape

    # map→pixel bounds of AOI∩image
    xmin, ymin, xmax, ymax = aoi_clip.bounds
    r0, c0 = rowcol(transform, xmin, ymax)  # UL
    r1, c1 = rowcol(transform, xmax, ymin)  # LR
    r0 = max(0, min(H-1, r0)); r1 = max(0, min(H-1, r1))
    c0 = max(0, min(W-1, c0)); c1 = max(0, min(W-1, c1))
    rmin, rmax = sorted([r0, r1]); cmin, cmax = sorted([c0, c1])

    # small pad
    pad = size
    rmin = max(0, rmin - pad); rmax = min(H, rmax + pad)
    cmin = max(0, cmin - pad); cmax = min(W, cmax + pad)

    # stride-aligned starts
    starts_r = list(range((rmin // stride) * stride, max(0, rmax - size + 1) + 1, stride))
    starts_c = list(range((cmin // stride) * stride, max(0, cmax - size + 1) + 1, stride))

    # if overlap thinner than size, we’ll force one later
    forced_center = None
    if not starts_r or not starts_c:
        rr = max(0, min(H - size, int((rmin + rmax - size) / 2)))
        cc = max(0, min(W - size, int((cmin + cmax - size) / 2)))
        forced_center = (rr, cc)

    # optional NDWI mask
    ndwi = None
    if water_only and item is not None:
        G, _, _   = read_band(item, "B03")
        NIR, _, _ = read_band(item, "B08")
        if G is not None and NIR is not None:
            ndwi = (G - NIR) / (G + NIR + 1e-6)

    def write_chip(r, c):
        from rasterio.windows import Window
        wpoly = window_poly(transform, Window(c, r, size, size))
        if not wpoly.intersects(aoi_clip): 
            return None
        chip = img[:, r:r+size, c:c+size]
        if chip.max() == 0:
            return None
        if ndwi is not None:
            nd = ndwi[r:r+size, c:c+size]
            if float((nd > 0.0).mean()) < water_min_frac:
                return None
        fname = f"{meta_row['scene_id']}_{r:05d}_{c:05d}.jpg"
        arr = np.transpose(chip, (1, 2, 0))
        imageio.v2.imwrite(out_dir / fname, arr, quality=jpeg_quality)
        xmin, ymin, xmax, ymax = wpoly.bounds
        return {**meta_row, "tile": fname, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}

    rows = []
    for r in starts_r or []:
        for c in starts_c or []:
            if r + size > H or c + size > W or over_quota():
                continue
            rec = write_chip(r, c)
            if rec: rows.append(rec)

    # Force one chip if sliding window produced none
    if not rows and forced_center and not over_quota():
        rr, cc = forced_center
        rec = write_chip(rr, cc)
        if rec: rows.append(rec)

    if debug:
        print(f"  debug: HxW={H}x{W}, clip_px=({rmin}:{rmax},{cmin}:{cmax}), wrote={len(rows)}")

    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", required=True, help="GeoJSON EPSG:4326")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end",   default="2023-12-31")
    ap.add_argument("--cloud", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out",   default="data/aerial")
    ap.add_argument("--size",  type=int, default=640)
    ap.add_argument("--stride",type=int, default=320)
    ap.add_argument("--water_only", action="store_true")
    ap.add_argument("--water_min_frac", type=float, default=0.02)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--save_previews", action="store_true",
                    help="write center preview if 0 chips")
    ap.add_argument("--max_bytes", type=int, default=0,
                    help="cap bytes written to tiles folder (0 = unlimited)")
    args = ap.parse_args()

    aoi_ll = load_aoi(args.aoi)
    out_root  = Path(args.out)
    tiles_dir = out_root / "tiles_png"
    index_csv = out_root / "index.csv"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    items = search_items(aoi_ll, args.start, args.end, args.cloud, "sentinel-2-l2a", args.limit)
    print(f"Found {len(items)} candidate items (capped by --limit).")

    all_rows = []
    for it in items:
        try:
            if not shape(it.geometry).intersects(aoi_ll):
                continue
        except Exception:
            pass

        try:
            img, transform, crs = read_visual_rgb(it)
        except Exception as e:
            print(f"skip {it.id}: {e}"); continue

        try:
            item_crs = CRS.from_user_input(crs)
            tfm = Transformer.from_crs(WGS84, item_crs, always_xy=True)
            aoi_item = shp_transform(lambda x, y: tfm.transform(x, y), aoi_ll)
        except Exception as e:
            print(f"skip {it.id}: reprojection error: {e}"); continue

        H, W = img.shape[1], img.shape[2]
        clip = aoi_item.intersection(raster_poly(transform, W, H))
        if clip.is_empty:
            continue

        meta = {
            "scene_id": it.id, "datetime": str(it.datetime),
            "collection": it.collection_id,
            "cloudcov": it.properties.get("eo:cloud_cover", it.properties.get("s2:cloud_cover", -1)),
        }

        rows = chip_and_save(
            img, transform, crs, clip, tiles_dir, meta,
            size=args.size, stride=args.stride,
            water_only=args.water_only, water_min_frac=args.water_min_frac,
            item=it, save_format="jpg", jpeg_quality=90,
            max_bytes=(args.max_bytes if args.max_bytes > 0 else None),
        )
        print(f"{it.id}: wrote {len(rows)} tiles")
        all_rows.extend(rows)

        if len(rows) == 0 and args.save_previews:
            import imageio.v2 as iio
            h0 = max(0, H // 2 - args.size // 2)
            w0 = max(0, W // 2 - args.size // 2)
            preview = np.transpose(img[:, h0:h0 + args.size, w0:w0 + args.size], (1, 2, 0))
            iio.imwrite(tiles_dir / f"{it.id}_preview.jpg", preview, quality=85)
            print(f"{it.id}: wrote 0 chips → saved preview for debugging.")

    with open(index_csv, "w", newline="") as f:
        fields = ["scene_id","datetime","collection","cloudcov","tile","xmin","ymin","xmax","ymax"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in all_rows: w.writerow(r)

    print("Done. Tiles:", len(all_rows), "| CSV:", index_csv)

if __name__ == "__main__":
    main()
