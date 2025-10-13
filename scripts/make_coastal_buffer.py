#!/usr/bin/env python3
from __future__ import annotations
import argparse, io, json, zipfile
from pathlib import Path

import geopandas as gpd
from shapely.ops import unary_union
from shapely.validation import make_valid as _make_valid
from shapely import set_precision
from shapely.geometry import shape, mapping

NE_COAST_URL = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_coastline.zip"

def make_valid(geom):
    # Try make_valid, then buffer(0) fallback, then precision snap
    try:
        g = _make_valid(geom)
    except Exception:
        g = geom
    if (g is None) or g.is_empty:
        g = geom.buffer(0)
    try:
        g = set_precision(g, 0.1)  # ~10 cm snap in projected coords
    except Exception:
        pass
    return g

def ensure_coast_shp() -> Path:
    out_dir = Path("data/aoi"); out_dir.mkdir(parents=True, exist_ok=True)
    shp_dir = out_dir / "ne_10m_coastline"
    shp_path = shp_dir / "ne_10m_coastline.shp"
    if not shp_path.exists():
        print("Downloading Natural Earth coastline...")
        import urllib.request
        data = urllib.request.urlopen(NE_COAST_URL).read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(shp_dir)
        print(f"✅ Coastline ready at {shp_path}")
    return shp_path

def load_aoi_wgs84(path: str) -> gpd.GeoSeries:
    gj = json.loads(Path(path).read_text())
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    geoms = [shape(f["geometry"]) for f in feats]
    gs = gpd.GeoSeries([unary_union(geoms)], crs="EPSG:4326")
    gs.iloc[0] = make_valid(gs.iloc[0])
    return gs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", required=True, help="AOI GeoJSON (EPSG:4326)")
    ap.add_argument("--out", required=True, help="Output coastal band GeoJSON")
    ap.add_argument("--km", type=float, default=10.0, help="Buffer distance (km) around coastline")
    args = ap.parse_args()

    # 1) AOI in WGS84 → 3857 (meters)
    aoi_ll = load_aoi_wgs84(args.aoi)
    aoi_m  = aoi_ll.to_crs(3857)  # EPSG:3857
    if aoi_m.iloc[0].is_empty:
        raise SystemExit("AOI is empty after validation.")

    # 2) Coastline (clip to AOI bbox) → 3857
    coast_shp = ensure_coast_shp()
    minx, miny, maxx, maxy = aoi_ll.total_bounds
    coast_ll = gpd.read_file(coast_shp, bbox=(minx, miny, maxx, maxy))
    if coast_ll.empty:
        raise SystemExit("No coastline within AOI bbox. Check AOI.")
    coast_m = coast_ll.to_crs(3857)

    # 3) Buffer coastline by K km (both sides) and dissolve → a band
    buf_m = float(args.km) * 1000.0
    # Clean linework first
    coast_m["geometry"] = coast_m.geometry.apply(make_valid)
    coast_m = coast_m[~coast_m.geometry.is_empty]

    try:
        buff = coast_m.buffer(buf_m)
    except Exception:
        # Robust fallback if GEOS complains
        buff = coast_m.buffer(buf_m, cap_style=1, join_style=1)

    band = gpd.GeoSeries([unary_union(buff)], crs=3857)
    band.iloc[0] = make_valid(band.iloc[0])
    if band.iloc[0].is_empty:
        raise SystemExit("Buffered band is empty; increase --km or expand AOI.")

    # 4) Intersect with AOI to keep just your area
    out_gdf = gpd.overlay(
        gpd.GeoDataFrame(geometry=band, crs=3857),
        gpd.GeoDataFrame(geometry=aoi_m, crs=3857),
        how="intersection",
    )
    out_gdf = out_gdf[~out_gdf.geometry.is_empty]
    if out_gdf.empty:
        raise SystemExit("Band ∩ AOI is empty; increase --km or adjust AOI.")

    # 5) Back to WGS84 and write GeoJSON
    out_ll = out_gdf.to_crs(4326).dissolve().explode(index_parts=False).reset_index(drop=True)
    geom = make_valid(out_ll.geometry.iloc[0])
    featcol = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"km": args.km}, "geometry": mapping(geom)}],
    }
    Path(args.out).write_text(json.dumps(featcol))
    print(f"✅ Wrote coastal band → {args.out}")

if __name__ == "__main__":
    main()
