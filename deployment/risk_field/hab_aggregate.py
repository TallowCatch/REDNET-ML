#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math
from pathlib import Path

import numpy as np

# Shapely is the only geo dependency we assume.
# (If you don't have it: pip install shapely)
from shapely.geometry import shape, Point, Polygon, MultiPolygon, mapping
from shapely.ops import unary_union


def read_geojson_points(path: str):
    g = json.loads(Path(path).read_text())
    pts = []
    for f in g.get("features", []):
        geom = f.get("geometry")
        if not geom:
            continue
        if geom.get("type") != "Point":
            continue
        lon, lat = geom["coordinates"][:2]
        if not (np.isfinite(lon) and np.isfinite(lat)):
            continue
        pts.append((float(lon), float(lat)))
    if not pts:
        raise ValueError(f"No Point geometries found in {path}")
    return np.array(pts, dtype=float)


def km_to_deg_lat(km: float) -> float:
    return km / 111.0


def km_to_deg_lon(km: float, lat_deg: float) -> float:
    return km / (111.0 * max(1e-6, math.cos(math.radians(lat_deg))))


def write_geojson(path: str, features):
    Path(path).write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    print(f"✅ wrote {path} ({len(features)} features)")


def make_density_grid(points_lonlat: np.ndarray, grid_km: float, cap_points: int | None):
    # Optional cap for performance (random subsample)
    pts = points_lonlat
    if cap_points and len(pts) > cap_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), size=cap_points, replace=False)
        pts = pts[idx]
        print(f"🔻 Subsampled points: {len(points_lonlat)} → {len(pts)}")

    lons = pts[:, 0]
    lats = pts[:, 1]

    lat0 = float(np.median(lats))
    dlat = km_to_deg_lat(grid_km)
    dlon = km_to_deg_lon(grid_km, lat0)

    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())

    # Bin indices
    ix = np.floor((lons - lon_min) / dlon).astype(int)
    iy = np.floor((lats - lat_min) / dlat).astype(int)

    # Count per cell
    # key = (iy, ix)
    counts = {}
    for a, b in zip(iy, ix):
        counts[(int(a), int(b))] = counts.get((int(a), int(b)), 0) + 1

    total = float(len(pts))
    maxc = max(counts.values())

    features = []
    for (cy, cx), c in counts.items():
        # cell bounds
        x0 = lon_min + cx * dlon
        x1 = x0 + dlon
        y0 = lat_min + cy * dlat
        y1 = y0 + dlat

        poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])

        p = c / total                 # probability mass in this cell
        p_norm = c / float(maxc)      # normalized 0..1 for coloring

        features.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {
                "count": int(c),
                "p": float(p),
                "p_norm": float(p_norm),
                "grid_km": float(grid_km),
            }
        })

    return features


def largest_polygon(geom):
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(list(geom.geoms), key=lambda g: g.area, default=None)
    # fallback
    try:
        polys = [g for g in geom.geoms if isinstance(g, Polygon)]
        return max(polys, key=lambda g: g.area, default=None)
    except Exception:
        return None


def make_envelope(points_lonlat: np.ndarray, buffer_m: float, simplify_m: float):
    # "Nice" footprint without needing concave-hull libs:
    # 1) buffer points (meters→deg approx)
    # 2) union buffers → polygon blob
    # 3) simplify
    # 4) also output convex hull for reference

    lons = points_lonlat[:, 0]
    lats = points_lonlat[:, 1]
    lat0 = float(np.median(lats))

    # meters -> degrees (approx)
    buf_km = buffer_m / 1000.0
    simp_km = simplify_m / 1000.0
    dlat_buf = km_to_deg_lat(buf_km)
    dlon_buf = km_to_deg_lon(buf_km, lat0)
    # use average of lat/lon degree sizes to make an isotropic-ish buffer
    buf_deg = float((dlat_buf + dlon_buf) / 2.0)

    dlat_s = km_to_deg_lat(simp_km)
    dlon_s = km_to_deg_lon(simp_km, lat0)
    simp_deg = float((dlat_s + dlon_s) / 2.0)

    pts_geom = [Point(float(x), float(y)) for x, y in points_lonlat]
    hull = unary_union(pts_geom).convex_hull

    blob = unary_union([p.buffer(buf_deg) for p in pts_geom])
    blob = largest_polygon(blob)
    if blob is None:
        blob = hull

    blob_s = blob.simplify(simp_deg, preserve_topology=True)
    blob_s = largest_polygon(blob_s) or blob

    features = [
        {
            "type": "Feature",
            "geometry": mapping(hull),
            "properties": {"type": "convex_hull"}
        },
        {
            "type": "Feature",
            "geometry": mapping(blob_s),
            "properties": {
                "type": "buffer_union_envelope",
                "buffer_m": float(buffer_m),
                "simplify_m": float(simplify_m),
            }
        }
    ]
    return features


def main():
    ap = argparse.ArgumentParser("Aggregate HAB particle points into density + envelope")
    ap.add_argument("--in_points_geojson", required=True)
    ap.add_argument("--out_density_geojson", required=True)
    ap.add_argument("--out_envelope_geojson", required=True)

    ap.add_argument("--grid_km", type=float, default=2.0, help="Density grid cell size in km")
    ap.add_argument("--cap_points", type=int, default=200000, help="Max points used for density (subsample if bigger). 0=disable")
    ap.add_argument("--buffer_m", type=float, default=1500.0, help="Envelope buffer radius around points (meters)")
    ap.add_argument("--simplify_m", type=float, default=300.0, help="Envelope simplification tolerance (meters)")

    args = ap.parse_args()

    pts = read_geojson_points(args.in_points_geojson)
    print(f"📥 Loaded points: {len(pts):,}")

    cap = None if args.cap_points == 0 else int(args.cap_points)

    dens = make_density_grid(pts, grid_km=float(args.grid_km), cap_points=cap)
    write_geojson(args.out_density_geojson, dens)

    env = make_envelope(pts, buffer_m=float(args.buffer_m), simplify_m=float(args.simplify_m))
    write_geojson(args.out_envelope_geojson, env)

    print("✅ DONE (density + envelope)")


if __name__ == "__main__":
    main()
