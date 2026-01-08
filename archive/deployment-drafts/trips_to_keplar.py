#!/usr/bin/env python3
import json
import csv
import argparse

def main():
    ap = argparse.ArgumentParser("Convert trips GeoJSON → Kepler Trip CSV")
    ap.add_argument("--in_geojson", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    with open(args.in_geojson) as f:
        g = json.load(f)

    rows = 0
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trajectory_id", "lon", "lat", "timestamp"])

        for tid, feat in enumerate(g["features"]):
            coords = feat["geometry"]["coordinates"]
            times = feat["properties"].get("timestamps")

            if not times or len(coords) != len(times):
                continue

            for (lon, lat), ts in zip(coords, times):
                writer.writerow([tid, lon, lat, int(ts)])
                rows += 1

    print(f"✅ Wrote {rows:,} rows to {args.out_csv}")

if __name__ == "__main__":
    main()
