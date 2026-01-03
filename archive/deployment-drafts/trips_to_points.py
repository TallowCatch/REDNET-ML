import json, csv

INP = "sur_trips.geojson"   # or sur_trips.geojson
OUT = "sur_points_kepler.csv"

g = json.load(open(INP))

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pid","lon","lat","time"])
    for pid, feat in enumerate(g["features"]):
        coords = feat["geometry"]["coordinates"]
        ts = feat["properties"]["timestamps"]
        for (lon, lat), t in zip(coords, ts):
            w.writerow([pid, lon, lat, t])

print("wrote", OUT)
