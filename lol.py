import json
from collections import defaultdict

g = json.load(open("/tmp/final.geojson"))

by_pid = defaultdict(list)
for f in g["features"]:
    if f["properties"]["kind"] == "particle_point":
        by_pid[f["properties"]["pid"]].append(f)

pid0 = sorted(by_pid)[0]
coords = [(f["geometry"]["coordinates"], f["properties"]["time"])
          for f in by_pid[pid0]]

print(coords[:5])
print(coords[-5:])
