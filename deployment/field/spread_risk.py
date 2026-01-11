# deployment/field/spread_risk.py

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

IN = "deployment/outputs/risk_grid_raw.csv"
OUT = "deployment/outputs/risk_grid_spread.csv"

SIGMA_KM = 15.0   # spread radius
EARTH_KM_PER_DEG = 111.0

df = pd.read_csv(IN, parse_dates=["time"])

out_rows = []

for (plant_id, time), g in df.groupby(["plant_id", "time"]):
    coords = g[["lon", "lat"]].values
    risks = g["risk"].values

    tree = cKDTree(coords)
    spread = np.zeros(len(g))

    for i, pt in enumerate(coords):
        dists, idx = tree.query(pt, k=len(coords))
        dists_km = dists * EARTH_KM_PER_DEG
        weights = np.exp(-(dists_km**2) / (2 * SIGMA_KM**2))
        spread[i] = np.sum(weights * risks)

    spread /= spread.max() if spread.max() > 0 else 1

    g2 = g.copy()
    g2["risk"] = spread
    out_rows.append(g2)

out = pd.concat(out_rows, ignore_index=True)
out.to_csv(OUT, index=False)
print(f"✅ wrote {OUT}")
