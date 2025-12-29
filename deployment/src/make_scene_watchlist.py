import json
import pandas as pd
from pathlib import Path

SCENE_CSV = Path("deployment/outputs/scene_alerts.csv")
THR_JSON = Path("deployment/artifacts/thresholds.json")
OUT = Path("deployment/outputs/scene_alerts_watchlist.csv")

df = pd.read_csv(SCENE_CSV)
thr = json.loads(THR_JSON.read_text())

best_f1 = thr["operating_points"]["best_f1"]["threshold"]
active = thr["default_threshold"]

watch = df[
    (df["scene_prob"] >= best_f1) &
    (df["scene_prob"] < active)
].sort_values("scene_prob", ascending=False)

watch.to_csv(OUT, index=False)

print(f"watchlist scenes: {len(watch)}")
print(f"range: [{best_f1:.3f}, {active:.3f})")
print("top 5:")
print(watch.head(5)[["scene_id", "scene_prob", "n_tiles", "n_tiles_over_thr"]])
