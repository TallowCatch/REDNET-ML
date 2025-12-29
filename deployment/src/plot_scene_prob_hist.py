import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SCENE_CSV = Path("deployment/outputs/scene_alerts.csv")
THR_JSON = Path("deployment/artifacts/thresholds.json")

df = pd.read_csv(SCENE_CSV)
thr = json.loads(THR_JSON.read_text())

best_f1 = thr["operating_points"]["best_f1"]["threshold"]
active = thr["default_threshold"]

plt.figure(figsize=(9,5))
plt.hist(df["scene_prob"], bins=40, alpha=0.75)
plt.axvline(best_f1, color="orange", linestyle="--", label=f"best F1 ({best_f1:.2f})")
plt.axvline(active, color="red", linestyle="--", label=f"alert ({active:.2f})")

plt.xlabel("scene_prob")
plt.ylabel("count")
plt.title("Scene-level HAB probability distribution")
plt.legend()
plt.tight_layout()
plt.show()
