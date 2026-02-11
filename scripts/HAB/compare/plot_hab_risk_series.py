import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("deployment/outputs/inference_with_probs_attempt.csv")

# group by scene (or date)
scene_ts = (
    df.groupby(["scene_id", "datetime"])
      .agg(
          hab_prob_mean=("hab_prob", "mean"),
          hab_prob_max=("hab_prob", "max"),
          hab_prob_p90=("hab_prob", lambda x: x.quantile(0.9))
      )
      .reset_index()
)

scene_ts.to_csv("deployment/outputs/scene_time_series.csv", index=False)
print(scene_ts.head())

scene_ts["datetime"] = pd.to_datetime(scene_ts["datetime"])

plt.figure(figsize=(12,4))
plt.plot(scene_ts["datetime"], scene_ts["hab_prob_max"], marker="o")
plt.axhline(0.3, linestyle="--", alpha=0.4)
plt.axhline(0.6, linestyle="--", alpha=0.4)
plt.ylabel("HAB probability")
plt.title("HAB risk time series (fixed Oman AOI)")
plt.show()
