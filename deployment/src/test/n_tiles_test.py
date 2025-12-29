import pandas as pd

df = pd.read_csv("runs/fusion/fusion_training_table_clean.csv")

# Derive scene_root the same way you did in lol.py
df["scene_root"] = df["scene_id"].str.split("_20").str[0]

summary = (
    df.groupby("scene_root")
      .agg(
          n_tiles=("tile", "count"),
          n_acquisitions=("scene_id", "nunique"),
      )
      .sort_values("n_tiles", ascending=False)
)

print(summary.describe())
print(summary.head(10))
