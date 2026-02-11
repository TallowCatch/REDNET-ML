import pandas as pd
df = pd.read_csv("deployment/outputs/cv4_test_infer.csv")
df["scene_root"] = df["scene_id"].map(lambda s: "_".join(str(s).split("_")[:5]))
print("unique scene_root:", df["scene_root"].nunique())
print(df.groupby("scene_root").size().sort_values(ascending=False).head(20))
