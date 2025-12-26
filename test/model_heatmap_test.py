import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("runs/fusion/fused_sets/fusion_enriched_norm_f1_v4/merged_features_debug.csv")

# keep only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# plot correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.show()
