import pandas as pd
from sklearn.metrics import roc_auc_score

for model in ["r50", "mb", "ssd"]:
    path = f"runs/fusion/fused_sets/p_frcnn_with_HAB_label_{model}.csv"
    df = pd.read_csv(path).dropna(subset=["hab_label", f"p_frcnn_{model}"])
    y_true = df["hab_label"].astype(int)
    y_score = df[f"p_frcnn_{model}"].astype(float)
    auroc = roc_auc_score(y_true, y_score)
    print(f"{model.upper()} AUROC = {auroc:.3f}   (n={len(df)})")

