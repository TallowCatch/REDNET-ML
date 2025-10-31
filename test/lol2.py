import pandas as pd
for det in ["p_frcnn_with_HAB_label_mb.csv", "p_frcnn_with_HAB_label_r50.csv", "p_frcnn_with_HAB_label_ssd.csv"]:
    df = pd.read_csv(f"runs/fusion/fused_sets/{det}")
    print(f"\n{det}")
    print(df.describe())
