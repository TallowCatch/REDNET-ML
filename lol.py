import pandas as pd

paths = [
    "runs/datasets/hab_candidates_review.csv",
    "runs/datasets/hab_train_mined.csv",
    "runs/datasets/hab_train_mined_aslabel.csv",
    "runs/datasets/hab_train_nonleaky.csv"
]
for p in paths:
    df = pd.read_csv(p)
    print(f"\n{p}: {len(df)} rows")
    print(df['hab_label'].value_counts(dropna=False))
