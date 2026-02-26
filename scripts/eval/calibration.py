import numpy as np
import pandas as pd
import sys

def calibration_table(y_true, y_prob, bins=10):
    df = pd.DataFrame({
        "y": y_true,
        "p": y_prob
    })

    df["bin"] = pd.cut(df["p"], bins=bins)

    table = df.groupby("bin", observed=False).agg(
        mean_prob=("p", "mean"),
        true_rate=("y", "mean"),
        count=("y", "size")
    ).reset_index()

    return table


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calibration.py <csv_file>")
        sys.exit(1)

    path = sys.argv[1]
    df = pd.read_csv(path)

    y = df["hab_label"].values
    p = df["hab_prob"].values

    table = calibration_table(y, p, bins=10)
    print(table)
