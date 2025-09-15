# plots a histogram of (pred - gt) errors
import argparse, pandas as pd, matplotlib.pyplot as plt, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--preds", default="runs/reg_log_mobilenet_imagenet/preds.csv")
ap.add_argument("--gt",    default="data/labels/regression.csv")
ap.add_argument("--out",   default="runs/reg_log_mobilenet_imagenet/eval/hist_error.png")
ap.add_argument("--bins",  type=int, default=30)
args = ap.parse_args()

gt   = pd.read_csv(args.gt)
pred = pd.read_csv(args.preds)
df   = gt.merge(pred, on="filepath", how="inner")

err = df["pred_chl"] - df["chl"]
plt.figure(figsize=(7,5))
plt.hist(err, bins=args.bins)
plt.title("Prediction error (pred − GT)")
plt.xlabel("Error (mg/m³)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(args.out, dpi=200)
print(f"Saved {args.out}  | n={len(df)}")
