# grid of example tiles: best, worst, and random
import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt
from PIL import Image, ImageOps
from math import ceil

ap = argparse.ArgumentParser()
ap.add_argument("--preds", default="runs/reg_log_mobilenet_imagenet/preds.csv")
ap.add_argument("--gt",    default="data/labels/regression.csv")
ap.add_argument("--k",     type=int, default=12)  # total samples
ap.add_argument("--out",   default="runs/reg_log_mobilenet_imagenet/eval/samples.png")
args = ap.parse_args()

gt   = pd.read_csv(args.gt)
pred = pd.read_csv(args.preds)
df   = gt.merge(pred, on="filepath", how="inner")

df["abs_err"] = (df["pred_chl"] - df["chl"]).abs()

# pick 1/3 lowest error, 1/3 highest error, 1/3 random
k = args.k
k3 = max(1, k // 3)
low  = df.nsmallest(k3, "abs_err")
high = df.nlargest(k3, "abs_err")
rest = df.drop(pd.concat([low, high]).index)
rnd  = rest.sample(min(k - len(low) - len(high), len(rest)), random_state=42)

sel = pd.concat([low, high, rnd]).reset_index(drop=True)

cols = 4
rows = ceil(len(sel)/cols)
plt.figure(figsize=(cols*3.1, rows*3.1))
for i, r in sel.iterrows():
    img = Image.open(r.filepath).convert("L")
    img = ImageOps.equalize(img)  # mild contrast normalization for display
    ax = plt.subplot(rows, cols, i+1)
    ax.imshow(img, cmap="gray"); ax.axis("off")
    ax.set_title(f"GT {r.chl:.2f} | Pred {r.pred_chl:.2f}")
plt.tight_layout()
plt.savefig(args.out, dpi=220)
print(f"Saved {args.out}  | n={len(sel)}")
