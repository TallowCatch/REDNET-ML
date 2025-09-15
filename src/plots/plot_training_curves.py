import argparse, pandas as pd, matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="runs/reg_log_mobilenet_imagenet/train_log.csv")
ap.add_argument("--val",   default="runs/reg_log_mobilenet_imagenet/val_log.csv")
ap.add_argument("--out",   default="runs/reg_log_mobilenet_imagenet/eval/curves.png")
args = ap.parse_args()

tr = pd.read_csv(args.train)  # columns: epoch,train_loss
va = pd.read_csv(args.val)    # columns: epoch,val_rmse

fig, ax1 = plt.subplots(figsize=(7,4))
ax1.plot(tr["epoch"], tr["train_loss"], label="train loss")
ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss")

ax2 = ax1.twinx()
ax2.plot(va["epoch"], va["val_rmse"], color="tab:red", label="val RMSE")
ax2.set_ylabel("val RMSE (mg/m³)")

ax1.legend(loc="upper right"); ax2.legend(loc="upper left")
plt.title("Training curves")
plt.tight_layout()
plt.savefig(args.out, dpi=220)
print(f"Saved {args.out}")
