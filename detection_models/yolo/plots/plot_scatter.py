# src/plots/plot_scatter.py
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_metrics(y, yh):
    mse  = np.mean((y - yh) ** 2)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(y - yh))
    return rmse, mae

def linear_scatter(y, yh, out, rmse, mae):
    plt.figure(figsize=(6, 6))
    plt.scatter(y, yh, alpha=0.6, s=15)
    lims = [min(y.min(), yh.min()), max(y.max(), yh.max())]
    plt.plot(lims, lims, "k-", lw=2)
    plt.xlabel("GT chlorophyll-a (mg/m³)")
    plt.ylabel("Predicted chlorophyll-a (mg/m³)")
    plt.title(f"Pred vs GT | RMSE={rmse:.3f}, MAE={mae:.3f}")
    plt.tight_layout()
    plt.savefig(f"{out}/scatter_linear.png", dpi=200)
    plt.close()

def loglog_scatter(y, yh, out, rmse, mae):
    plt.figure(figsize=(6, 6))
    plt.scatter(y, yh, alpha=0.6, s=15)
    lo = max(1e-3, min(y.min(), yh.min()))
    hi = max(y.max(), yh.max())
    plt.plot([lo, hi], [lo, hi], "k-", lw=2)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("GT chlorophyll-a (mg/m³) [log]")
    plt.ylabel("Predicted chlorophyll-a (mg/m³) [log]")
    plt.title(f"Pred vs GT (log-log) | RMSE={rmse:.3f}, MAE={mae:.3f}")
    plt.tight_layout()
    plt.savefig(f"{out}/scatter_loglog.png", dpi=200)
    plt.close()

def linear_hexbin(y, yh, out, gridsize=40, mincnt=1, cmap="viridis", rmse=None, mae=None):
    plt.figure(figsize=(6, 6))
    hb = plt.hexbin(y, yh, gridsize=gridsize, mincnt=mincnt, cmap=cmap, edgecolors='none')
    # removed unity line to avoid visual clutter
    cbar = plt.colorbar(hb); cbar.set_label("count")
    plt.xlabel("GT chlorophyll-a (mg/m³)")
    plt.ylabel("Predicted chlorophyll-a (mg/m³)")
    if rmse is not None:
        plt.title(f"Pred vs GT (hexbin) | RMSE={rmse:.3f}, MAE={mae:.3f}")
    plt.tight_layout()
    plt.savefig(f"{out}/hexbin_linear.png", dpi=200)
    plt.close()

def loglog_hexbin(y, yh, out, gridsize=40, mincnt=1, cmap="viridis", rmse=None, mae=None):
    # filter nonpositive before log scale
    m = (y > 0) & (yh > 0)
    y2, yh2 = y[m], yh[m]
    plt.figure(figsize=(6, 6))
    hb = plt.hexbin(
        y2, yh2,
        gridsize=gridsize, mincnt=mincnt, cmap=cmap,
        xscale="log", yscale="log", edgecolors='none'
    )
    # removed unity line that was faintly visible and distracting
    cbar = plt.colorbar(hb); cbar.set_label("count")
    plt.xlabel("GT chlorophyll-a (mg/m³) [log]")
    plt.ylabel("Predicted chlorophyll-a (mg/m³) [log]")
    if rmse is not None:
        plt.title(f"Pred vs GT (hexbin log-log) | RMSE={rmse:.3f}, MAE={mae:.3f}")
    plt.tight_layout()
    plt.savefig(f"{out}/hexbin_loglog.png", dpi=200)
    plt.close()

def main(args):
    gt   = pd.read_csv(args.csv)
    pred = pd.read_csv(args.preds)
    df = gt.merge(pred, on="filepath", how="inner")
    y  = df.chl.values.astype(float)
    yh = df.pred_chl.values.astype(float)

    rmse, mae = compute_metrics(y, yh)

    # always generate the two scatters
    linear_scatter(y, yh, args.out, rmse, mae)
    loglog_scatter(y, yh, args.out, rmse, mae)

    # optionally also generate hexbin density plots
    if args.hexbin:
        linear_hexbin(y, yh, args.out, gridsize=args.gridsize, mincnt=args.mincnt, cmap=args.cmap, rmse=rmse, mae=mae)
        loglog_hexbin(y, yh, args.out, gridsize=args.gridsize, mincnt=args.mincnt, cmap=args.cmap, rmse=rmse, mae=mae)

    print(
        f"Saved:\n"
        f"  {args.out}/scatter_linear.png\n"
        f"  {args.out}/scatter_loglog.png\n"
        + (f"  {args.out}/hexbin_linear.png\n  {args.out}/hexbin_loglog.png\n" if args.hexbin else "")
        + f"Metrics  RMSE={rmse:.3f}  MAE={mae:.3f}"
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",   default="data/labels/regression.csv")
    ap.add_argument("--preds", default="runs/reg_log_mobilenet_imagenet/preds.csv")
    ap.add_argument("--out",   default="runs/reg_log_mobilenet_imagenet/eval")
    ap.add_argument("--hexbin", action="store_true", help="also save hexbin density plots")
    ap.add_argument("--gridsize", type=int, default=40, help="hexbin grid size")
    ap.add_argument("--mincnt", type=int, default=1, help="minimum count to color a hex")
    ap.add_argument("--cmap", default="viridis", help="matplotlib colormap for hexbin")
    args = ap.parse_args()
    main(args)
