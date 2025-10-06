# src/eval/eval_regression.py
import argparse, os, math, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.train.train_regression import ChlTiles, build_model


def load_state_dict_strict(model, ckpt_path, device):
    # Robust loader for PyTorch 2.6+
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and any(k in ckpt for k in ("model", "state_dict")):
        sd = ckpt.get("model", ckpt.get("state_dict"))
    else:
        sd = ckpt
    if any(k.startswith("module.") for k in sd.keys()):
        from collections import OrderedDict
        sd = OrderedDict((k.replace("module.", "", 1), v) for k, v in sd.items())
    model.load_state_dict(sd, strict=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True,
                    help="Directory to write plots; also used to find target_stats.json")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # dataset (test split)
    ds = ChlTiles(args.csv, "test", aug=False, log_target=False, pretrained=True)  # log flag handled via stats
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(pretrained=True).to(device).eval()
    load_state_dict_strict(model, args.ckpt, device)

    # load target stats (prefer alongside ckpt; else use args.out)
    stats_path = os.path.join(os.path.dirname(args.ckpt), "target_stats.json")
    if not os.path.isfile(stats_path):
        stats_path = os.path.join(args.out, "target_stats.json")
    with open(stats_path, "r") as f:
        st = json.load(f)
    mu, sigma = float(st["mu"]), float(st["sigma"])
    log_target_flag = bool(st.get("log_target", 1))

    preds_std, trues = [], []
    with torch.no_grad():
        for x, y in dl:
            p_std = model(x.to(device)).cpu().numpy().ravel()  # standardized output
            preds_std.append(p_std)
            trues.append(y.numpy().ravel())                    # raw/log per training flag (raw here)
    preds_std = np.concatenate(preds_std)
    trues = np.concatenate(trues)

    # de-standardize -> (raw/log) space
    preds = preds_std * sigma + mu

    # invert log if trained in log space to PHYSICAL mg/m^3
    if log_target_flag:
        preds = np.expm1(preds)

    # Metrics
    rmse = math.sqrt(np.mean((preds - trues) ** 2))
    mae  = float(np.mean(np.abs(preds - trues)))
    r2   = 1.0 - float(np.sum((preds - trues) ** 2) / np.sum((trues - trues.mean()) ** 2 + 1e-12))
    print(f"Test RMSE={rmse:.3f}  MAE={mae:.3f}  R^2={r2:.3f}")

    # Save predictions CSV next to plots
    out_csv = os.path.join(args.out, "preds.csv")
    pd.DataFrame({"filepath": ds.df["filepath"], "chl": trues, "pred_chl": preds}).to_csv(out_csv, index=False)

    # Plots
    import matplotlib
    matplotlib.use("Agg")

    plt.figure()
    plt.scatter(trues, preds, s=8)
    plt.xlabel("True chlor_a (mg/m³)")
    plt.ylabel("Predicted")
    plt.title("Pred vs True (test)")
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(args.out, "scatter.png"), dpi=150); plt.close()

    plt.figure()
    plt.hist(preds - trues, bins=20)
    plt.xlabel("Error (mg/m³)")
    plt.ylabel("Count")
    plt.title("Prediction error")
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(args.out, "error_hist.png"), dpi=150); plt.close()
