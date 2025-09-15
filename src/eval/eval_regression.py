# src/eval/eval_regression.py
import argparse, os, math, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from src.train.train_regression import ChlTiles, build_model
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log_target", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # dataset without aug; in eval we still read the transformed target
    ds = ChlTiles(args.csv, "test", aug=False, log_target=bool(args.log_target))
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for x, y in dl:
            p = model(x.to(device)).cpu().numpy().ravel()
            y = y.numpy().ravel()
            preds.append(p); trues.append(y)
    preds = np.concatenate(preds); trues = np.concatenate(trues)

    # invert if trained on log-space
    if args.log_target:
        preds = np.expm1(preds)
        trues = np.expm1(trues)

    rmse = math.sqrt(np.mean((preds-trues)**2))
    mae  = np.mean(np.abs(preds-trues))
    r2   = 1 - np.sum((preds-trues)**2)/np.sum((trues-trues.mean())**2)
    print(f"Test RMSE={rmse:.3f}  MAE={mae:.3f}  R^2={r2:.3f}")

    # quick plots
    plt.figure()
    plt.scatter(trues, preds, s=8)
    plt.xlabel("True chlor_a (mg/m³)")
    plt.ylabel("Predicted")
    plt.title("Pred vs True (test)")
    plt.savefig(os.path.join(args.out, "scatter.png"), dpi=150); plt.close()

    plt.figure()
    plt.hist(preds-trues, bins=20)
    plt.xlabel("Error (mg/m³)"); plt.ylabel("Count")
    plt.title("Prediction error")
    plt.savefig(os.path.join(args.out, "error_hist.png"), dpi=150); plt.close()
