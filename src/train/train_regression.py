# src/train/train_regression.py
import os, csv, json, math, time, argparse, random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Small_Weights


# -----------------------------
# Dataset
# -----------------------------
class ChlTiles(Dataset):
    """
    Reads PNG tiles and chl labels from data/labels/regression.csv
    CSV columns: filepath,split,chl
    """
    def __init__(self, csv_path: str, split: str, aug: bool = False,
                 log_target: bool = False, pretrained: bool = True):
        df = pd.read_csv(csv_path)
        self.df = df[df.split == split].reset_index(drop=True)
        self.aug = aug
        self.log_target = log_target
        self.pretrained = pretrained

        if pretrained:
            # ImageNet normalization + resize from the weights object
            weights = MobileNet_V3_Small_Weights.DEFAULT
            base_tx = weights.transforms()  # ToTensor + Normalize + resize/crop
            self.tx = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                base_tx
            ])
            aug_geo = transforms.RandomChoice([
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.RandomRotation(10, fill=0),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0),
            ])
            self.tx_aug = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                aug_geo,
                base_tx
            ])
        else:
            # Simple path (training from scratch)
            self.tx = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            self.tx_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(10, fill=0),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        img = Image.open(r.filepath)  # transforms handle conversion
        x = (self.tx_aug if self.aug else self.tx)(img)

        y = float(r.chl)
        if self.log_target:
            y = np.log1p(y)  # stable for small values
        y = torch.tensor([y], dtype=torch.float32)
        return x, y


# -----------------------------
# Model
# -----------------------------
def build_model(pretrained: bool = True) -> nn.Module:
    if pretrained:
        m = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    else:
        m = models.mobilenet_v3_small(weights=None)

    in_feat = m.classifier[0].in_features
    # Replace classifier head with a small MLP regressor
    m.classifier[-1] = nn.Identity()
    m.classifier = nn.Sequential(
        nn.Linear(in_feat, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(128, 1)
    )
    return m


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Training
# -----------------------------
def train(args):
    set_seed(42)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    os.makedirs(args.out, exist_ok=True)

    tr = ChlTiles(args.csv, "train", aug=True,
                  log_target=bool(args.log_target),
                  pretrained=bool(args.pretrained))
    va = ChlTiles(args.csv, "val", aug=False,
                  log_target=bool(args.log_target),
                  pretrained=bool(args.pretrained))

    tl = DataLoader(tr, batch_size=args.bs, shuffle=True,  num_workers=0, pin_memory=False)
    vl = DataLoader(va, batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=False)

    # ---- compute target stats (in log space if requested) from TRAIN split ----
    y_train = tr.df["chl"].astype(float).values
    if bool(args.log_target):
        y_train = np.log1p(y_train)
    mu, sigma = float(np.mean(y_train)), float(np.std(y_train) + 1e-8)

    # save stats for evaluation
    with open(os.path.join(args.out, "target_stats.json"), "w") as f:
        json.dump({"mu": mu, "sigma": sigma, "log_target": int(args.log_target)}, f)

    def _standardize(t: torch.Tensor) -> torch.Tensor:
        return (t - mu) / sigma

    def _destandardize(t: torch.Tensor) -> torch.Tensor:
        return t * sigma + mu

    model = build_model(pretrained=bool(args.pretrained)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.MSELoss()  # standardized target
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3
    )

    

    best_rmse = math.inf

    # ---- CSV logs ----
    tr_csv_path = os.path.join(args.out, "train_log.csv")
    va_csv_path = os.path.join(args.out, "val_log.csv")
    tr_csv = open(tr_csv_path, "w", newline=""); trw = csv.writer(tr_csv); trw.writerow(["epoch","train_loss"])
    va_csv = open(va_csv_path, "w", newline=""); vaw = csv.writer(va_csv); vaw.writerow(["epoch","val_rmse"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0

        for x, y in tl:
            x, y = x.to(device), y.to(device)
            y_std = _standardize(y)
            opt.zero_grad(set_to_none=True)
            pred_std = model(x)
            loss = crit(pred_std, y_std)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)

        tr_loss = loss_sum / len(tr)

        # ---- validation RMSE in original physical space (mg/m^3) ----
        model.eval()
        se, n = 0.0, 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                pred_std = model(x)
                pred_t   = _destandardize(pred_std)   # back to raw/log space
                if bool(args.log_target):
                    pred_phys = torch.expm1(pred_t)
                    y_phys    = torch.expm1(y)
                else:
                    pred_phys = pred_t
                    y_phys    = y
                se += ((pred_phys - y_phys) ** 2).sum().item()
                n  += y_phys.numel()

        rmse = math.sqrt(se / n)
        sched.step(rmse)

        old_lr = opt.param_groups[0]['lr']
        new_lr = opt.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"⚠️ LR reduced: {old_lr:.2e} → {new_lr:.2e}")

        print(f"epoch {epoch:03d} | train {tr_loss:.4f} | val RMSE {rmse:.3f} | "
              f"lr {opt.param_groups[0]['lr']:.2e} | {time.time()-t0:.1f}s")

        trw.writerow([epoch, tr_loss]); tr_csv.flush()
        vaw.writerow([epoch, rmse]);    va_csv.flush()

        torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))

    tr_csv.close(); va_csv.close()


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/labels/regression.csv")
    ap.add_argument("--out", default="runs/reg_log_mobilenet_imagenet")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--pretrained", type=int, default=1)   # 1=use ImageNet weights
    ap.add_argument("--log_target", type=int, default=1)   # 1=train on log1p(chl)
    args = ap.parse_args()
    train(args)
