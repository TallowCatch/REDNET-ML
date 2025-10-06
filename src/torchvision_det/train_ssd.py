# src/torchvision_det/train_ssd.py
from __future__ import annotations

# --- MPS + OpenMP hygiene (Mac) ---
import os
from src.torchvision_det.mps_patch import *  # sets MPS fallback + CPU NMS
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time, torch, torchvision
from pathlib import Path
from torch.utils.data import DataLoader

from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from src.torchvision_det.ds_torchvision import HABDetDataset, collate_fn, ResizeWithBoxes
from src.torchvision_det.mps_patch import *  # keeps NMS/ROI ops safe on MPS by CPU fallback

OUT_DIR = Path("runs/detect/ssd_mobilenet")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def build_ssd(num_classes: int = 2):
    """
    Proper, version-agnostic way: let torchvision build SSDLite with a fresh head
    by passing num_classes at construction, while using ImageNet backbone weights.
    """
    model = ssdlite320_mobilenet_v3_large(
        weights=None,                      # don't load COCO head
        weights_backbone="DEFAULT",        # MobileNetV3 backbone weights
        num_classes=num_classes            # 2 -> {background, HAB}
    )
    return model

def make_loaders(bs=8, workers=0, img_size=320):
    tfm = ResizeWithBoxes((img_size, img_size))  # keeps boxes consistent with resize
    train_ds = HABDetDataset("train", transforms=tfm, filter_empty=True)
    val_ds   = HABDetDataset("val",   transforms=tfm, filter_empty=False)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                          num_workers=workers, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds, batch_size=bs, shuffle=False,
                          num_workers=workers, collate_fn=collate_fn)
    return train_dl, val_dl

@torch.no_grad()
def eval_loss(model, loader, dev):
    # SSD returns losses only in train() mode
    model.train()
    total, n = 0.0, 0
    for imgs, tgts in loader:
        imgs = [i.to(dev) for i in imgs]
        tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
        loss_dict = model(imgs, tgts)
        total += float(sum(loss_dict.values()))
        n += 1
    return total / max(1, n)

def train(epochs=30, bs=8, lr=1e-3, img_size=320):
    dev = get_device()
    print("device:", dev)
    train_dl, val_dl = make_loaders(bs=bs, workers=0, img_size=img_size)
    model = build_ssd(num_classes=2).to(dev)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        for imgs, tgts in train_dl:
            imgs = [i.to(dev) for i in imgs]
            tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
            loss = sum(model(imgs, tgts).values())
            opt.zero_grad()
            loss.backward()
            opt.step()

        v = eval_loss(model, val_dl, dev)
        sched.step()
        print(f"[SSD] epoch {ep:03d}  val_loss={v:.4f}")
        if v < best:
            best = v
            torch.save(model.state_dict(), OUT_DIR / "best.pt")

    # quick latency check
    model.eval()
    imgs, _ = next(iter(val_dl))
    x = [imgs[0].to(dev)]
    for _ in range(5): model(x)
    t0 = time.time()
    for _ in range(100): model(x)
    print(f"Inference ~{100/(time.time()-t0):.1f} FPS on {dev}")

if __name__ == "__main__":
    train()
