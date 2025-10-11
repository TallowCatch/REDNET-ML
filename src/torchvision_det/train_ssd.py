# src/torchvision_det/train_ssd.py
from __future__ import annotations

# ---- MPS / OpenMP hygiene BEFORE torch imports ----
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Ensure our fallback patches (NMS/ROI Align CPU) load early
from src.torchvision_det.mps_patch import *  # noqa: F401,F403

import time, math, argparse, csv
from pathlib import Path
import torch, torchvision
from torch.utils.data import DataLoader

from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from src.torchvision_det.ds_torchvision import HABDetDataset, collate_fn, ResizeWithBoxes

OUT_DIR = Path("runs/detect/ssd_mobilenet")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- CSV logging ----------
LOG_CSV = OUT_DIR / "val_log.csv"
def _init_csv():
    if not LOG_CSV.exists():
        with open(LOG_CSV, "w", newline="") as f:
            csv.writer(f).writerow(["phase", "epoch", "val_loss", "seconds", "ts"])
def _log_row(phase, epoch, val_loss, seconds):
    with open(LOG_CSV, "a", newline="") as f:
        csv.writer(f).writerow([phase, int(epoch), float(val_loss), float(seconds), int(time.time())])
# -------------------------------

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def build_ssd(num_classes: int = 2):
    """
    Version-agnostic SSD Lite with MobileNetV3 backbone.
    Fresh detection head via num_classes; ImageNet backbone weights.
    """
    model = ssdlite320_mobilenet_v3_large(
        weights=None,                 # no COCO head
        weights_backbone="DEFAULT",   # ImageNet backbone
        num_classes=num_classes       # {background, HAB}
    )
    return model

def make_loaders(bs=8, workers=0, img_size=320, filter_empty_train=True):
    tfm = ResizeWithBoxes((img_size, img_size))
    train_ds = HABDetDataset("train", transforms=tfm, filter_empty=filter_empty_train)
    val_ds   = HABDetDataset("val",   transforms=tfm, filter_empty=False)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=workers,
                          collate_fn=collate_fn, persistent_workers=workers>0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=workers,
                          collate_fn=collate_fn, persistent_workers=workers>0, pin_memory=False)
    return train_dl, val_dl

def _coerce_target_dtypes(t, dev):
    out = {}
    if "boxes" in t:    out["boxes"] = t["boxes"].to(dev, dtype=torch.float32)
    if "labels" in t:   out["labels"] = t["labels"].to(dev, dtype=torch.int64)
    if "image_id" in t: out["image_id"] = t["image_id"].to(dev, dtype=torch.int64)
    if "iscrowd" in t:  out["iscrowd"] = t["iscrowd"].to(dev, dtype=torch.int64)
    if "area" in t:     out["area"] = t["area"].to(dev, dtype=torch.float32)
    return out

@torch.no_grad()
def eval_loss(model, loader, dev, amp=False):
    # SSD returns losses only in train() mode
    model.train()
    total, n = 0.0, 0
    device_type = "cuda" if dev.type == "cuda" else ("cpu" if dev.type == "cpu" else "mps")
    ac_enabled = bool(amp and dev.type == "cuda")
    ac_dtype = torch.float16 if dev.type == "cuda" else torch.float32
    ctx = torch.autocast(device_type=device_type, dtype=ac_dtype, enabled=ac_enabled)
    for imgs, tgts in loader:
        imgs = [i.to(dev) for i in imgs]
        tgts = [_coerce_target_dtypes(t, dev) for t in tgts]
        with ctx:
            loss_dict = model(imgs, tgts)
            loss = sum(loss_dict.values())
        total += float(loss); n += 1
    return total / max(1, n)

def train(
    epochs=40,
    bs=12,
    lr=1e-3,
    img_size=320,
    workers=0,
    amp=True,
    accum=1,
    eval_every=1,
    init_from: str = ""
):
    dev = get_device(); print("device:", dev)
    train_dl, val_dl = make_loaders(bs=bs, workers=workers, img_size=img_size, filter_empty_train=True)
    model = build_ssd(num_classes=2).to(dev)

    if init_from:
        sd = torch.load(init_from, map_location=dev)
        model.load_state_dict(sd, strict=True)
        print(f"Loaded init weights from {init_from}")

    # Optim & sched (SSD likes SGD)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # AMP: CUDA only; keep fp32 on MPS/CPU to avoid dtype issues
    device_type = "cuda" if dev.type == "cuda" else ("cpu" if dev.type == "cpu" else "mps")
    use_autocast = bool(amp and dev.type == "cuda")
    ac_dtype = torch.float16 if dev.type == "cuda" else torch.float32

    def do_epoch():
        model.train()
        step = 0
        opt.zero_grad(set_to_none=True)
        ctx = torch.autocast(device_type=device_type, dtype=ac_dtype, enabled=use_autocast)
        for imgs, tgts in train_dl:
            imgs = [i.to(dev) for i in imgs]
            tgts = [_coerce_target_dtypes(t, dev) for t in tgts]
            with ctx:
                loss_dict = model(imgs, tgts)
                loss = sum(loss_dict.values()) / accum
            loss.backward()
            step += 1
            if step % accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)

    _init_csv()
    best = float("inf")
    for ep in range(1, epochs + 1):
        t0 = time.time()
        do_epoch()
        v = eval_loss(model, val_dl, dev, amp=use_autocast) if (ep % eval_every == 0 or ep == epochs) else math.nan
        sched.step()
        secs = time.time() - t0
        print(f"[SSD-MN] epoch {ep:03d}  val_loss={v:.4f}  {secs:.1f}s")
        if not math.isnan(v):
            _log_row("main", ep, v, secs)
            if v < best:
                best = v
                torch.save(model.state_dict(), OUT_DIR / "best_ssd.pt")

    # quick latency check
    model.eval()
    imgs, _ = next(iter(val_dl))
    x = [imgs[0].to(dev)]
    for _ in range(5): model(x)
    t0 = time.time()
    for _ in range(100): model(x)
    print(f"Inference ~{100/(time.time()-t0):.1f} FPS on {dev}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",      type=int,   default=40)
    ap.add_argument("--bs",          type=int,   default=12)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--img_size",    type=int,   default=320)
    ap.add_argument("--workers",     type=int,   default=0)
    ap.add_argument("--amp",         type=int,   default=1, help="autocast (CUDA only)")
    ap.add_argument("--accum",       type=int,   default=1, help="gradient accumulation steps")
    ap.add_argument("--eval_every",  type=int,   default=1)
    ap.add_argument("--init_from",   type=str,   default="", help="optional checkpoint to load")
    args = ap.parse_args()
    train(**vars(args))
