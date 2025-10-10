# src/torchvision_det/train_frcnn_mobilenet.py
from __future__ import annotations

# ---- env before torch/torchvision ----
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# monkey-patch MPS fallbacks (NMS / ROI Align)
from src.torchvision_det.mps_patch import *  # noqa: F401,F403

import time, math, argparse, csv
from pathlib import Path
import torch, torchvision
from torch.utils.data import DataLoader

from src.torchvision_det.ds_torchvision import HABDetDataset, collate_fn, ResizeWithBoxes

OUT_DIR = Path("runs/detect/frcnn_mobilenet")
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

def build_model(num_classes: int = 2, variant: str = "large_320_fpn"):
    """
    variant: 'large_320_fpn' (fast) or 'large_fpn' (bigger input head).
    """
    if variant == "large_fpn":
        m = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    else:
        m = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
    in_feats = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, num_classes)
    return m

def get_loaders(bs=8, img_size=320, workers=0, filter_empty_train=True):
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
            loss = sum(model(imgs, tgts).values())
        total += float(loss); n += 1
    return total / max(1, n)

def freeze_backbone(m):
    for p in m.backbone.parameters(): p.requires_grad = False

def unfreeze_rpn_and_head(m):
    for p in m.rpn.parameters(): p.requires_grad = True
    for p in m.roi_heads.parameters(): p.requires_grad = True

def train(
    epochs_head=10,
    epochs_rpn=40,
    bs=8,
    img_size=320,
    lr=7.5e-4,
    workers=0,
    amp=True,
    accum=1,
    eval_every=1,
    init_from: str = "",
    variant: str = "large_320_fpn",
):
    dev = get_device(); print("device:", dev)

    train_dl, val_dl = get_loaders(bs=bs, img_size=img_size, workers=workers, filter_empty_train=True)
    m = build_model(num_classes=2, variant=variant).to(dev)

    if init_from:
        sd = torch.load(init_from, map_location=dev)
        m.load_state_dict(sd, strict=True)
        print(f"Loaded init weights from {init_from}")

    device_type = "cuda" if dev.type == "cuda" else ("cpu" if dev.type == "cpu" else "mps")
    use_autocast = bool(amp and dev.type == "cuda")
    ac_dtype = torch.float16 if dev.type == "cuda" else torch.float32

    def do_epoch(opt):
        m.train(); step = 0
        opt.zero_grad(set_to_none=True)
        ctx = torch.autocast(device_type=device_type, dtype=ac_dtype, enabled=use_autocast)
        for imgs, tgts in train_dl:
            imgs = [i.to(dev) for i in imgs]
            tgts = [_coerce_target_dtypes(t, dev) for t in tgts]
            with ctx:
                losses = m(imgs, tgts)
                loss = sum(losses.values()) / accum
            loss.backward()
            step += 1
            if step % accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)

    _init_csv()
    best = float("inf")

    # ---- Phase A: ROI head only ----
    if epochs_head > 0:
        freeze_backbone(m); unfreeze_rpn_and_head(m)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=6, gamma=0.5)

        for ep in range(1, epochs_head+1):
            t0 = time.time()
            do_epoch(opt)
            v = eval_loss(m, val_dl, dev, amp=use_autocast) if (ep % eval_every == 0 or ep==epochs_head) else math.nan
            sched.step()
            secs = time.time()-t0
            print(f"[FRCNN-MOBILENET:head] ep {ep:02d}  val_loss={v:.4f}  {secs:.1f}s")
            if not math.isnan(v):
                _log_row("head", ep, v, secs)
                if v < best:
                    best = v; torch.save(m.state_dict(), OUT_DIR / "best_head_mobilenet.pt")

    # ---- Phase B: RPN + ROI head (backbone still frozen) ----
    unfreeze_rpn_and_head(m)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr*0.5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=6, gamma=0.5)

    for ep in range(1, epochs_rpn+1):
        t0 = time.time()
        do_epoch(opt)
        v = eval_loss(m, val_dl, dev, amp=use_autocast) if (ep % eval_every == 0 or ep==epochs_rpn) else math.nan
        sched.step()
        secs = time.time()-t0
        print(f"[FRCNN-MOBILENET:rpn+head] ep {ep:02d}  val_loss={v:.4f}  {secs:.1f}s")
        if not math.isnan(v):
            _log_row("rpn+head", ep, v, secs)
            if v < best:
                best = v; torch.save(m.state_dict(), OUT_DIR / "best_mobilenet.pt")

    # quick latency check
    m.eval()
    imgs, _ = next(iter(val_dl))
    x = [imgs[0].to(dev)]
    for _ in range(5): m(x)
    t0 = time.time()
    for _ in range(50): m(x)
    print(f"Inference ~{50/(time.time()-t0):.1f} FPS on {dev}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs_head", type=int, default=10)
    ap.add_argument("--epochs_rpn",  type=int, default=40)
    ap.add_argument("--bs",          type=int, default=8)
    ap.add_argument("--img_size",    type=int, default=320)
    ap.add_argument("--lr",          type=float, default=7.5e-4)
    ap.add_argument("--workers",     type=int, default=0)
    ap.add_argument("--amp",         type=int, default=1, help="autocast (CUDA only)")
    ap.add_argument("--accum",       type=int, default=1)
    ap.add_argument("--eval_every",  type=int, default=1)
    ap.add_argument("--init_from",   type=str,  default="", help="optional checkpoint to load")
    ap.add_argument("--variant",     type=str,  default="large_320_fpn", choices=["large_320_fpn","large_fpn"])
    args = ap.parse_args()
    train(**vars(args))
