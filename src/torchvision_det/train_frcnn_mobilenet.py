# src/torchvision_det/train_frcnn_mobilenet.py
from __future__ import annotations
import os, time, torch, torchvision
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from pathlib import Path
from torch.utils.data import DataLoader
import torchvision.transforms as T

from src.torchvision_det.ds_torchvision import HABDetDataset, collate_fn, ResizeWithBoxes

OUT_DIR = Path("runs/detect/frcnn_mobilenet"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def build_model(num_classes=2):
    m = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
    in_feats = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, num_classes)
    return m

def get_loaders(bs=8, img_size=320, workers=0, filter_empty_train=True):
    tfm = ResizeWithBoxes((img_size, img_size))
    train_ds = HABDetDataset("train", transforms=tfm, filter_empty=filter_empty_train)
    val_ds   = HABDetDataset("val",   transforms=tfm, filter_empty=False)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=workers, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=workers, collate_fn=collate_fn)
    return train_dl, val_dl

@torch.no_grad()
def eval_loss(model, loader, dev):
    model.train()
    total, n = 0.0, 0
    for imgs, tgts in loader:
        imgs = [i.to(dev) for i in imgs]
        tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
        loss = sum(model(imgs, tgts).values())
        total += float(loss); n += 1
    return total / max(1, n)

def freeze_backbone(m):
    for p in m.backbone.parameters():
        p.requires_grad = False

def unfreeze_rpn_and_head(m):
    for p in m.rpn.parameters():
        p.requires_grad = True
    for p in m.roi_heads.parameters():
        p.requires_grad = True

def train(epochs_head=8, epochs_rpn=8, bs=8, img_size=320, lr=7.5e-4):
    dev = get_device(); print("device:", dev)
    train_dl, val_dl = get_loaders(bs=bs, img_size=img_size, workers=0, filter_empty_train=True)
    m = build_model().to(dev)

    # Phase A: ROI head only
    freeze_backbone(m); unfreeze_rpn_and_head(m)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=6, gamma=0.5)

    best = float("inf")
    for ep in range(1, epochs_head+1):
        m.train()
        for imgs, tgts in train_dl:
            imgs = [i.to(dev) for i in imgs]
            tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
            loss = sum(m(imgs, tgts).values())
            opt.zero_grad(); loss.backward(); opt.step()
        v = eval_loss(m, val_dl, dev); sched.step()
        print(f"[FRCNN-MN] ep {ep:02d}  val_loss={v:.4f}")
        if v < best:
            best = v
            torch.save(m.state_dict(), OUT_DIR / "best_head.pt")

    # Phase B: RPN + head (backbone still frozen)
    unfreeze_rpn_and_head(m)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr * 0.5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=6, gamma=0.5)

    for ep in range(1, epochs_rpn+1):
        m.train()
        for imgs, tgts in train_dl:
            imgs = [i.to(dev) for i in imgs]
            tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
            loss = sum(m(imgs, tgts).values())
            opt.zero_grad(); loss.backward(); opt.step()
        v = eval_loss(m, val_dl, dev); sched.step()
        print(f"[FRCNN-MN+] ep {ep:02d}  val_loss={v:.4f}")
        if v < best:
            best = v
            torch.save(m.state_dict(), OUT_DIR / "best.pt")

    # latency check
    m.eval()
    imgs, _ = next(iter(val_dl))
    x = [imgs[0].to(dev)]
    for _ in range(5): m(x)
    t0 = time.time()
    for _ in range(50): m(x)
    print(f"Inference ~{50/(time.time()-t0):.1f} FPS on {dev}")

if __name__ == "__main__":
    # Optional: export LIMIT_N=0 (or unset) for full set; small for smoke tests
    train()
