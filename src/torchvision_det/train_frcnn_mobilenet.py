from __future__ import annotations
from src.torchvision_det.mps_patch import *  # sets MPS fallback + CPU NMS
import os, time, torch, torchvision
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# MPS -> CPU fallback for missing ops (nms/roi_align)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from src.torchvision_det.ds_torchvision import (
    HABDetDataset, ResizeWithBoxes, collate_fn
)

OUT_DIR = Path("runs/detect/frcnn_mobilenet")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def get_model(num_classes=2):
    # lightweight FRCNN with MobileNetV3-Large FPN (320-px head)
    m = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
    in_feats = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    return m

def freeze_backbone(m, freeze=True):
    for p in m.backbone.parameters():
        p.requires_grad = not (freeze)

def loaders(img_size=512, bs=4, workers=0):
    tfm = ResizeWithBoxes((img_size, img_size))
    tr = HABDetDataset("train", transforms=tfm)
    tr.ids = [i for i in tr.ids if len(tr.coco.anns_by_img.get(i, [])) > 0]
    va = HABDetDataset("val", transforms=tfm)
    tr_dl = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=workers, collate_fn=collate_fn)
    va_dl = DataLoader(va, batch_size=bs, shuffle=False, num_workers=workers, collate_fn=collate_fn)
    return tr_dl, va_dl

@torch.no_grad()
def val_loss(m, dl, dev):
    m.train()
    tot = 0.0
    for imgs, tgts in dl:
        imgs = [x.to(dev) for x in imgs]
        tgts = [{k:v.to(dev) for k,v in t.items()} for t in tgts]
        tot += float(sum(m(imgs, tgts).values()))
    return tot / max(1, len(dl))

def phase(m, tr_dl, va_dl, dev, epochs, lr, tag):
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    best = 1e9
    log = (OUT_DIR / "log.csv").open("a", buffering=1)
    print("phase,epoch,val_loss", file=log)

    for ep in range(1, epochs+1):
        m.train()
        for imgs, tgts in tr_dl:
            imgs = [x.to(dev) for x in imgs]
            tgts = [{k:v.to(dev) for k,v in t.items()} for t in tgts]
            loss = sum(m(imgs, tgts).values())
            opt.zero_grad(); loss.backward(); opt.step()

        v = val_loss(m, va_dl, dev)
        print(f"[FRCNN-MN:{tag}] ep {ep:02d}  val_loss={v:.4f}")
        print(f"{tag},{ep},{v:.6f}", file=log)
        if v < best:
            best = v
            torch.save(m.state_dict(), OUT_DIR / "best.pt")

    log.close()

def main():
    dev = device(); print("device:", dev)
    tr_dl, va_dl = loaders(img_size=512, bs=4, workers=0)
    m = get_model(num_classes=2).to(dev)

    # Phase A: head only
    freeze_backbone(m, freeze=True)
    phase(m, tr_dl, va_dl, dev, epochs=8, lr=6e-4, tag="head")

    # Phase B: unfreeze & fine-tune
    freeze_backbone(m, freeze=False)
    phase(m, tr_dl, va_dl, dev, epochs=8, lr=1.5e-4, tag="rpn+head")

    # latency
    m.eval()
    x = [next(iter(va_dl))[0][0].to(dev)]
    for _ in range(5): m(x)
    t0 = time.time()
    for _ in range(50): m(x)
    print(f"Inference ~{50/(time.time()-t0):.1f} FPS on {dev}")

if __name__ == "__main__":
    main()
