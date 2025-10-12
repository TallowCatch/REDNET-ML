from __future__ import annotations
import os, time, math, argparse
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from src.torchvision_det.mps_patch import *  # noqa

import torch, torchvision
from pathlib import Path
from torch.utils.data import DataLoader

from src.torchvision_det.ds_aerial import AerialDetDataset, collate_fn, ResizeWithBoxes

OUT_DIR = Path("runs/detect/frcnn_resnet50_aerial"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def build_model(num_classes=2):
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_feats = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, num_classes)
    return m

def get_loaders(bs=8, img_size=640, workers=0, filter_empty_train=True):
    tfm = ResizeWithBoxes((img_size, img_size))
    train_ds = AerialDetDataset("train", transforms=tfm, filter_empty=filter_empty_train)
    val_ds   = AerialDetDataset("val",   transforms=tfm, filter_empty=False)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=workers, collate_fn=collate_fn, persistent_workers=workers>0)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=workers, collate_fn=collate_fn, persistent_workers=workers>0)
    return train_dl, val_dl

def _coerce(t, dev):
    out={}
    if "boxes" in t: out["boxes"]=t["boxes"].to(dev, dtype=torch.float32)
    if "labels" in t: out["labels"]=t["labels"].to(dev, dtype=torch.int64)
    if "image_id" in t: out["image_id"]=t["image_id"].to(dev, dtype=torch.int64)
    if "iscrowd" in t: out["iscrowd"]=t["iscrowd"].to(dev, dtype=torch.int64)
    if "area" in t: out["area"]=t["area"].to(dev, dtype=torch.float32)
    return out

@torch.no_grad()
def eval_loss(m, dl, dev):
    m.train()
    tot,n=0.0,0
    for imgs,tgts in dl:
        imgs=[i.to(dev) for i in imgs]; tgts=[_coerce(t,dev) for t in tgts]
        loss=sum(m(imgs,tgts).values()); tot+=float(loss); n+=1
    return tot/max(1,n)

def train(epochs_head=10, epochs_rpn=40, bs=8, img_size=640, lr=5e-4, workers=0, accum=1, init_from=""):
    dev=get_device(); print("device:",dev)
    train_dl,val_dl=get_loaders(bs,img_size,workers,True)
    m=build_model().to(dev)
    if init_from:
        m.load_state_dict(torch.load(init_from, map_location=dev), strict=True)

    def do_epoch(opt):
        m.train(); step=0; opt.zero_grad(set_to_none=True)
        for imgs,tgts in train_dl:
            imgs=[i.to(dev) for i in imgs]; tgts=[_coerce(t,dev) for t in tgts]
            loss=sum(m(imgs,tgts).values())/accum
            loss.backward(); step+=1
            if step%accum==0: opt.step(); opt.zero_grad(set_to_none=True)

    best=float("inf")
    # Phase A: head only
    for p in m.backbone.parameters(): p.requires_grad=False
    for p in m.rpn.parameters(): p.requires_grad=True
    for p in m.roi_heads.parameters(): p.requires_grad=True
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    sched=torch.optim.lr_scheduler.StepLR(opt, step_size=6, gamma=0.5)
    for ep in range(1,epochs_head+1):
        t0=time.time(); do_epoch(opt); v=eval_loss(m,val_dl,dev); sched.step()
        print(f"[AERIAL-R50:head] ep {ep:02d}  val_loss={v:.4f}  {time.time()-t0:.1f}s")
        if v<best: best=v; torch.save(m.state_dict(), OUT_DIR/"best_head.pt")

    # Phase B: rpn+head (backbone still frozen)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr*0.5)
    sched=torch.optim.lr_scheduler.StepLR(opt, step_size=6, gamma=0.5)
    for ep in range(1,epochs_rpn+1):
        t0=time.time(); do_epoch(opt); v=eval_loss(m,val_dl,dev); sched.step()
        print(f"[AERIAL-R50:rpn+head] ep {ep:02d}  val_loss={v:.4f}  {time.time()-t0:.1f}s")
        if v<best: best=v; torch.save(m.state_dict(), OUT_DIR/"best.pt")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs_head",type=int,default=10)
    ap.add_argument("--epochs_rpn",type=int,default=40)
    ap.add_argument("--bs",type=int,default=8)
    ap.add_argument("--img_size",type=int,default=640)
    ap.add_argument("--lr",type=float,default=5e-4)
    ap.add_argument("--workers",type=int,default=0)
    ap.add_argument("--accum",type=int,default=1)
    ap.add_argument("--init_from",type=str,default="")
    args=ap.parse_args()
    train(**vars(args))
