# src/torchvision_det/train_ssd.py
import os, time, torch, torchvision
from pathlib import Path
from torch.utils.data import DataLoader
from .ds_torchvision import HABDetDataset, collate_fn

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

OUT_DIR = Path("runs/detect/ssd_mobilenet"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def get_model(num_classes=2):
    m = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights="DEFAULT")
    # swap head
    in_channels = m.head.classification_head.num_classes
    # torchvision helper:
    m = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights_backbone="DEFAULT")
    m.head.classification_head.num_classes = num_classes
    # torchscript workaround to reset head properly
    m.reset_parameters()
    return m

def loaders(bs=8, workers=0):
    tr = HABDetDataset("train")
    va = HABDetDataset("val")
    trdl = DataLoader(tr, batch_size=bs, shuffle=True,  num_workers=workers, collate_fn=collate_fn)
    vadl = DataLoader(va, batch_size=bs, shuffle=False, num_workers=workers, collate_fn=collate_fn)
    return trdl, vadl

@torch.no_grad()
def eval_loss(model, loader, dev):
    model.train()
    s, n = 0.0, 0
    for imgs, tgts in loader:
        imgs = [i.to(dev) for i in imgs]
        tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
        s += float(sum(model(imgs, tgts).values())); n += 1
    return s / max(1, n)

def train(epochs=30, bs=8, lr=1e-3):
    dev = device(); print("device:", dev)
    trdl, vadl = loaders(bs, workers=0)
    m = get_model().to(dev)

    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = 1e9
    for ep in range(1, epochs+1):
        m.train()
        for imgs, tgts in trdl:
            imgs = [i.to(dev) for i in imgs]
            tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
            loss = sum(m(imgs, tgts).values())
            opt.zero_grad(); loss.backward(); opt.step()
        v = eval_loss(m, vadl, dev); sch.step()
        print(f"[SSD] epoch {ep:03d}  val_loss={v:.4f}")
        if v < best: best = v; torch.save(m.state_dict(), OUT_DIR / "best.pt")

    m.eval(); imgs, _ = next(iter(vadl)); x = [imgs[0].to(dev)]
    for _ in range(5): m(x)
    t0 = time.time()
    for _ in range(100): m(x)
    print(f"Inference ~{100/(time.time()-t0):.1f} FPS on {dev}")

if __name__ == "__main__":
    train()
