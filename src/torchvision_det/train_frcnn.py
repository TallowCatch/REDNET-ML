import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import time, torch, torchvision
from pathlib import Path
import torchvision.transforms as T
from src.torchvision_det.ds_torchvision import HABDetDataset, collate_fn
from torch.utils.data import DataLoader

OUT_DIR = Path("runs/detect/frcnn"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def get_model(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model

def get_loaders(bs=4, img_size=640, workers=0):
    tfm = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    train_ds = HABDetDataset("train", transforms=tfm)
    val_ds   = HABDetDataset("val",   transforms=tfm)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                          num_workers=workers, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                          num_workers=workers, collate_fn=collate_fn)
    return train_dl, val_dl

@torch.no_grad()
def evaluate_loss(model, loader, device):
    # detection losses only produced in TRAIN mode
    model.train()
    total, n = 0.0, 0
    for imgs, targets in loader:
        imgs = [i.to(device) for i in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        total += float(sum(loss_dict.values()))
        n += 1
    return total / max(1, n)

def train(epochs=20, bs=4, img_size=640, lr=5e-4):
    device = get_device()
    print(f"device: {device}")
    train_dl, val_dl = get_loaders(bs, img_size, workers=0)  # macOS -> workers=0
    model = get_model(num_classes=2).to(device)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

    best = 1e9
    for ep in range(1, epochs+1):
        model.train()
        for imgs, targets in train_dl:
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            opt.zero_grad(); loss.backward(); opt.step()

        val_loss = evaluate_loss(model, val_dl, device)
        sched.step()
        with open(OUT_DIR/'log.csv', 'a') as f:
            if ep == 1: f.write('epoch,val_loss\n')
            f.write(f'{ep},{val_loss:.6f}\n')
        print(f"[FRCNN] epoch {ep:03d}  val_loss={val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), OUT_DIR / "best.pt")

    # latency check
    model.eval()
    imgs, _ = next(iter(val_dl))
    x = [imgs[0].to(device)]
    for _ in range(5): model(x)
    t0 = time.time()
    for _ in range(50): model(x)
    print(f"Inference ~{50/(time.time()-t0):.1f} FPS on {device}")

if __name__ == "__main__":
    train()
