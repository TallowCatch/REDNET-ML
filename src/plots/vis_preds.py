# src/plots/vis_preds.py
from __future__ import annotations
from src.torchvision_det.mps_patch import *  # sets MPS fallback + CPU NMS
import os, argparse, time, torch, torchvision
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Make missing MPS ops (nms/roi_align) fall back to CPU on Mac
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from src.torchvision_det.ds_torchvision import HABDetDataset, collate_fn, ResizeWithBoxes

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def build_model(arch: str, num_classes=2):
    arch = arch.lower()
    if arch in ("r50", "resnet50", "frcnn_resnet50"):
        m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    elif arch in ("mn", "mobilenet", "frcnn_mobilenet"):
        m = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    else:
        raise ValueError(f"Unknown arch '{arch}'. Use 'resnet50' or 'mobilenet'.")
    in_feats = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, num_classes)
    return m

def auto_arch_from_path(p: Path) -> str:
    s = str(p).lower()
    if "mobilenet" in s: return "mobilenet"
    if "resnet50" in s or "frcnn" in s: return "resnet50"
    return "resnet50"  # default

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True,
                    help="Path to best.pt (resnet or mobilenet)")
    ap.add_argument("--arch", type=str, default=None, choices=["resnet50","mobilenet"],
                    help="Override backbone arch (auto-detected from path if omitted)")
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--conf", type=float, default=0.35, help="score threshold for drawing")
    ap.add_argument("--k", type=int, default=20, help="max boxes to draw")
    args = ap.parse_args()

    wpath = Path(args.weights)
    arch = args.arch or auto_arch_from_path(wpath)

    dev = get_device()
    print(f"device: {dev} | arch: {arch} | weights: {wpath}")

    # dataset / loader
    tfm = ResizeWithBoxes((args.img_size, args.img_size))
    ds = HABDetDataset(args.split, transforms=tfm)

    # model
    m = build_model(arch, num_classes=2)
    state = torch.load(wpath, map_location="cpu")
    m.load_state_dict(state, strict=True)   # strict so we catch true mismatches
    m.to(dev).eval()

    out_dir = Path("runs/plots/vis"); out_dir.mkdir(parents=True, exist_ok=True)
    # visualize up to 8 images
    n_show = min(8, len(ds))
    for i in range(n_show):
        img, target = ds[i]
        img_np = (img.numpy().transpose(1,2,0) * 255).astype(np.uint8)
        gt = target["boxes"].numpy() if target["boxes"].numel() else np.zeros((0,4))

        pred = m([img.to(dev)])[0]
        boxes = pred["boxes"].detach().cpu().numpy()
        scores = pred["scores"].detach().cpu().numpy()

        # keep top-k above threshold
        keep = scores >= args.conf
        boxes, scores = boxes[keep], scores[keep]
        if len(boxes) > args.k:
            idx = np.argsort(-scores)[:args.k]
            boxes, scores = boxes[idx], scores[idx]

        # plot
        fig, ax = plt.subplots(1,2, figsize=(12,6))
        ax[0].imshow(img_np, cmap="gray"); ax[0].set_title("Ground truth (green)")
        for (x1,y1,x2,y2) in gt:
            ax[0].add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, ec="g", lw=2))

        ax[1].imshow(img_np, cmap="gray"); ax[1].set_title(f"Predictions (red)  k={len(boxes)}")
        for (x1,y1,x2,y2), s in zip(boxes, scores):
            ax[1].add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, fill=False, ec="r", lw=2))
            ax[1].text(x1, y1-2, f"{s:.2f}", color="r", fontsize=8)
        for a in ax: a.axis("off")

        out = out_dir / f"{arch}_{wpath.parent.name}_{i:02d}.png"
        fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)

    print(f"Saved {n_show} figures to {out_dir}")

if __name__ == "__main__":
    main()
