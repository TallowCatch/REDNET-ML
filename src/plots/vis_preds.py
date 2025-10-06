# src/plots/vis_preds.py
from __future__ import annotations
import os, argparse, numpy as np, torch, torchvision
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms

# Mac MPS safety (lets unsupported ops fall back to CPU)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from src.torchvision_det.ds_torchvision import HABDetDataset, ResizeWithBoxes, collate_fn

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def build_model(arch: str, num_classes=2):
    if arch == "resnet50":
        m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_feats = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, num_classes)
        return m
    if arch == "mobilenet":
        m = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
        in_feats = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, num_classes)
        return m
    if arch == "ssd":
        return torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights=None, weights_backbone="DEFAULT", num_classes=num_classes
        )
    raise ValueError(f"Unknown arch {arch}")

def infer_arch_from_path(p: Path) -> str:
    s = str(p).lower()
    if "ssd" in s: return "ssd"
    if "mobilenet" in s: return "mobilenet"
    return "resnet50"

def to_pil(img_t: torch.Tensor) -> Image.Image:
    # img_t: CxHxW in [0,1]
    arr = (img_t.detach().cpu().numpy().transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return Image.fromarray(arr)

def draw_boxes(draw: ImageDraw.ImageDraw, boxes, color, width=2, labels=None, font=None):
    for i, (x1,y1,x2,y2) in enumerate(boxes):
        draw.rectangle([x1,y1,x2,y2], outline=color, width=width)
        if labels is not None:
            txt = labels[i]
            if isinstance(txt, float): txt = f"{txt:.2f}"
            draw.text((x1, max(0, y1-12)), str(txt), fill=color, font=font)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=str)
    ap.add_argument("--arch", default="", choices=["", "resnet50", "mobilenet", "ssd"])
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--img_size", type=int, default=640)  # use 320 for SSD, 640 for FRCNN-R50
    ap.add_argument("--score_thr", type=float, default=0.3)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--outdir", type=str, default="runs/plots/vis")
    args = ap.parse_args()

    wpath = Path(args.weights)
    arch = args.arch or infer_arch_from_path(wpath)

    dev = get_device()
    print(f"device: {dev} | arch: {arch} | weights: {wpath}")

    # model + weights
    model = build_model(arch, num_classes=2).to(dev).eval()
    state = torch.load(wpath, map_location=dev)
    model.load_state_dict(state, strict=True)

    # dataset (resize + keep boxes consistent)
    tfm = ResizeWithBoxes((args.img_size, args.img_size))
    ds = HABDetDataset(args.split, transforms=tfm, filter_empty=False)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except:
        font = None

    # make side-by-side canvases
    for i in range(min(12, len(ds))):
        img_t, tgt = ds[i]
        pred = model([img_t.to(dev)])[0]

        # Get boxes/scores and apply score-threshold + NMS + topK on CPU (safe on MPS)
        boxes  = pred["boxes"].detach().cpu()
        scores = pred["scores"].detach().cpu()
        keep = scores >= args.score_thr
        boxes, scores = boxes[keep], scores[keep]
        if len(boxes) > 0:
            keep_nms = nms(boxes, scores, args.nms_iou)
            boxes, scores = boxes[keep_nms], scores[keep_nms]
        if len(boxes) > args.topk:
            topk_idx = torch.topk(scores, args.topk).indices
            boxes, scores = boxes[topk_idx], scores[topk_idx]

        # Compose side-by-side image
        left  = to_pil(img_t)        # GT
        right = left.copy()          # Pred
        draw_gt   = ImageDraw.Draw(left)
        draw_pred = ImageDraw.Draw(right)

        # draw ground truth in green
        if len(tgt["boxes"]) > 0:
            draw_boxes(draw_gt, tgt["boxes"].numpy(), color=(0,255,0), width=2, labels=None, font=font)

        # draw predictions in red with scores
        if len(boxes) > 0:
            draw_boxes(draw_pred, boxes.numpy(), color=(255,0,0), width=2, labels=scores.tolist(), font=font)

        # assemble side-by-side
        W = left.width; H = left.height
        canvas = Image.new("RGB", (2*W + 40, H + 60), (12,38,41))
        canvas.paste(left,  (20, 40))
        canvas.paste(right, (W+40, 40))
        # headings
        d = ImageDraw.Draw(canvas)
        d.text((W//2 - 80, 10), "Ground truth (green)", fill=(220,220,220), font=font)
        d.text((W + W//2 - 80, 10), f"Predictions (red)  k≤{args.topk}", fill=(220,220,220), font=font)

        canvas.save(outdir / f"{arch}_{args.split}_{i:02d}.png")

    print(f"Saved figures to: {outdir}")

if __name__ == "__main__":
    main()
