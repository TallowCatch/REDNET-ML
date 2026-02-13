#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from src.torchvision_det.ds_torchvision import (
    HABDetDataset,
    collate_fn,
    ResizeWithBoxes
)
from src.torchvision_det.eval_ap import (
    get_device,
    build_model_for_arch
)

# -----------------------------
# helpers (unchanged)
# -----------------------------
def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)

def area_overlap(gt_boxes: torch.Tensor, pred_boxes: torch.Tensor) -> float:
    """
    Fraction of GT area covered by predicted boxes (union approx).
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0
    ious = box_iou(gt_boxes, pred_boxes)  # [G, P]
    inter = ious * box_area(gt_boxes).unsqueeze(1)
    covered = inter.max(dim=1).values     # best overlap per GT
    denom = box_area(gt_boxes).sum()
    return float(covered.sum() / denom) if float(denom) > 0 else 0.0


def _tensor_chw_to_uint8_hwc(img: torch.Tensor) -> np.ndarray:
    """
    Converts a CHW torch image to HWC uint8 numpy for Ultralytics.
    Assumes img is float in [0,1] OR [0,255].
    """
    x = img.detach().float().cpu()
    if x.max() <= 1.5:
        x = x * 255.0
    x = x.clamp(0, 255).byte()
    x = x.permute(1, 2, 0).numpy()  # HWC
    return x


@torch.no_grad()
def evaluate(
    weights: str,
    arch: str,
    img_size: int,
    split: str = "val",
    score_thr: float = 0.3,
    iou_thr: float = 0.1
) -> dict:
    device = get_device()

    tfm = ResizeWithBoxes((img_size, img_size))
    ds = HABDetDataset(split, transforms=tfm, filter_empty=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # -----------------------------
    # Load model depending on arch
    # -----------------------------
    yolo_model = None
    tv_model = None

    if arch.lower() in {"yolo", "yolov8", "yolov8n", "yolov8n3"}:
        # Ultralytics YOLOv8
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError(
                "Ultralytics not installed. Try: pip install ultralytics"
            ) from e

        yolo_model = YOLO(weights)
        # Ultralytics handles device via predict(..., device=)
        # but keeping a device string is helpful:
        device_str = "cuda" if (torch.cuda.is_available() and str(device).startswith("cuda")) else "cpu"
    else:
        # Torchvision detector (your existing path)
        tv_model = build_model_for_arch(arch)
        tv_model.load_state_dict(torch.load(weights, map_location=device))
        tv_model.to(device).eval()

    # -----------------------------
    # Evaluate
    # -----------------------------
    gt_regions = 0
    recalled_regions = 0
    overlap_scores = []

    for imgs, targets in dl:
        img = imgs[0].to(device)
        tgt = targets[0]
        gt_boxes = tgt["boxes"]  # already resized into img_size coords by ResizeWithBoxes

        if len(gt_boxes) == 0:
            continue

        if yolo_model is not None:
            # Convert resized tensor to uint8 HWC so YOLO predictions are in same coordinate frame
            img_np = _tensor_chw_to_uint8_hwc(img)

            # Predict on exactly this image size; since it's already img_size x img_size,
            # Ultralytics letterboxing won't change geometry.
            res = yolo_model.predict(
                source=img_np,
                imgsz=img_size,
                conf=score_thr,
                verbose=False,
                device=device_str
            )[0]

            if res.boxes is None or len(res.boxes) == 0:
                boxes = torch.empty((0, 4), dtype=torch.float32)
                scores = torch.empty((0,), dtype=torch.float32)
            else:
                boxes = res.boxes.xyxy.detach().cpu().float()  # [N,4]
                scores = res.boxes.conf.detach().cpu().float()  # [N]

        else:
            pred = tv_model([img])[0]
            boxes = pred.get("boxes", torch.empty(0, 4)).detach().cpu()
            scores = pred.get("scores", torch.empty(0)).detach().cpu()

            keep = scores >= score_thr
            boxes = boxes[keep]
            scores = scores[keep]

        if len(boxes) == 0:
            overlap_scores.append(0.0)
            gt_regions += len(gt_boxes)
            continue

        # regional recall
        ious = box_iou(gt_boxes.cpu(), boxes)
        recalled = (ious.max(dim=1).values >= iou_thr)

        gt_regions += len(gt_boxes)
        recalled_regions += int(recalled.sum().item())

        # area overlap
        overlap_scores.append(area_overlap(gt_boxes.cpu(), boxes))

    return {
        "regional_recall": recalled_regions / max(1, gt_regions),
        "mean_area_overlap": float(np.mean(overlap_scores)) if overlap_scores else 0.0,
        "num_gt_regions": int(gt_regions),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument(
        "--arch",
        required=True,
        choices=["frcnn_resnet50", "frcnn_mobilenet", "ssd_mobilenet", "yolo"],
        help="Use 'yolo' for Ultralytics YOLOv8 .pt weights"
    )
    ap.add_argument("--img_size", type=int, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--score_thr", type=float, default=0.3)
    ap.add_argument("--iou_thr", type=float, default=0.1)
    args = ap.parse_args()

    out = evaluate(
        args.weights,
        args.arch,
        args.img_size,
        args.split,
        args.score_thr,
        args.iou_thr
    )

    print("\n=== Regional Evaluation ===")
    for k, v in out.items():
        print(f"{k:20s}: {v}")


if __name__ == "__main__":
    main()
