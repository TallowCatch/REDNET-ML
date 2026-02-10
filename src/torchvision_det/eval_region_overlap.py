import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from pycocotools.coco import COCO
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
# helpers
# -----------------------------
def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * \
           (boxes[:, 3] - boxes[:, 1]).clamp(min=0)

def area_overlap(gt_boxes, pred_boxes):
    """
    Fraction of GT area covered by predicted boxes (union approx).
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0

    ious = box_iou(gt_boxes, pred_boxes)  # [G, P]
    inter = ious * box_area(gt_boxes).unsqueeze(1)
    covered = inter.max(dim=1).values     # best overlap per GT
    return float(covered.sum() / box_area(gt_boxes).sum())

# -----------------------------
# main eval
# -----------------------------
@torch.no_grad()
def evaluate(
    weights,
    arch,
    img_size,
    split="val",
    score_thr=0.3,
    iou_thr=0.1
):
    device = get_device()
    model = build_model_for_arch(arch)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()

    tfm = ResizeWithBoxes((img_size, img_size))
    ds = HABDetDataset(split, transforms=tfm, filter_empty=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    gt_regions = 0
    recalled_regions = 0
    overlap_scores = []

    for imgs, targets in dl:
        img = imgs[0].to(device)
        tgt = targets[0]
        gt_boxes = tgt["boxes"]

        if len(gt_boxes) == 0:
            continue

        pred = model([img])[0]
        boxes = pred.get("boxes", torch.empty(0, 4)).cpu()
        scores = pred.get("scores", torch.empty(0)).cpu()

        keep = scores >= score_thr
        boxes = boxes[keep]

        if len(boxes) == 0:
            overlap_scores.append(0.0)
            gt_regions += len(gt_boxes)
            continue

        # regional recall
        ious = box_iou(gt_boxes, boxes)
        recalled = (ious.max(dim=1).values >= iou_thr)

        gt_regions += len(gt_boxes)
        recalled_regions += recalled.sum().item()

        # area overlap
        overlap_scores.append(area_overlap(gt_boxes, boxes))

    return {
        "regional_recall": recalled_regions / max(1, gt_regions),
        "mean_area_overlap": float(np.mean(overlap_scores)),
        "num_gt_regions": gt_regions
    }

# -----------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--arch", required=True,
                    choices=["frcnn_resnet50","frcnn_mobilenet","ssd_mobilenet"])
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
