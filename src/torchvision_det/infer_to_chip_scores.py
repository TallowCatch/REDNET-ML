#!/usr/bin/env python3
from __future__ import annotations
import os, sys, argparse, math
# --- set env BEFORE importing torch/torchvision ---
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# --- import our patch BEFORE torchvision so the monkey-patch is in place ---
from pathlib import Path
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights
import torchvision


# --- make repo importable when run as a script ---
_THIS = Path(__file__).resolve()
_SRC  = _THIS.parents[1]            # .../src
_ROOT = _SRC.parent                 # project root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch, torchvision
import pandas as pd
from torch.utils.data import DataLoader

from src.torchvision_det.ds_torchvision import HABDetDataset, ResizeWithBoxes, collate_fn
from src.torchvision_det.mps_patch import *  # noqa: F401,F403


# ------------------------- model builders -------------------------
def build_model(kind: str, num_classes: int = 2):
    k = kind.lower()
    if k == "frcnn_resnet50":
        m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_feats = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, num_classes)
        return m
    if k == "frcnn_mobilenet":
        m = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
        in_feats = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, num_classes)
        return m
    if k == "ssd_mobilenet":
        # Version-robust construction: load ImageNet backbone only, fresh detection head with your num_classes
        try:
            m = ssdlite320_mobilenet_v3_large(
                weights=None,
                weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,  # or .DEFAULT
                num_classes=num_classes
            )
        except TypeError:
            # older torchvision doesn't support weights_backbone
            m = ssdlite320_mobilenet_v3_large(weights=None, num_classes=num_classes)
        return m

    raise ValueError(f"Unknown model_kind: {kind}")

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

# ----------------------------- infer ------------------------------
@torch.no_grad()
def infer_to_chip_scores(model, loader, device, iou_thresh=0.5, score_key="scores"):
    """
    Returns dict chip_id -> (max_prob, num_dets)
    """
    model.eval().to(device)
    out = {}
    nms = torchvision.ops.nms

    for imgs, tgts in loader:
        # imgs: list[T]; tgts: list[dict]
        imgs_dev = [im.to(device) for im in imgs]
        preds = model(imgs_dev)

        for pred, tgt in zip(preds, tgts):
            # identify chip id (robust to dataset implementation)
            chip_id = None
            if "image_id_str" in tgt:
                chip_id = tgt["image_id_str"]
                if isinstance(chip_id, torch.Tensor):
                    chip_id = chip_id.item() if chip_id.numel()==1 else str(chip_id)
            if chip_id is None and "image_id" in tgt:
                # map numeric id to dataset string if present
                val = tgt["image_id"]
                chip_id = int(val.item()) if isinstance(val, torch.Tensor) else int(val)
            # Fallback: try file path if dataset exposes it
            if chip_id is None and "file_name" in tgt:
                chip_id = Path(str(tgt["file_name"])).stem

            # preds: boxes, scores, labels
            boxes  = pred.get("boxes", torch.empty((0,4), device=device))
            scores = pred.get(score_key, torch.empty((0,), device=device))
            if boxes.numel() > 0 and scores.numel() > 0:
                keep = nms(boxes, scores, iou_thresh)
                kept_scores = scores[keep]
                pmax = float(kept_scores.max().item()) if kept_scores.numel() else 0.0
                ndet = int(kept_scores.numel())
            else:
                pmax, ndet = 0.0, 0

            out[str(chip_id)] = (pmax, ndet)
    return out

# ----------------------------- main -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_kind", required=True, choices=["frcnn_resnet50","frcnn_mobilenet","ssd_mobilenet"])
    ap.add_argument("--ckpt", required=True, help="path to .pt checkpoint")
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--score_col", default=None, help="column name to write, default auto from model_kind")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU")
    args = ap.parse_args()

    # dataset/loader (no filtering on val/test; keep empties)
    tfm = ResizeWithBoxes((args.img_size, args.img_size))
    ds  = HABDetDataset(args.split, transforms=tfm, filter_empty=False)
    dl  = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # build + load
    dev = get_device()
    model = build_model(args.model_kind, num_classes=2)
    sd = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(sd, strict=True)
    print(f"Loaded {args.model_kind} weights from {args.ckpt}")

    # run
    scores = infer_to_chip_scores(model, dl, dev, iou_thresh=args.iou)
    # choose column name
    col = args.score_col or {
        "frcnn_resnet50": "p_frcnn_r50",
        "frcnn_mobilenet": "p_frcnn_mb",
        "ssd_mobilenet": "p_ssd_mb",
    }[args.model_kind]

    # write CSV
    rows = [{"chip_id": k, col: v[0], f"{col}_count": v[1]} for k, v in scores.items()]
    out_df = pd.DataFrame(rows).sort_values("chip_id")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"✓ wrote {args.out_csv}  (rows={len(out_df)})")

if __name__ == "__main__":
    main()
