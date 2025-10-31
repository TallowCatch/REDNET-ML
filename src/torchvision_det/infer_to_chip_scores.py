#!/usr/bin/env python3
from __future__ import annotations
import os, sys, argparse, math
# --- set env BEFORE importing torch/torchvision ---
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
import torch, torchvision
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights

# --- make repo importable when run as a script ---
_THIS = Path(__file__).resolve()
_SRC  = _THIS.parents[1]            # .../src
_ROOT = _SRC.parent                 # project root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.torchvision_det.ds_torchvision import HABDetDataset, ResizeWithBoxes, collate_fn
from src.torchvision_det.mps_patch import *  # noqa: F401,F403

# ================================================================
#                        MODEL BUILDERS
# ================================================================
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
        try:
            m = ssdlite320_mobilenet_v3_large(
                weights=None,
                weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1,
                num_classes=num_classes
            )
        except TypeError:
            # for older torchvision versions
            m = ssdlite320_mobilenet_v3_large(weights=None, num_classes=num_classes)
        return m
    raise ValueError(f"Unknown model_kind: {kind}")

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ================================================================
#                          INFERENCE
# ================================================================
@torch.no_grad()
def infer_to_chip_scores(model, loader, device, iou_thresh=0.5, score_key="scores"):
    """
    Returns dict[chip_id] -> (max_prob, num_dets)
    """
    model.eval().to(device)
    out = {}
    nms = torchvision.ops.nms

    for imgs, tgts in loader:
        imgs_dev = [im.to(device) for im in imgs]
        preds = model(imgs_dev)

        for pred, tgt in zip(preds, tgts):
            # identify chip id (robust to dataset implementation)
            chip_id = None
            if "image_id_str" in tgt:
                chip_id = tgt["image_id_str"]
                if isinstance(chip_id, torch.Tensor):
                    chip_id = chip_id.item() if chip_id.numel() == 1 else str(chip_id)
            if chip_id is None and "image_id" in tgt:
                val = tgt["image_id"]
                chip_id = int(val.item()) if isinstance(val, torch.Tensor) else int(val)
            if chip_id is None and "file_name" in tgt:
                chip_id = Path(str(tgt["file_name"])).stem

            # 🔧 NEW fallback: try dataset lookup if target didn't include filename
            if chip_id is None and hasattr(loader.dataset, "files"):
                idx = int(tgt.get("image_id", -1))
                if 0 <= idx < len(loader.dataset.files):
                    chip_id = Path(loader.dataset.files[idx]).stem

            # final fallback (just string-cast)
            if chip_id is None:
                chip_id = str(tgt.get("image_id", "unknown"))

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

# ================================================================
#                            MAIN
# ================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_kind", required=True, choices=["frcnn_resnet50","frcnn_mobilenet","ssd_mobilenet"])
    ap.add_argument("--ckpt", required=True, help="path to .pt checkpoint")
    ap.add_argument("--split", default="val", help="comma-separated splits, e.g. train,val")
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--score_col", default=None, help="column name to write (default auto from model_kind)")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    args = ap.parse_args()

    # ------------------ build + load model ------------------
    dev = get_device()
    model = build_model(args.model_kind, num_classes=2)
    sd = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(sd, strict=True)
    print(f"✓ Loaded {args.model_kind} weights from {args.ckpt}")

    # ------------------ handle multiple splits ------------------
    splits = [s.strip() for s in args.split.split(",")]
    all_rows = []

    for split in splits:
        print(f"→ Running inference on split: {split}")
        tfm = ResizeWithBoxes((args.img_size, args.img_size))
        ds  = HABDetDataset(split, transforms=tfm, filter_empty=False)
        dl  = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate_fn)

        scores = infer_to_chip_scores(model, dl, dev, iou_thresh=args.iou)
        for k, v in scores.items():
            all_rows.append((k, v[0], v[1]))

    # ------------------ write output CSV ------------------
    col = args.score_col or {
        "frcnn_resnet50": "p_frcnn_r50",
        "frcnn_mobilenet": "p_frcnn_mb",
        "ssd_mobilenet": "p_ssd_mb",
    }[args.model_kind]

    rows = []
    for chip_id, pmax, ndet in all_rows:
        rows.append({
            "chip_id": chip_id,
            col: pmax,
            f"{col}_count": ndet
        })

    out_df = pd.DataFrame(rows).sort_values("chip_id")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"✓ wrote {args.out_csv}  (rows={len(out_df)})")

# ================================================================
if __name__ == "__main__":
    main()
