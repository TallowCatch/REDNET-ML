#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import torch
import pandas as pd
from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image


# -------------------------------------------------
# utils
# -------------------------------------------------
def load_image(p: Path, img_size: int):
    img = Image.open(p).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return tfm(img)

def chip_id_from_path(p: Path) -> str:
    return p.stem

def max_score_from_outputs(outputs, score_thresh=0.05):
    if not outputs:
        return 0.0
    scores = outputs[0].get("scores")
    if scores is None or len(scores) == 0:
        return 0.0
    scores = scores.detach().cpu()
    scores = scores[scores >= score_thresh]
    return float(scores.max()) if len(scores) else 0.0


# -------------------------------------------------
# model builders (MATCH TRAINING EXACTLY)
# -------------------------------------------------
def build_frcnn_resnet50(num_classes: int, ckpt: Path, device: str):
    m = fasterrcnn_resnet50_fpn(weights=None)
    in_feats = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.to(device).eval()

def build_frcnn_mobilenet(num_classes: int, ckpt: Path, device: str):
    m = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_feats = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.to(device).eval()

def build_ssd(num_classes: int, ckpt: Path, device: str):
    m = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone="DEFAULT",
        num_classes=num_classes,
    )
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.to(device).eval()


# -------------------------------------------------
# detector runner (UNCHANGED)
# -------------------------------------------------
def run_detector(model, chips_dir: Path, img_size: int, colname: str, device: str):
    rows = []

    img_paths = sorted(
        list(chips_dir.glob("*.jpg")) +
        list(chips_dir.glob("*.jpeg")) +
        list(chips_dir.glob("*.png")) +
        list(chips_dir.glob("*.JPG"))
    )

    if not img_paths:
        return pd.DataFrame(columns=["chip_id", colname])

    for img_path in img_paths:
        x = load_image(img_path, img_size).to(device)
        with torch.no_grad():
            out = model([x])
        prob = max_score_from_outputs(out)
        rows.append({
            "chip_id": chip_id_from_path(img_path),
            colname: prob,
        })

    return pd.DataFrame(rows)


# -------------------------------------------------
# main
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True,
                    help="Folder containing YYYY-MM subfolders")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--frcnn_r50", required=True)
    ap.add_argument("--frcnn_mb", required=True)
    ap.add_argument("--ssd_mb", required=True)

    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models ONCE
    frcnn_r50 = build_frcnn_resnet50(args.num_classes, Path(args.frcnn_r50), args.device)
    frcnn_mb  = build_frcnn_mobilenet(args.num_classes, Path(args.frcnn_mb), args.device)
    ssd_mb    = build_ssd(args.num_classes, Path(args.ssd_mb), args.device)

    # Traverse months
    for month_dir in sorted(root_dir.iterdir()):
        if not month_dir.is_dir():
            continue

        chips_dir = month_dir / "chips" / "tiles_png"
        if not chips_dir.exists():
            print(f"⏭️  Skipping {month_dir.name} (no chips/tiles_png)")
            continue

        print(f"\n📆 Processing {month_dir.name}")

        dfs = [
            run_detector(frcnn_r50, chips_dir, 640, "p_frcnn_r50_med", args.device),
            run_detector(frcnn_mb,  chips_dir, 320, "p_frcnn_mb_med",  args.device),
            run_detector(ssd_mb,    chips_dir, 320, "p_ssd_mb_med",    args.device),
        ]

        det = dfs[0]
        for d in dfs[1:]:
            det = det.merge(d, on="chip_id", how="outer")

        det.fillna(0.0, inplace=True)

        out_csv = out_dir / f"{month_dir.name}_detector_scores.csv"
        det.to_csv(out_csv, index=False)
        print(f"✅ wrote {out_csv}")

if __name__ == "__main__":
    main()
