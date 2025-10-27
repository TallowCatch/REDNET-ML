#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch, torchvision
from torchvision.transforms import functional as TF

# ---- model loader (matches your training choices) ----
def build_model(kind: str, num_classes: int = 2):
    k = kind.lower()
    if k == "frcnn_resnet50":
        m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_feats = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, num_classes)
        return m
    if k == "frcnn_mobilenet":
        m = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
        in_feats = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, num_classes)
        return m
    if k == "ssd_mobilenet":
        # ssdlite320 with ImageNet backbone but no COCO head
        try:
            m = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
                weights=None, weights_backbone="DEFAULT", num_classes=num_classes
            )
        except TypeError:
            # older torchvision fallback
            m = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
                pretrained_backbone=True, num_classes=num_classes
            )
        return m
    raise SystemExit(f"Unknown kind: {kind}")

def load_image(path: Path, size: int):
    im = Image.open(path).convert("RGB")
    im = im.resize((size, size))
    t = TF.to_tensor(im)  # [0,1]
    return t

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", required=True, help="CSV with a column 'tile' listing filenames")
    ap.add_argument("--img_roots", nargs="+", required=True, help="Directories to search for tiles")
    ap.add_argument("--model_kind", required=True, choices=["frcnn_resnet50","frcnn_mobilenet","ssd_mobilenet"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--score_mode", default="max", choices=["max","mean_top3","sum_top3"])
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest_csv)
    if "tile" not in df.columns:
        raise SystemExit(f"{args.manifest_csv} must have a 'tile' column")

    # build search index of files
    
    # --- build a recursive index: filename -> absolute Path ---
    roots = [Path(r) for r in args.img_roots]

    index = {}
    exts = {".jpg", ".jpeg", ".png"}
    for root in roots:
        if not root.exists():
            print(f"[warn] img_root not found: {root}")
            continue
        for p in root.rglob("*"):
            if p.suffix.lower() in exts and p.is_file():
                index[p.name] = p.resolve()

    def find_path(name: str) -> Path | None:
        return index.get(Path(name).name)

    tiles = df["tile"].astype(str).apply(lambda s: Path(s).name).tolist()
    paths = [find_path(t) for t in tiles]
    miss = [t for t,p in zip(tiles, paths) if p is None]
    if miss:
        print(f"[warn] {len(miss)} tiles not found under img_roots; first few: {miss[:5]}")

    dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = build_model(args.model_kind).to(dev)
    sd = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(sd, strict=True)
    model.eval()

    rows = []
    for t, p in zip(tiles, paths):
        if p is None:
            rows.append({"tile": t, f"p_{args.model_kind}": np.nan})
            continue
        x = load_image(p, args.img_size).to(dev).unsqueeze(0)
        out = model(x)[0]
        scores = out.get("scores", torch.tensor([])).detach().cpu().numpy()
        if scores.size == 0:
            s = 0.0
        else:
            scores = np.sort(scores)[::-1]
            if args.score_mode == "max": s = float(scores[0])
            elif args.score_mode == "mean_top3": s = float(scores[:3].mean())
            else: s = float(scores[:3].sum())
        rows.append({"tile": t, f"p_{args.model_kind}": s})

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"✓ wrote {args.out_csv} (rows={len(rows)})")

if __name__ == "__main__":
    main()
