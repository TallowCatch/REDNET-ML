# src/torchvision_det/rebuild_coco_from_split.py
# Rebuild COCO JSONs from existing split folders, INCLUDING images with 0 annotations.
# Layout expected (what your training code already uses):
#   data/labels/detection/
#     images/{train,val,test}/*.png|jpg
#     labels/{train,val,test}/*.txt            # YOLO format; may be missing or empty
#
# Usage (from repo root):
#   python -m src.torchvision_det.rebuild_coco_from_split \
#     --img-root data/labels/detection/images \
#     --lbl-root data/labels/detection/labels \
#     --out      data/labels/coco

from __future__ import annotations
from pathlib import Path
from PIL import Image
import json, argparse

CATEGORIES = [{"id": 1, "name": "HAB"}]   # single class

EXTS = {".png", ".jpg", ".jpeg"}

def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in EXTS])

def yolo_line_to_coco(line: str, w: int, h: int):
    """
    YOLO: cls cx cy bw bh   (all normalized 0..1)
    COCO: [x, y, width, height] in pixels, category_id >= 1
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls, cx, cy, bw, bh = map(float, parts[:5])
    except ValueError:
        return None

    # map YOLO class 0 -> COCO 1 (HAB). If you have multiple classes, extend here.
    cat_id = int(cls) + 1

    x = (cx - bw / 2.0) * w
    y = (cy - bh / 2.0) * h
    ww = bw * w
    hh = bh * h

    # clamp minimal sizes/coords
    x = max(0.0, x)
    y = max(0.0, y)
    ww = max(1.0, ww)
    hh = max(1.0, hh)

    return {
        "category_id": cat_id,
        "bbox": [float(x), float(y), float(ww), float(hh)],
        "area": float(ww * hh),
        "iscrowd": 0,
        "segmentation": [],
    }

def build_split(split: str, img_root: Path, lbl_root: Path, out_dir: Path):
    img_dir = img_root / split
    lbl_dir = lbl_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    ann_id = 1
    img_id = 1

    for img_path in list_images(img_dir):
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            # skip unreadable images
            continue

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
            "license": 1
        })

        txt = lbl_dir / (img_path.stem + ".txt")
        if txt.exists():
            lines = [l for l in txt.read_text().splitlines() if l.strip()]
        else:
            lines = []

        for ln in lines:
            ann = yolo_line_to_coco(ln, w, h)
            if ann is None:
                continue
            ann["image_id"] = img_id
            ann["id"] = ann_id
            annotations.append(ann)
            ann_id += 1

        # IMPORTANT: if there are no lines, we still keep the image (zero anns)

        img_id += 1

    coco = {
        "info": {
            "description": "REDNET HAB detection — rebuilt from splits (keeps zero-annotation images)",
            "version": "1.0",
        },
        "licenses": [{"id": 1, "name": "unknown", "url": ""}],
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }

    out_path = out_dir / f"instances_{split}.json"
    out_path.write_text(json.dumps(coco, indent=2))
    print(f"{split:5s}: images={len(images):4d}, anns={len(annotations):4d} -> {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-root", default="data/labels/detection/images", type=Path)
    ap.add_argument("--lbl-root", default="data/labels/detection/labels", type=Path)
    ap.add_argument("--out",      default="data/labels/coco",            type=Path)
    args = ap.parse_args()

    for split in ("train", "val", "test"):
        build_split(split, args.img_root, args.lbl_root, args.out)

if __name__ == "__main__":
    main()
