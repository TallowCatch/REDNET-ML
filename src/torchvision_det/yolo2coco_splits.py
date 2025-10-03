# src/torchvision_det/yolo2coco_splits.py
from pathlib import Path
import json, random
from PIL import Image

BASE = Path("data")
IMG_ROOT = BASE / "labels" / "detection" / "images"
LBL_ROOT = BASE / "labels" / "detection" / "labels"
OUT_DIR  = BASE / "labels" / "coco"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]
CATEGORIES = [{"id": 1, "name": "HAB"}]

INFO = {
    "description": "REDNET HAB detection (YOLO→COCO)",
    "version": "1.0",
    "year": 2025,
}
LICENSES = [{"id": 1, "name": "unknown", "url": ""}]

def list_images(split):
    exts = {".png", ".jpg", ".jpeg"}
    return sorted([p for p in (IMG_ROOT / split).glob("*") if p.suffix.lower() in exts])

def yolo_to_coco_boxes(txt_path, w, h):
    anns = []
    if not txt_path.exists():
        return anns
    for line in txt_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        # YOLO: cls cx cy bw bh (all normalized)
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = map(float, parts)
        x = (cx - bw/2) * w
        y = (cy - bh/2) * h
        anns.append({
            "category_id": 1,
            "bbox": [max(0, x), max(0, y), max(1, bw*w), max(1, bh*h)],
            "area": (bw*w) * (bh*h),
            "iscrowd": 0
        })
    return anns

def build_split(split):
    images, annotations = [], []
    ann_id = 1
    img_id = 1  # ensure numeric ids
    for img_path in list_images(split):
        w, h = Image.open(img_path).size
        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w, "height": h, "license": 1
        })
        txt_path = LBL_ROOT / split / (img_path.stem + ".txt")
        for a in yolo_to_coco_boxes(txt_path, w, h):
            a["image_id"] = img_id
            a["id"] = ann_id
            annotations.append(a)
            ann_id += 1
        img_id += 1

    coco = {
        "info": INFO,
        "licenses": LICENSES,
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES
    }
    out = OUT_DIR / f"instances_{split}.json"
    out.write_text(json.dumps(coco, indent=2))
    print(f"{split:5s}: images={len(images):3d}, anns={len(annotations):4d} -> {out}")

if __name__ == "__main__":
    for s in SPLITS:
        build_split(s)
