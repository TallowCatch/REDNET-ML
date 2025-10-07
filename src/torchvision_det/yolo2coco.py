# src/torchvision_det/yolo2coco.py
from __future__ import annotations
import json, re
from pathlib import Path
from PIL import Image

# --- paths (run from repo root) ---
BASE = Path(".")
IMG_DIR = BASE / "data" / "chl_tiles" / "tiles_png"        # where your PNG tiles live
LBL_ROOT = BASE / "data" / "labels" / "detection"          # will be searched recursively
OUT_DIR = BASE / "data" / "labels" / "coco"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [{"id": 1, "name": "HAB"}]
IMG_EXTS = {".png", ".jpg", ".jpeg"}

def all_label_files(root: Path):
    return sorted(p for p in root.rglob("*.txt") if p.is_file())

def find_image_for_stem(stem: str):
    for ext in IMG_EXTS:
        p = IMG_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def parse_yolo_file(txt_path: Path, w: int, h: int):
    anns = []
    lines = [ln.strip() for ln in txt_path.read_text().splitlines() if ln.strip()]
    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        # YOLO: cls cx cy bw bh (normalized)
        try:
            cls, cx, cy, bw, bh = map(float, parts[:5])
        except Exception:
            continue
        x = (cx - bw/2.0) * w
        y = (cy - bh/2.0) * h
        ww = max(1.0, bw * w)
        hh = max(1.0, bh * h)
        anns.append({
            "category_id": int(cls) + 1,  # YOLO 0 -> COCO 1
            "bbox": [max(0.0, x), max(0.0, y), ww, hh],
            "area": float(ww * hh),
            "iscrowd": 0,
            "segmentation": [],
        })
    return anns

def build_coco():
    images, annotations = [], []
    ann_id = 1
    img_id = 1

    label_files = all_label_files(LBL_ROOT)
    stems_with_labels = set()
    for lf in label_files:
        stems_with_labels.add(lf.stem)

    used = 0
    for stem in sorted(stems_with_labels):
        img_path = find_image_for_stem(stem)
        if img_path is None:
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        # find the *best* matching label file for this stem (prefer same-name in any subdir)
        # if multiple exist, we merge their boxes.
        anns = []
        for lf in [p for p in label_files if p.stem == stem]:
            anns += parse_yolo_file(lf, w, h)
        if not anns:
            continue

        images.append({
            "id": img_id,
            "file_name": img_path.name,  # ds_torchvision loads from IMG_DIR + this name
            "width": w, "height": h,
        })
        for a in anns:
            a["image_id"] = img_id
            a["id"] = ann_id
            annotations.append(a)
            ann_id += 1

        img_id += 1
        used += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
        "info": {"description": "HAB (YOLO→COCO, recursive)", "version": "1.0"},
        "licenses": []
    }
    return coco

if __name__ == "__main__":
    if not IMG_DIR.exists():
        raise SystemExit(f"Missing images dir: {IMG_DIR.resolve()}")
    if not LBL_ROOT.exists():
        raise SystemExit(f"Missing labels root: {LBL_ROOT.resolve()}")

    coco = build_coco()
    out = OUT_DIR / "instances_all.json"
    out.write_text(json.dumps(coco, indent=2))
    print(f"Wrote {out}")
    print(f"  images with labels: {len(coco['images'])}")
    print(f"  annotations:       {len(coco['annotations'])}")
