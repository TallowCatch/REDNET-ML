# src/torchvision_det/yolo2coco.py
import json
from pathlib import Path
from PIL import Image

# ---- PATHS (run from repo root: REDNET-ML/) ----
BASE     = Path(".")                                  # <— repo root
IMG_DIR  = BASE / "data" / "chl_tiles" / "tiles_png"  # images
YOLO_DIR = BASE / "data" / "labels" / "detection"     # yolo .txts
OUT_DIR  = BASE / "data" / "labels" / "coco"          # coco output
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Single class (HAB). COCO category ids start at 1.
CATEGORIES = [{"id": 1, "name": "HAB"}]
EXTS = {".png", ".jpg", ".jpeg"}

def _ensure_layout():
    missing = [p for p in [IMG_DIR, YOLO_DIR] if not p.exists()]
    if missing:
        msg = "Missing required path(s):\n" + "\n".join(f"  - {p}" for p in missing)
        msg += "\nRun this from the REDNET-ML repo root."
        raise SystemExit(msg)

def list_images():
    return sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in EXTS])

def yolo_txt_to_boxes(txt_path, w, h):
    boxes = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, bw, bh = map(float, parts[:5])
        x  = (cx - bw/2.0) * w
        y  = (cy - bh/2.0) * h
        ww = bw * w
        hh = bh * h
        boxes.append({
            "category_id": int(cls) + 1,        # YOLO 0 -> COCO 1
            "bbox": [max(0.0, x), max(0.0, y), max(1.0, ww), max(1.0, hh)],
            "area": float(ww * hh),
            "iscrowd": 0,
            "segmentation": [],
        })
    return boxes

def build_coco(img_paths):
    images, annotations = [], []
    ann_id = 1
    for img_id, img_path in enumerate(img_paths, start=1):
        with Image.open(img_path) as im:
            w, h = im.size
        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })
        txt = YOLO_DIR / f"{img_path.stem}.txt"
        for b in yolo_txt_to_boxes(txt, w, h):
            b["image_id"] = img_id
            b["id"] = ann_id
            annotations.append(b)
            ann_id += 1
    return {"images": images, "annotations": annotations, "categories": CATEGORIES}

def main():
    _ensure_layout()
    imgs = list_images()
    if not imgs:
        raise SystemExit(f"No images found in {IMG_DIR}")
    coco = build_coco(imgs)
    out = OUT_DIR / "instances_all.json"
    out.write_text(json.dumps(coco, indent=2))
    print(f"Wrote {out} | images={len(coco['images'])}, anns={len(coco['annotations'])}")

if __name__ == "__main__":
    main()
