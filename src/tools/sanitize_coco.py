from __future__ import annotations
from pathlib import Path
import json

COCO_DIR = Path("data/labels/coco")

# Where images might live for each split
ROOTS = [
    Path("data/labels/detection/images"),  # expected
    Path("data/labels"),                   # your current val/
    Path("data/chl_tiles/tiles_png"),      # raw tiles (no split subfolder)
]

def find_image(split: str, file_name: str) -> Path | None:
    # Try split subfolder roots
    for r in ROOTS[:2]:
        p = r / split / file_name
        if p.exists():
            return p
    # Try raw tiles (no split)
    p = ROOTS[2] / file_name
    return p if p.exists() else None

def sanitize(split: str):
    jpath = COCO_DIR / f"instances_{split}.json"
    coco = json.loads(jpath.read_text())
    imgs = coco["images"]
    anns = coco["annotations"]

    keep_img_ids = []
    new_images = []
    for im in imgs:
        fn = im["file_name"]
        if find_image(split, fn) is not None:
            new_images.append(im)
            keep_img_ids.append(im["id"])

    keep_img_ids = set(keep_img_ids)
    new_annotations = [a for a in anns if a["image_id"] in keep_img_ids]

    out = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco.get("categories", [{"id": 1, "name": "HAB"}]),
    }
    out_path = COCO_DIR / f"instances_{split}.sanitized.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[{split}] kept {len(new_images)} images, {len(new_annotations)} anns -> {out_path}")

if __name__ == "__main__":
    for s in ("train", "val", "test"):
        p = COCO_DIR / f"instances_{s}.json"
        if p.exists():
            sanitize(s)
