# src/torchvision_det/ds_torchvision.py
from __future__ import annotations
from pathlib import Path
import json
import os
from typing import Dict, List, Tuple, Any

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

# -----------------------------------------------------------------------------
# Project paths (matches your repo layout)
# -----------------------------------------------------------------------------
DATA_DIR = Path("data")
COCO_DIR = DATA_DIR / "labels" / "coco"
IMG_ROOT = DATA_DIR / "labels" / "detection" / "images"  # has train/val/test

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
class CocoIndex:
    """Tiny COCO indexer for fast image/annotation lookup."""
    def __init__(self, coco_json: Dict[str, Any]):
        # images: list of {"id": <str|int>, "file_name": "...", "width": W, "height": H}
        # annotations: list of {"image_id": <id>, "bbox": [x,y,w,h], "category_id": ...}
        self.imgs: Dict[str, Dict[str, Any]] = {}
        for im in coco_json["images"]:
            im_id = str(im["id"])  # normalize image_id to str internally
            self.imgs[im_id] = im

        self.anns_by_img: Dict[str, List[Dict[str, Any]]] = {}
        for ann in coco_json["annotations"]:
            k = str(ann["image_id"])
            self.anns_by_img.setdefault(k, []).append(ann)

        self.categories = coco_json.get("categories", [{"id": 1, "name": "HAB"}])

def _xywh_to_xyxy(xywh):
    x, y, w, h = xywh
    return [x, y, x + w, y + h]

def _clip_box_xyxy(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(w - 1), float(x1)))
    y1 = max(0.0, min(float(h - 1), float(y1)))
    x2 = max(0.0, min(float(w - 1), float(x2)))
    y2 = max(0.0, min(float(h - 1), float(y2)))
    return [x1, y1, x2, y2]

def _has_target(anns: List[Dict[str, Any]]) -> bool:
    """True if at least one bbox with positive size."""
    for a in anns:
        _, _, w, h = a["bbox"]
        if w > 1 and h > 1:
            return True
    return False

# -----------------------------------------------------------------------------
# Resize transform that also rescales boxes
# -----------------------------------------------------------------------------
class ResizeWithBoxes:
    """Resize PIL image and scale [x1,y1,x2,y2] boxes accordingly."""
    def __init__(self, size: Tuple[int, int]):
        self.size = size  # (H, W)

    def __call__(self, img: Image.Image, target: Dict[str, torch.Tensor]):
        orig_w, orig_h = img.size
        new_h, new_w = self.size
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)

        if "boxes" in target and target["boxes"].numel() > 0:
            sx = new_w / float(orig_w)
            sy = new_h / float(orig_h)
            boxes = target["boxes"].clone()
            boxes[:, 0] *= sx
            boxes[:, 1] *= sy
            boxes[:, 2] *= sx
            boxes[:, 3] *= sy
            target["boxes"] = boxes

        return img, target

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class HABDetDataset(Dataset):
    """
    Detection dataset for HAB tiles (COCO-style labels).
    split: 'train' | 'val' | 'test'
    transforms: either None, or a callable that accepts (PIL, target) and
                returns (PIL, target); ToTensor() is applied afterwards.
    filter_empty: if True, drop images without any bbox (training only).
    """
    def __init__(self, split: str = "train", transforms=None, filter_empty: bool = False):
        assert split in {"train", "val", "test"}
        self.split = split
        coco_path = COCO_DIR / f"instances_{split}.json"
        if not coco_path.exists():
            raise FileNotFoundError(f"Missing COCO json: {coco_path}")

        self.coco = CocoIndex(json.loads(coco_path.read_text()))
        ids = list(self.coco.imgs.keys())
        if filter_empty:
            ids = [i for i in ids if _has_target(self.coco.anns_by_img.get(i, []))]

        # Optional quick-limit for debugging
        limit_n = int(os.getenv("LIMIT_N", "0") or "0")
        if limit_n > 0:
            ids = ids[:limit_n]

        self.ids: List[str] = ids
        self.img_only_tf = T.ToTensor()
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, file_name: str) -> Image.Image:
        # Try several layout candidates
        candidates = [
            Path("data/labels/detection/images") / self.split / file_name,  # expected layout
            Path("data/labels") / self.split / file_name,                   # your current "val" folder
            Path("data/chl_tiles/tiles_png") / file_name,                   # raw tile dump, just in case
        ]
        for p in candidates:
            if p.exists():
                return Image.open(p).convert("RGB")
        raise FileNotFoundError(f"Image not found for {file_name}. Tried: " + " | ".join(map(str, candidates)))



    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        info = self.coco.imgs[img_id]
        file_name = info["file_name"]
        w, h = int(info["width"]), int(info["height"])

        img = self._load_image(file_name)

        anns = self.coco.anns_by_img.get(img_id, [])
        boxes, labels, area, iscrowd = [], [], [], []
        for a in anns:
            xyxy = _xywh_to_xyxy(a["bbox"])
            xyxy = _clip_box_xyxy(xyxy, w, h)
            boxes.append(xyxy)
            labels.append(int(a.get("category_id", 1)))  # 1 = HAB
            area.append(float(a.get("area", (xyxy[2]-xyxy[0])*(xyxy[3]-xyxy[1]))))
            iscrowd.append(int(a.get("iscrowd", 0)))

        if len(boxes) > 0:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            area_t = torch.tensor(area, dtype=torch.float32)
            iscrowd_t = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            area_t = torch.zeros((0,), dtype=torch.float32)
            iscrowd_t = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            # IMPORTANT: use the true COCO image id (int), not the dataset index
            "image_id": torch.tensor(int(info["id"])),
            "area": area_t,
            "iscrowd": iscrowd_t,
            "orig_size": torch.tensor([h, w], dtype=torch.int64),
        }

        # joint transforms (e.g., resize that also rescales boxes)
        if isinstance(self.transforms, ResizeWithBoxes):
            img, target = self.transforms(img, target)
            img = self.img_only_tf(img)
        elif callable(self.transforms):
            # if user passed a PIL-only transform, ensure Tensor at the end
            x = self.transforms(img)
            img = x if isinstance(x, torch.Tensor) else self.img_only_tf(x)
        else:
            img = self.img_only_tf(img)

        return img, target

# -----------------------------------------------------------------------------
# Dataloader collate
# -----------------------------------------------------------------------------
def collate_fn(batch):
    imgs, tgts = zip(*batch)
    return list(imgs), list(tgts)
