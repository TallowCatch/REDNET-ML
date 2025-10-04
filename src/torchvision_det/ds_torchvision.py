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
DATA_DIR   = Path("data")
COCO_DIR   = DATA_DIR / "labels" / "coco"
IMG_ROOT   = DATA_DIR / "labels" / "detection" / "images"   # has train/val/test subfolders

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
            # normalize id to str to avoid type mismatches
            im_id = str(im["id"])
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

def ids_with_boxes(coco: CocoIndex) -> List[str]:
    """Return image ids that have at least one annotation."""
    return [img_id for img_id, anns in coco.anns_by_img.items() if len(anns) > 0]

# -----------------------------------------------------------------------------
# Resize transform that also rescales boxes
# -----------------------------------------------------------------------------
class ResizeWithBoxes:
    """Resize PIL image and scale [x1,y1,x2,y2] boxes accordingly."""
    def __init__(self, size: Tuple[int, int]):
        self.size = size  # (H, W)

    def __call__(self, img: Image.Image, target: Dict[str, torch.Tensor]):
        orig_w, orig_h = img.size
        new_h, new_w   = self.size
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)

        if "boxes" in target and target["boxes"].numel() > 0:
            sx = new_w / float(orig_w)
            sy = new_h / float(orig_h)
            boxes = target["boxes"].clone()
            boxes[:, 0] = boxes[:, 0] * sx
            boxes[:, 1] = boxes[:, 1] * sy
            boxes[:, 2] = boxes[:, 2] * sx
            boxes[:, 3] = boxes[:, 3] * sy
            target["boxes"] = boxes

        return img, target

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class HABDetDataset(Dataset):
    """
    Detection dataset for your HAB tiles (COCO-style labels).
    split: 'train' | 'val' | 'test'
    transforms: either a torchvision transform that returns a Tensor, or a
                callable that accepts (PIL, target) and returns (PIL, target),
                followed by ToTensor().
    """
    def __init__(self, split: str = "train", transforms=None):
        assert split in {"train", "val", "test"}
        self.split = split
        coco_path = COCO_DIR / f"instances_{split}.json"
        if not coco_path.exists():
            raise FileNotFoundError(f"Missing COCO json: {coco_path}")

        self.coco = CocoIndex(json.loads(coco_path.read_text()))
        self.ids: List[str] = list(self.coco.imgs.keys())

        # Optional quick-limit for debugging
        limit_n = int(os.getenv("LIMIT_N", "0") or "0")
        if limit_n > 0:
            self.ids = self.ids[:limit_n]

        # default: simple ToTensor
        self.img_only_tf = T.ToTensor()
        # transforms can be a callable that handles (img, target) joint ops
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, file_name: str) -> Image.Image:
        img_path = IMG_ROOT / self.split / file_name
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        return Image.open(img_path).convert("RGB")

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        info   = self.coco.imgs[img_id]
        file_name = info["file_name"]
        w, h   = int(info["width"]), int(info["height"])

        img = self._load_image(file_name)

        anns = self.coco.anns_by_img.get(img_id, [])
        boxes, labels, area, iscrowd = [], [], [], []
        for a in anns:
            xyxy = _xywh_to_xyxy(a["bbox"])
            xyxy = _clip_box_xyxy(xyxy, w, h)
            boxes.append(xyxy)
            labels.append(int(a.get("category_id", 1)))
            area.append(float(a.get("area", (xyxy[2]-xyxy[0])*(xyxy[3]-xyxy[1]))))
            iscrowd.append(int(a.get("iscrowd", 0)))

        if len(boxes) > 0:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            area_t = torch.tensor(area, dtype=torch.float32)
            iscrowd_t = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes_t   = torch.zeros((0, 4), dtype=torch.float32)
            labels_t  = torch.zeros((0,), dtype=torch.int64)
            area_t    = torch.zeros((0,), dtype=torch.float32)
            iscrowd_t = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([int(idx)]),  # kept as idx (COCO will map by file in eval)
            "area": area_t,
            "iscrowd": iscrowd_t,
            "orig_size": torch.tensor([h, w], dtype=torch.int64),
        }

        # joint transforms (e.g., resize that also rescales boxes)
        if isinstance(self.transforms, ResizeWithBoxes):
            img, target = self.transforms(img, target)
            img = self.img_only_tf(img)  # finally to tensor
        elif callable(self.transforms):
            # If user passed a Compose of custom callable(s) returning PIL,
            # make sure the last step gives us a Tensor.
            img = self.transforms(img)
            if not isinstance(img, torch.Tensor):
                img = self.img_only_tf(img)
        else:
            img = self.img_only_tf(img)

        return img, target

# -----------------------------------------------------------------------------
# Dataloader collate
# -----------------------------------------------------------------------------
def collate_fn(batch):
    imgs, tgts = zip(*batch)
    return list(imgs), list(tgts)
