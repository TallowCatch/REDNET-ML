# src/torchvision_det/ds_aerial.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F

class ResizeWithBoxes:
    def __init__(self, size: Tuple[int,int]): self.size = size
    def __call__(self, img: Image.Image, target: Dict):
        w0, h0 = img.size; img = img.resize(self.size, Image.BILINEAR)
        sx, sy = self.size[0]/w0, self.size[1]/h0
        if "boxes" in target:
            b = target["boxes"].clone()
            b[:, [0,2]] *= sx; b[:, [1,3]] *= sy
            target = {**target, "boxes": b}
        return F.to_tensor(img), target

def collate_fn(batch):  # same as your current
    imgs, tgts = list(zip(*batch))
    return list(imgs), list(tgts)

class AerialDetDataset(Dataset):
    def __init__(self, split="train", root="data", transforms=None, filter_empty=True):
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.filter_empty = filter_empty

        coco_path = self.root / "labels" / "coco" / f"instances_{split}.json"
        with open(coco_path, "r") as f:
            coco = json.load(f)

        self.id2img = {im["id"]: im for im in coco["images"]}
        self.anns_by_img = {}
        for ann in coco["annotations"]:
            if ann["image_id"] not in self.anns_by_img:
                self.anns_by_img[ann["image_id"]] = []
            self.anns_by_img[ann["image_id"]].append(ann)

        # image dir (you can symlink all tiles here or use a curated subset)
        self.img_dir = self.root / "labels" / "detection" / "images" / split
        self.ids = sorted(list(self.id2img.keys()))

        if self.filter_empty:
            self.ids = [i for i in self.ids if len(self.anns_by_img.get(i, [])) > 0]

    def __len__(self): return len(self.ids)

    def _load_img(self, file_name):
        p = self.img_dir / file_name
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        return Image.open(p).convert("RGB")

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.id2img[img_id]
        img = self._load_img(info["file_name"])
        anns = self.anns_by_img.get(img_id, [])

        boxes, labels, area, iscrowd = [], [], [], []
        for a in anns:
            x,y,w,h = a["bbox"]
            boxes.append([x,y,x+w,y+h])
            labels.append(a["category_id"])
            area.append(w*h); iscrowd.append(a.get("iscrowd", 0))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([img_id], dtype=torch.int64).squeeze(0),
            "area": torch.tensor(area, dtype=torch.float32) if area else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
        }

        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target
