# src/torchvision_det/ds_torchvision.py
from pathlib import Path
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

DATA_DIR = Path("data")
IMG_DIR = DATA_DIR / "labels" / "detection" / "images"
COCO_DIR = DATA_DIR / "labels" / "coco"

class CocoDict:
    def __init__(self, coco_json):
        self.imgs = {im["id"]: im for im in coco_json["images"]}
        self.anns_by_img = {}
        for ann in coco_json["annotations"]:
            self.anns_by_img.setdefault(ann["image_id"], []).append(ann)

class HABDetDataset(Dataset):
    def __init__(self, split="train", transforms=None):
        coco_path = COCO_DIR / f"instances_{split}.json"
        coco = CocoDict(json.loads(coco_path.read_text()))
        self.ids = list(coco.imgs.keys())  # COCO numeric image ids
        self.coco = coco
        self.split = split
        self.transforms = transforms or T.ToTensor()

    def __len__(self): 
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]  # <-- use COCO img id, not the index
        info = self.coco.imgs[img_id]
        img_path = (IMG_DIR / self.split) / info["file_name"]
        img = Image.open(img_path).convert("RGB")

        anns = self.coco.anns_by_img.get(img_id, [])
        boxes, labels, area, iscrowd = [], [], [], []
        for a in anns:
            boxes.append(a["bbox"])         # COCO [x, y, w, h]
            labels.append(a["category_id"])
            area.append(a["area"])
            iscrowd.append(a.get("iscrowd", 0))

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            # to torchvision format [x1, y1, x2, y2]
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": torch.tensor(labels if labels else [], dtype=torch.int64),
            "image_id": torch.tensor([int(img_id)]),  # <-- fixed
            "area": torch.tensor(area if area else [], dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd if iscrowd else [], dtype=torch.int64),
        }
        img = self.transforms(img)
        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))
