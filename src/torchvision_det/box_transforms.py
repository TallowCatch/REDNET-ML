# src/torchvision_det/box_transforms.py
from typing import Tuple
from PIL import Image
import torchvision.transforms.functional as F
import torch

class ResizeWithBoxes:
    """
    Resize a PIL image to (W,H) and scale target['boxes'] (xyxy, pixels) accordingly.
    Use BEFORE ToTensor().
    """
    def __init__(self, size: Tuple[int, int] | int):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = (int(size[0]), int(size[1]))

    def __call__(self, img: Image.Image, target: dict):
        assert isinstance(img, Image.Image), "ResizeWithBoxes expects PIL image"
        w0, h0 = img.size
        img = F.resize(img, self.size, antialias=True)
        w1, h1 = self.size

        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"].clone().float()
            sx, sy = w1 / w0, h1 / h0
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
            target = dict(target)
            target["boxes"] = boxes
        return img, target

class ComposeDet:
    """
    Compose transforms that act on BOTH (img, target) BEFORE ToTensor.
    """
    def __init__(self, tfms):
        self.tfms = tfms

    def __call__(self, img, target):
        for t in self.tfms:
            img, target = t(img, target)
        return img, target
