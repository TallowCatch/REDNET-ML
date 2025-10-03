# src/tools/make_yolo_from_tiles.py
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import random
import shutil

TILES_DIR = Path("data/chl_tiles/tiles_png")   # your PNG tiles
OUT_IMG   = Path("data/labels/detection/images")
OUT_LBL   = Path("data/labels/detection/labels")
SPLITS    = {"train":0.7, "val":0.15, "test":0.15}

# thresholds (tweak if needed)
# we detect "bright" regions (candidate HAB patches) in grayscale 0..255
MIN_AREA_PIX   = 100    # drop tiny specks
BINARY_THRESH  = 180    # pixel > THR considered "bloom-ish" (try 170-200)
DILATE_ITERS   = 1      # mild dilation to merge specks

def yolo_from_bbox(xmin, ymin, xmax, ymax, W, H):
    # YOLO txt wants: class cx cy w h (normalized 0..1)
    cx = (xmin + xmax) / 2 / W
    cy = (ymin + ymax) / 2 / H
    w  = (xmax - xmin) / W
    h  = (ymax - ymin) / H
    return cx, cy, w, h

def main():
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_LBL.mkdir(parents=True, exist_ok=True)

    tiles = sorted(TILES_DIR.glob("*.png"))
    if not tiles:
        print(f"No PNG tiles in {TILES_DIR}"); return

    # split
    random.seed(42)
    random.shuffle(tiles)
    n = len(tiles)
    n_tr = int(SPLITS["train"]*n)
    n_va = int(SPLITS["val"]*n)
    splits = (["train"]*n_tr) + (["val"]*n_va) + (["test"]*(n-n_tr-n_va))

    counts = {"train":0,"val":0,"test":0}
    for tile, split in zip(tiles, splits):
        img = np.array(Image.open(tile).convert("L"))
        H, W = img.shape

        # threshold + morphology
        _, bw = cv2.threshold(img, BINARY_THRESH, 255, cv2.THRESH_BINARY)
        if DILATE_ITERS > 0:
            bw = cv2.dilate(bw, np.ones((3,3), np.uint8), iterations=DILATE_ITERS)

        # connected components → bounding boxes
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if w*h >= MIN_AREA_PIX:
                boxes.append((x,y,x+w,y+h))

        # copy image and write label (class 0: HAB)
        out_img = OUT_IMG / split / tile.name
        out_lbl = OUT_LBL / split / (tile.stem + ".txt")
        out_img.parent.mkdir(parents=True, exist_ok=True)
        out_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tile, out_img)

        with open(out_lbl, "w") as f:
            for (xmin,ymin,xmax,ymax) in boxes:
                cx,cy,w,h = yolo_from_bbox(xmin,ymin,xmax,ymax,W,H)
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        counts[split] += 1

    print("Wrote YOLO dataset:")
    for k,v in counts.items():
        print(f"  {k}: {v} images  →  {OUT_IMG/k} / {OUT_LBL/k}")
    print("Class map: 0=HAB")
    print("Tip: adjust BINARY_THRESH / MIN_AREA_PIX if you get too many/few boxes.")
if __name__ == "__main__":
    main()
