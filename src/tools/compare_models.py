from __future__ import annotations
import os, time, argparse, csv, traceback, json
from pathlib import Path

# --- Mac MPS hygiene
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch, torchvision
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.models.detection.rpn import AnchorGenerator

from src.torchvision_det.mps_patch import *  # CPU NMS/ROI fallback on MPS
from src.torchvision_det.ds_torchvision import HABDetDataset, ResizeWithBoxes, collate_fn

# ---------- device ----------
def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

# ---------- model builders (no anchors yet) ----------
def build_frcnn_resnet50(num_classes=2):
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_feats = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, num_classes)
    return m

def build_frcnn_mobilenet(num_classes=2):
    m = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    in_feats = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, num_classes)
    return m

def build_ssd_mobilenet(num_classes=2):
    # matches your training config
    return torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=None, weights_backbone="DEFAULT", num_classes=num_classes
    )

# ---------- anchor config that adapts to backbone at runtime ----------
@torch.no_grad()
def configure_rpn_anchors_runtime(model, img_size, base_sizes=(8,16,32,64,128), ratios=(0.5,1.0,2.0), device="cpu"):
    """
    Probe the backbone to discover how many FPN levels it outputs,
    then set AnchorGenerator with exactly that many (size,) tuples.
    NO-OP for models without RPN (e.g., SSD).
    """
    # If it's not a Faster R-CNN-style model, bail out gracefully.
    if not hasattr(model, "rpn") or not hasattr(model.rpn, "anchor_generator"):
        return None

    model.eval()
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)
    feats = model.backbone(dummy)
    if isinstance(feats, dict):
        n_levels = len(feats)
    elif isinstance(feats, (list, tuple)):
        n_levels = len(feats)
    else:
        n_levels = 1

    sizes   = tuple((s,) for s in base_sizes[:n_levels])
    aspects = (tuple(ratios),) * n_levels
    model.rpn.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspects)
    return n_levels

ARCHES = {
    "frcnn_resnet50": {"builder": build_frcnn_resnet50, "img_size": 640},
    "frcnn_mobilenet": {"builder": build_frcnn_mobilenet, "img_size": 320},
    "ssd_mobilenet":   {"builder": build_ssd_mobilenet,   "img_size": 320},
}

# ---------- helpers ----------
def _ensure_coco_info(ann_path: Path):
    with ann_path.open("r") as f:
        data = json.load(f)
    if "info" not in data:
        data["info"] = {"description": "HAB dataset"}
        with ann_path.open("w") as f:
            json.dump(data, f)

@torch.no_grad()
def coco_eval(model, split="val", img_size=640, device=torch.device("cpu")):
    tfm = ResizeWithBoxes((img_size, img_size))
    ds = HABDetDataset(split, transforms=tfm, filter_empty=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    results = []
    for imgs, tgts in dl:
        img = imgs[0].to(device)
        coco_img_id = int(tgts[0]["image_id"].item())
        pred = model([img])[0]
        boxes = pred.get("boxes", torch.empty(0))
        scores= pred.get("scores", torch.empty(0))
        labels= pred.get("labels", torch.empty(0))
        if boxes.numel() == 0:
            continue
        for (x1,y1,x2,y2), s, c in zip(
            boxes.detach().cpu().tolist(),
            scores.detach().cpu().tolist(),
            labels.detach().cpu().tolist()
        ):
            results.append({
                "image_id": coco_img_id,
                "category_id": int(c),
                "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                "score": float(s),
            })

    ann_path = Path("data/labels/coco") / f"instances_{split}.json"
    _ensure_coco_info(ann_path)
    coco_gt = COCO(str(ann_path))
    coco_dt = coco_gt.loadRes(results) if results else coco_gt.loadRes([])
    E = COCOeval(coco_gt, coco_dt, iouType="bbox")
    E.evaluate(); E.accumulate(); E.summarize()
    return float(E.stats[0]), float(E.stats[1])  # mAP50-95, mAP50

@torch.no_grad()
def measure_fps(model, img_size=640, device=torch.device("cpu"), warmup=5, runs=50):
    model.eval()
    x = torch.rand(1, 3, img_size, img_size, device=device)
    for _ in range(warmup): model([x[0]])
    t0 = time.time()
    for _ in range(runs): model([x[0]])
    return runs / (time.time() - t0)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--outdir", default="runs/compare")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--limit_n", type=int, default=0)
    ap.add_argument("--weights_r50", default="runs/detect/frcnn_resnet50/best.pt")
    ap.add_argument("--weights_mn",  default="runs/detect/frcnn_mobilenet/best.pt")
    ap.add_argument("--weights_ssd", default="runs/detect/ssd_mobilenet/best.pt")
    args = ap.parse_args()

    if args.limit_n > 0:
        os.environ["LIMIT_N"] = str(args.limit_n)

    dev = get_device()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("Faster R-CNN ResNet50-FPN", "frcnn_resnet50", args.weights_r50),
        ("Faster R-CNN MobileNetV3-320", "frcnn_mobilenet", args.weights_mn),
        ("SSD-Lite MobileNetV3-320", "ssd_mobilenet", args.weights_ssd),
    ]

    rows = []
    for nice_name, arch_key, wpath in jobs:
        print(f"\n=== {nice_name} ===", flush=True)
        info = ARCHES[arch_key]
        img_size = info["img_size"]
        builder  = info["builder"]

        rec = {
            "model": nice_name, "arch": arch_key, "img_size": img_size,
            "params_M": None, "FPS": None, "mAP50-95": None, "mAP50": None,
            "weights": str(wpath), "error": "",
        }

        try:
            assert Path(wpath).exists(), f"weights not found: {wpath}"

            # 1) build on CPU
            m = builder(num_classes=2).cpu().eval()

            # 2) ONLY configure anchors if the model has an RPN (i.e., Faster R-CNN)
            if hasattr(m, "rpn") and hasattr(m.rpn, "anchor_generator"):
                configure_rpn_anchors_runtime(m, img_size, device="cpu")

            # 3) move to real device and load weights
            m = m.to(dev).eval()
            m.load_state_dict(torch.load(wpath, map_location=dev), strict=True)

            rec["params_M"] = round(count_params(m)/1e6, 2)
            rec["FPS"]      = round(measure_fps(m, img_size, dev, args.warmup, args.runs), 1)
            map_all, map50  = coco_eval(m, args.split, img_size, dev)
            rec["mAP50-95"] = round(map_all, 4)
            rec["mAP50"]    = round(map50, 4)

        except Exception as e:
            rec["error"] = repr(e)
            print("  !! error:", repr(e), flush=True)
            traceback.print_exc()

        rows.append(rec)

    # write artifacts even if some rows failed
    csv_path = outdir / "metrics.csv"
    md_path  = outdir / "metrics.md"

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    with md_path.open("w") as f:
        f.write("| Model | Arch | Img | Params(M) | FPS | mAP50-95 | mAP50 | Error |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            f.write(f"| {r['model']} | `{r['arch']}` | {r['img_size']} | {r['params_M']} | "
                    f"{r['FPS']} | {r['mAP50-95']} | {r['mAP50']} | {r['error']} |\n")

    print("\nSaved:")
    print("  -", csv_path)
    print("  -", md_path)
    print("\nPreview:")
    for r in rows:
        print(f"  {r['model']:<28} FPS={r['FPS']}  mAP50={r['mAP50']}  "
              f"mAP50-95={r['mAP50-95']}  Params={r['params_M']}M  err={r['error']}", flush=True)

if __name__ == "__main__":
    main()
