# src/torchvision_det/eval_ap.py
import os, argparse, torch, torchvision
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from src.torchvision_det.mps_patch import *  # MPS fallbacks
from src.torchvision_det.ds_torchvision import HABDetDataset, collate_fn, ResizeWithBoxes

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

# ---- build models exactly like training ----
def build_frcnn_r50(num_classes=2):
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_feats = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, num_classes)
    return m

def build_frcnn_mn(num_classes=2):
    m = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    in_feats = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, num_classes)
    return m

def build_ssd(num_classes=2):
    return torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=None, weights_backbone="DEFAULT", num_classes=num_classes
    )

def build_model_for_arch(arch: str):
    if arch == "frcnn_resnet50": return build_frcnn_r50()
    if arch == "frcnn_mobilenet": return build_frcnn_mn()
    if arch == "ssd_mobilenet":   return build_ssd()
    raise ValueError(f"Unknown arch {arch}")

@torch.no_grad()
def infer_to_coco(model, coco_gt: COCO, split='val', device='cpu', img_size=640):
    """
    IMPORTANT: We run inference on resized images (img_size x img_size),
    then scale predictions back to the original (w,h) from COCO GT.
    """
    tfm = ResizeWithBoxes((img_size, img_size))
    ds = HABDetDataset(split, transforms=tfm, filter_empty=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    results = []
    for imgs, targets in dl:
        img = imgs[0].to(device)
        coco_img_id = int(targets[0]["image_id"].item())
        # original size from COCO GT
        info = coco_gt.imgs[coco_img_id]
        W0, H0 = float(info["width"]), float(info["height"])
        # current (resized) size = img_size x img_size by our transform
        sx = W0 / float(img_size)
        sy = H0 / float(img_size)

        pred = model([img])[0]
        boxes = pred.get("boxes")
        if boxes is None or boxes.numel() == 0:
            continue

        boxes  = boxes.detach().cpu()
        scores = pred["scores"].detach().cpu().tolist()
        labels = pred["labels"].detach().cpu().tolist()

        # x1,y1,x2,y2 (resized space) -> scale back to original, then to COCO [x,y,w,h]
        x1y1x2y2 = boxes.numpy()
        x1y1x2y2[:, [0,2]] *= sx
        x1y1x2y2[:, [1,3]] *= sy
        xywh = []
        for (x1, y1, x2, y2) in x1y1x2y2:
            x, y = float(x1), float(y1)
            w, h = float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))
            xywh.append([x, y, w, h])

        for b, s, c in zip(xywh, scores, labels):
            results.append({
                "image_id": coco_img_id,
                "category_id": int(c),     # should be 1 for HAB
                "bbox": [float(v) for v in b],
                "score": float(s)
            })
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--arch", type=str, required=True,
                    choices=["frcnn_resnet50","frcnn_mobilenet","ssd_mobilenet"])
    ap.add_argument("--split", type=str, default="val", choices=["train","val","test"])
    ap.add_argument("--img_size", type=int, default=640, help="training resize used (640 for R50, 320 for MN/SSD)")
    args = ap.parse_args()

    dev = get_device()
    model = build_model_for_arch(args.arch)
    state = torch.load(args.weights, map_location=dev)
    model.load_state_dict(state, strict=True)
    model.to(dev).eval()

    ann_path = Path("data/labels/coco") / f"instances_{args.split}.json"
    coco_gt = COCO(str(ann_path))
    if "info" not in coco_gt.dataset: coco_gt.dataset["info"] = {"description": "HAB tiles"}
    if "licenses" not in coco_gt.dataset: coco_gt.dataset["licenses"] = []
    coco_gt.createIndex()

    results = infer_to_coco(model, coco_gt, split=args.split, device=dev, img_size=args.img_size)

    coco_dt = coco_gt.loadRes(results) if results else coco_gt.loadRes([])
    E = COCOeval(coco_gt, coco_dt, iouType="bbox")
    E.evaluate(); E.accumulate(); E.summarize()
    print({"mAP50-95": E.stats[0], "mAP50": E.stats[1]})

if __name__ == "__main__":
    main()
