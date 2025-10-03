# src/torchvision_det/eval_ap.py
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json, torch, torchvision
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
import torchvision.transforms as T
from .ds_torchvision import HABDetDataset, collate_fn

def load_model(weights, num_classes=2, device='cpu'):
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_feats = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, num_classes)
    m.load_state_dict(torch.load(weights, map_location=device))
    m.to(device).eval()
    return m

@torch.no_grad()
def infer_to_coco(model, coco_gt, split='val', img_size=640, device='cpu'):
    # NOTE: square resize will distort aspect ratio; we correct by scaling back
    tfm = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    ds = HABDetDataset(split, transforms=tfm)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    results = []
    for imgs, targets in dl:
        img = imgs[0].to(device)
        image_id = int(targets[0]["image_id"])
        pred = model([img])[0]

        # original size from COCO
        im_info = coco_gt.imgs[image_id]
        W0, H0 = float(im_info["width"]), float(im_info["height"])
        sx, sy = W0 / float(img_size), H0 / float(img_size)

        boxes = pred['boxes'].cpu().tolist()
        scores = pred['scores'].cpu().tolist()
        labels = pred['labels'].cpu().tolist()
        for (x1, y1, x2, y2), s, c in zip(boxes, scores, labels):
            # scale back to original size and convert to [x, y, w, h]
            x1o, y1o, x2o, y2o = x1 * sx, y1 * sy, x2 * sx, y2 * sy
            results.append({
                "image_id": image_id,
                "category_id": int(c),
                "bbox": [float(x1o), float(y1o), float(x2o - x1o), float(y2o - y1o)],
                "score": float(s),
            })
    return results

def eval_split(weights='runs/detect/frcnn/best.pt', split='val', img_size=640, device=None):
    device = device or ('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available()
                        else 'cpu')
    model = load_model(weights, device=device)

    ann_path = Path('data/labels/coco') / f'instances_{split}.json'
    coco_gt = COCO(str(ann_path))

    results = infer_to_coco(model, coco_gt, split, img_size, device)
    coco_dt = coco_gt.loadRes(results) if results else coco_gt.loadRes([])

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    return {"mAP50-95": float(coco_eval.stats[0]), "mAP50": float(coco_eval.stats[1])}

if __name__ == "__main__":
    print(eval_split())
