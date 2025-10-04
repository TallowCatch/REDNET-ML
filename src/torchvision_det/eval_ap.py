from src.torchvision_det.mps_patch import *  # sets MPS fallback + CPU NMS
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must be before torchvision import

import json, torch, torchvision
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from .ds_torchvision import HABDetDataset, collate_fn

def load_frcnn_resnet50(weights, num_classes=2, device='cpu'):
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_feats = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, num_classes)
    m.load_state_dict(torch.load(weights, map_location=device))
    m.to(device).eval()
    return m

@torch.no_grad()
def infer_to_coco(model, split='val', device='cpu', img_size=640):
    import torchvision.transforms as T
    tfm = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    from .ds_torchvision import HABDetDataset, collate_fn
    ds = HABDetDataset(split, transforms=tfm)   # <- same resize as training
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    results = []
    for imgs, targets in dl:
        img = imgs[0].to(device)
        image_id = int(targets[0]["image_id"].item())
        pred = model([img])[0]
        # do NOT filter by score; COCOeval uses them
        for (x1, y1, x2, y2), s, c in zip(pred['boxes'].cpu(), pred['scores'].cpu(), pred['labels'].cpu()):
            results.append({
                "image_id": image_id,
                "category_id": int(c),                          # 1 = HAB
                "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],  # xywh
                "score": float(s)
            })
    return results
# the path to change
def eval_split(weights='runs/detect/frcnn/best.pt', split='val', device=None, cache_dir='runs/detect/coco_cache'):
    device = device or ('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available()
                        else 'cpu')

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_dir) / f"pred_{Path(weights).parent.name}_{split}.json"

    if cache_path.exists():
        results = json.loads(cache_path.read_text())
    else:
        model = load_frcnn_resnet50(weights, device=device)
        results = infer_to_coco(model, split, device)
        cache_path.write_text(json.dumps(results))

    ann_path = Path('data/labels/coco') / f'instances_{split}.json'
    coco_gt = COCO(str(ann_path))
    coco_dt = coco_gt.loadRes(results) if results else coco_gt.loadRes([])
    E = COCOeval(coco_gt, coco_dt, iouType='bbox')
    E.evaluate(); E.accumulate(); E.summarize()
    return {"mAP50-95": E.stats[0], "mAP50": E.stats[1]}

if __name__ == "__main__":
    print(eval_split())
