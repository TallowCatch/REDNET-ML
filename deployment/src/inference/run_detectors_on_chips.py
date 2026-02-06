#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import torch
import pandas as pd
from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont


# -------------------------------------------------
# utils
# -------------------------------------------------
def load_image(p: Path, img_size: int):
    img = Image.open(p).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return img, tfm(img)

def chip_id_from_path(p: Path) -> str:
    return p.stem

def max_score_from_outputs(outputs, score_thresh=0.05):
    if not outputs:
        return 0.0
    scores = outputs[0].get("scores")
    if scores is None or len(scores) == 0:
        return 0.0
    scores = scores.detach().cpu()
    scores = scores[scores >= score_thresh]
    return float(scores.max()) if len(scores) else 0.0

def extract_boxes(outputs, score_thresh=0.25):
    """
    Returns list of dicts: xmin,ymin,xmax,ymax,score,label
    """
    if not outputs:
        return []
    o = outputs[0]
    boxes = o.get("boxes")
    scores = o.get("scores")
    labels = o.get("labels")
    if boxes is None or scores is None or len(scores) == 0:
        return []

    boxes = boxes.detach().cpu()
    scores = scores.detach().cpu()
    labels = labels.detach().cpu() if labels is not None else torch.zeros(len(scores), dtype=torch.long)

    keep = scores >= float(score_thresh)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    out = []
    for b, s, lab in zip(boxes, scores, labels):
        x0, y0, x1, y1 = [float(v) for v in b.tolist()]
        out.append({
            "xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1,
            "score": float(s), "label": int(lab)
        })
    return out

def draw_boxes_on_image(img: Image.Image, dets: list[dict], color="red", width=3):
    im = img.copy().convert("RGB")
    draw = ImageDraw.Draw(im)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for d in dets:
        x0,y0,x1,y1 = d["xmin"], d["ymin"], d["xmax"], d["ymax"]
        for w in range(width):
            draw.rectangle([x0-w, y0-w, x1+w, y1+w], outline=color)
        txt = f'{d.get("label",0)}:{d.get("score",0.0):.2f}'
        draw.text((x0+2, max(0, y0-12)), txt, fill=color, font=font)

    return im


# -------------------------------------------------
# model builders (MATCH TRAINING EXACTLY)
# -------------------------------------------------
def build_frcnn_resnet50(num_classes: int, ckpt: Path, device: str):
    m = fasterrcnn_resnet50_fpn(weights=None)
    in_feats = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.to(device).eval()

def build_frcnn_mobilenet(num_classes: int, ckpt: Path, device: str):
    m = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_feats = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.to(device).eval()

def build_ssd(num_classes: int, ckpt: Path, device: str):
    m = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone="DEFAULT",
        num_classes=num_classes,
    )
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.to(device).eval()


# -------------------------------------------------
# detector runner
# -------------------------------------------------
def run_detector(
    model,
    chips_dir: Path,
    img_size: int,
    colname: str,
    device: str,
    *,
    score_thresh: float = 0.05,
    export_boxes_csv: Path | None = None,
    export_vis_dir: Path | None = None,
    vis_topk: int = 0,
    vis_min_score: float = 0.25,
):
    """
    Always returns per-chip max score table (chip_id, colname).
    Optionally exports:
      - boxes CSV (one row per detection)
      - annotated images for top-k chips by max score
    """
    rows = []
    det_rows = []

    img_paths = sorted(
        list(chips_dir.glob("*.jpg")) +
        list(chips_dir.glob("*.jpeg")) +
        list(chips_dir.glob("*.png")) +
        list(chips_dir.glob("*.JPG"))
    )

    if not img_paths:
        return pd.DataFrame(columns=["chip_id", colname])

    # First pass: compute max score per chip (cheap)
    chip_scores = []
    with torch.no_grad():
        for img_path in img_paths:
            img_pil, x = load_image(img_path, img_size)
            x = x.to(device)
            out = model([x])
            m = max_score_from_outputs(out, score_thresh=score_thresh)
            cid = chip_id_from_path(img_path)
            rows.append({"chip_id": cid, colname: m})
            chip_scores.append((cid, img_path, img_pil, out, m))

    df_scores = pd.DataFrame(rows)

    # If no exports requested, return now
    want_boxes = export_boxes_csv is not None
    want_vis = export_vis_dir is not None and vis_topk > 0

    if not want_boxes and not want_vis:
        return df_scores

    # Choose which chips to process for boxes/vis
    # - For boxes CSV, we export for ALL chips (unless you only want topk, easy tweak)
    # - For vis, we do top-k by max score
    chip_scores_sorted = sorted(chip_scores, key=lambda t: t[4], reverse=True)
    vis_set = set()
    if want_vis:
        vis_set = set([t[0] for t in chip_scores_sorted[:vis_topk]])

    # Second pass: extract boxes (can reuse out we already computed)
    for cid, img_path, img_pil, out, m in chip_scores:
        dets = extract_boxes(out, score_thresh=vis_min_score)

        if want_boxes:
            for d in dets:
                det_rows.append({
                    "chip_id": cid,
                    "model": colname,
                    "score_max": float(m),
                    **d,
                    "image_path": str(img_path),
                })

        if want_vis and cid in vis_set:
            vis_img = draw_boxes_on_image(img_pil, dets, color="red", width=3)
            export_vis_dir.mkdir(parents=True, exist_ok=True)
            vis_img.save(export_vis_dir / f"{cid}_{colname}.png", quality=95)

    if want_boxes:
        export_boxes_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(det_rows).to_csv(export_boxes_csv, index=False)

    return df_scores


# -------------------------------------------------
# main
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root_dir", required=True, help="Folder containing YYYY-MM subfolders")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--frcnn_r50", required=True)
    ap.add_argument("--frcnn_mb", required=True)
    ap.add_argument("--ssd_mb", required=True)

    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # NEW
    ap.add_argument("--score_thresh", type=float, default=0.05, help="threshold for max score aggregation")
    ap.add_argument("--export_boxes", action="store_true", help="export raw boxes CSV per month per model")
    ap.add_argument("--export_vis", action="store_true", help="export annotated images with boxes")
    ap.add_argument("--vis_topk", type=int, default=40, help="annotate top-k chips by max score")
    ap.add_argument("--vis_min_score", type=float, default=0.25, help="min score for a box to be drawn/exported")
    ap.add_argument("--vis_dirname", default="det_vis", help="folder name under out_dir for annotated images")

    args = ap.parse_args()

    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models ONCE
    frcnn_r50 = build_frcnn_resnet50(args.num_classes, Path(args.frcnn_r50), args.device)
    frcnn_mb  = build_frcnn_mobilenet(args.num_classes, Path(args.frcnn_mb), args.device)
    ssd_mb    = build_ssd(args.num_classes, Path(args.ssd_mb), args.device)

    # Traverse months
    for month_dir in sorted(root_dir.iterdir()):
        if not month_dir.is_dir():
            continue

        chips_dir = month_dir / "chips" / "tiles_png"
        if not chips_dir.exists():
            print(f"⏭️  Skipping {month_dir.name} (no chips/tiles_png)")
            continue

        print(f"\n📆 Processing {month_dir.name}")

        # optional output paths
        vis_dir = (out_dir / args.vis_dirname / month_dir.name) if args.export_vis else None

        # Run models
        dfs = []

        # FRCNN R50
        boxes_csv = (out_dir / f"{month_dir.name}_boxes_frcnn_r50.csv") if args.export_boxes else None
        dfs.append(
            run_detector(
                frcnn_r50, chips_dir, 640, "p_frcnn_r50_med", args.device,
                score_thresh=args.score_thresh,
                export_boxes_csv=boxes_csv,
                export_vis_dir=vis_dir,
                vis_topk=args.vis_topk,
                vis_min_score=args.vis_min_score,
            )
        )

        # FRCNN MB
        boxes_csv = (out_dir / f"{month_dir.name}_boxes_frcnn_mb.csv") if args.export_boxes else None
        dfs.append(
            run_detector(
                frcnn_mb, chips_dir, 320, "p_frcnn_mb_med", args.device,
                score_thresh=args.score_thresh,
                export_boxes_csv=boxes_csv,
                export_vis_dir=vis_dir,
                vis_topk=args.vis_topk,
                vis_min_score=args.vis_min_score,
            )
        )

        # SSD MB
        boxes_csv = (out_dir / f"{month_dir.name}_boxes_ssd_mb.csv") if args.export_boxes else None
        dfs.append(
            run_detector(
                ssd_mb, chips_dir, 320, "p_ssd_mb_med", args.device,
                score_thresh=args.score_thresh,
                export_boxes_csv=boxes_csv,
                export_vis_dir=vis_dir,
                vis_topk=args.vis_topk,
                vis_min_score=args.vis_min_score,
            )
        )

        det = dfs[0]
        for d in dfs[1:]:
            det = det.merge(d, on="chip_id", how="outer")
        det.fillna(0.0, inplace=True)

        out_csv = out_dir / f"{month_dir.name}_detector_scores.csv"
        det.to_csv(out_csv, index=False)
        print(f"✅ wrote {out_csv}")

    print("\n✅ Done.")

if __name__ == "__main__":
    main()
