# src/infer/batch_onnx.py
import argparse, onnxruntime as ort
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd

def load_img(fp, h=224, w=224):
    im = Image.open(fp).convert("L").resize((w, h), Image.BILINEAR)
    arr = np.array(im, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5           # normalize like training
    arr = np.stack([arr, arr, arr], axis=0)  # 3-ch
    return arr[None, ...]             # [1,3,H,W]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tiles_dir", default="data/chl_tiles/tiles_png")
    ap.add_argument("--out_csv", default="runs/onnx_preds.csv")
    ap.add_argument("--log_target", type=int, default=1)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    rows = []
    tiles = sorted(Path(args.tiles_dir).glob("*.png"))
    for i, fp in enumerate(tiles, 1):
        x = load_img(fp)
        y = sess.run([out_name], {in_name: x})[0].ravel()[0]
        if args.log_target:
            y = float(np.exp(y))  # invert log transform
        rows.append({"filepath": str(fp), "pred_chl": y})
        if i % 500 == 0:
            print(f"{i}/{len(tiles)}")

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Wrote predictions → {args.out_csv}")
