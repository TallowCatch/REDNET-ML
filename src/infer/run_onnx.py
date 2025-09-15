# src/infer/run_onnx.py
import argparse, onnxruntime as ort, numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(p, hw=(224,224)):
    img = Image.open(p).convert("L")                     # grayscale
    img = Image.merge("RGB", (img, img, img))            # 3-ch
    img = img.resize(hw, Image.BILINEAR)
    x = np.asarray(img).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2,0,1))[None, ...]              # NCHW
    return x

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--log_target", type=int, default=1)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    x = preprocess(args.img)
    y = sess.run(["pred"], {"images": x})[0]             # shape [1,1]
    pred = float(y[0,0])
    if args.log_target:
        pred = np.expm1(pred)                            # back to mg/m^3
    print(f"Predicted chlorophyll-a: {pred:.3f} mg/m^3")
