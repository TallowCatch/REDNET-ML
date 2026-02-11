# src/infer/export_onnx.py
import argparse, torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

def build_model():
    m = mobilenet_v3_small(weights=None)
    in_feat = m.classifier[0].in_features
    m.classifier[-1] = nn.Identity()
    m.classifier = nn.Sequential(
        nn.Linear(in_feat, 128), nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(128, 1)
    )
    return m

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out",  default="rednet_regressor.onnx")
    ap.add_argument("--input_h", type=int, default=224)
    ap.add_argument("--input_w", type=int, default=224)
    args = ap.parse_args()

    device = torch.device("cpu")
    model = build_model().to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    dummy = torch.randn(1, 3, args.input_h, args.input_w, device=device)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["images"], output_names=["pred"],
        dynamic_axes={"images": {0: "batch"}, "pred": {0: "batch"}},
        opset_version=17, do_constant_folding=True
    )
    print(f"Exported ONNX → {args.out}")
