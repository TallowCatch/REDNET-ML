# src/torchvision_det/mps_patch.py
"""
Force torchvision NMS to run on CPU on Apple Silicon (MPS) and enable
CPU fallback for any other unsupported ops.
Import this module BEFORE importing torchvision models.
"""
# src/torchvision_det/mps_patch.py
# Simple, explicit CPU fallbacks for ops missing on MPS.
from __future__ import annotations
import torch
from torchvision.ops import nms as _tv_nms
from torchvision.ops import roi_align as _tv_roi_align

def _to_cpu(x):
    return x.detach().to("cpu") if isinstance(x, torch.Tensor) else x

def nms_cpu(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float):
    return _tv_nms(_to_cpu(boxes), _to_cpu(scores), iou_threshold)

def roi_align_cpu(
    input: torch.Tensor,
    boxes,
    output_size,
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
):
    # move feature maps + boxes to CPU, run roi_align, return to original device
    dev = input.device
    out = _tv_roi_align(
        _to_cpu(input),
        [b.to("cpu") for b in boxes],
        output_size,
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
        aligned=aligned,
    )
    return out.to(dev)

def enable_mps_fallbacks():
    # Monkeypatch where torchvision looks during Faster R-CNN forward passes
    import torchvision.ops as ops
    ops.nms = nms_cpu
    ops.roi_align = roi_align_cpu

# When imported, enable patches automatically
enable_mps_fallbacks()
