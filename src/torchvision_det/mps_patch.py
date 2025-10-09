"""
MPS helpers for TorchVision detection:
- Force CPU fallback for NMS (and keep dtypes consistent)
- Provide a robust CPU fallback for ROI Align that always returns float32
Import this module BEFORE constructing TorchVision detection models.
"""

from __future__ import annotations
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

# -------------------------
# NMS -> force CPU fallback
# -------------------------
try:
    from torchvision.ops import boxes as _box_ops

    if not hasattr(_box_ops, "_orig_nms"):
        _box_ops._orig_nms = _box_ops.nms

    def _nms_cpu_fallback(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float):
        # Ensure float32 on CPU for torchvision's NMS kernel
        b_cpu = boxes.detach().to("cpu", dtype=torch.float32)
        s_cpu = scores.detach().to("cpu", dtype=torch.float32)
        keep_cpu = _box_ops._orig_nms(b_cpu, s_cpu, float(iou_threshold))  # int64 indices
        return keep_cpu.to(boxes.device)

    _box_ops.nms = _nms_cpu_fallback
except Exception:
    pass

# -------------------------------------------
# ROI Align -> robust CPU fallback (float32)
# -------------------------------------------
try:
    import torchvision.ops as _ops_mod
    from torchvision.ops import roi_align as _roi_align_orig
    import torchvision.ops.poolers as _poolers_mod

    def roi_align_cpu_fallback(
        input: torch.Tensor,
        boxes,
        output_size,
        spatial_scale: float = 1.0,
        sampling_ratio: int = -1,
        aligned: bool = False,
    ):
        """
        Try native op first; if it errors on MPS, run on CPU in float32 and
        return float32 back to the original device.
        """
        try:
            return _roi_align_orig(input, boxes, output_size, spatial_scale, sampling_ratio, aligned)
        except Exception:
            dev = input.device
            inp_cpu = input.detach().to("cpu", dtype=torch.float32)
            if isinstance(boxes, (list, tuple)):
                boxes_cpu = [b.detach().to("cpu", dtype=torch.float32) for b in boxes]
            else:
                boxes_cpu = boxes.detach().to("cpu", dtype=torch.float32)
            out = _roi_align_orig(inp_cpu, boxes_cpu, output_size, spatial_scale, sampling_ratio, aligned)
            return out.to(dev, dtype=torch.float32)

    # Monkey-patch both entry points used by detection heads
    _ops_mod.roi_align = roi_align_cpu_fallback
    _poolers_mod.roi_align = roi_align_cpu_fallback
except Exception:
    pass
