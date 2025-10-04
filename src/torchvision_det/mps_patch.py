# src/torchvision_det/mps_patch.py
"""
Force torchvision NMS to run on CPU on Apple Silicon (MPS) and enable
CPU fallback for any other unsupported ops.
Import this module BEFORE importing torchvision models.
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
import torchvision  # noqa: E402
from torchvision.ops import boxes as box_ops  # noqa: E402
from torchvision.ops import nms as tv_nms  # noqa: E402


def _nms_cpu(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float):
    """
    Wrapper that moves inputs to CPU, runs torchvision NMS, returns indices (on CPU).
    RPN only needs the indices; device does not matter for the return.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64)
    return tv_nms(boxes.to("cpu"), scores.to("cpu"), iou_threshold)


# Patch the function used internally by RPN -> box_ops.batched_nms -> box_ops.nms
box_ops.nms = _nms_cpu
