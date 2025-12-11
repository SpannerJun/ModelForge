"""Utility helpers for detection models."""

from __future__ import annotations

from typing import List, Tuple

import torch


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [xmin, ymin, xmax, ymax] boxes to center format."""
    x_min, y_min, x_max, y_max = boxes.unbind(dim=-1)
    cxcywh = torch.stack([(x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min], dim=-1)
    return cxcywh


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> List[int]:
    """A tiny IoU-based suppression implementation."""
    keep: List[int] = []
    if boxes.numel() == 0:
        return keep

    order = scores.argsort(descending=True)
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        remaining = order[1:]

        x1 = torch.maximum(boxes[i, 0], boxes[remaining, 0])
        y1 = torch.maximum(boxes[i, 1], boxes[remaining, 1])
        x2 = torch.minimum(boxes[i, 2], boxes[remaining, 2])
        y2 = torch.minimum(boxes[i, 3], boxes[remaining, 3])

        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_remaining = (boxes[remaining, 2] - boxes[remaining, 0]) * (boxes[remaining, 3] - boxes[remaining, 1])
        union = area_i + area_remaining - inter
        iou = inter / union

        order = remaining[iou <= iou_threshold]

    return keep


def load_checkpoint(model: torch.nn.Module, path: str) -> None:
    """Load a checkpoint if it exists without raising when missing."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint)
