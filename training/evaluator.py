"""Evaluation utilities for detection models."""

from __future__ import annotations

from typing import Dict

import torch

from models.utils import nms


def evaluate(outputs: Dict[str, torch.Tensor], score_threshold: float = 0.5, iou_threshold: float = 0.5):
    """Basic evaluation stub running NMS and thresholding."""
    predictions = outputs.get("predictions")
    scores = torch.rand(predictions.shape[1]) if predictions is not None else torch.tensor([])
    selected = nms(predictions[0, 0], scores, iou_threshold) if predictions is not None else []
    return {"selected_indices": selected, "score_threshold": score_threshold}
