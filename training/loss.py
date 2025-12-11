"""Loss dispatchers for detection models."""

from __future__ import annotations

from typing import Dict

from models.detr import detr_loss
from models.yolo import yolo_loss


def compute_loss(model_name: str, outputs, targets) -> float:
    """Select an appropriate loss implementation based on model name."""
    name = model_name.lower()
    if name == "detr":
        return float(detr_loss(outputs, targets))
    if name.startswith("yolo"):
        return float(yolo_loss(outputs, targets))
    raise ValueError(f"Unsupported model name: {model_name}")
