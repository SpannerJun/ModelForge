"""YOLO-style model skeleton."""

from __future__ import annotations

from typing import Dict

from torch import nn

from .backbone import build_resnet_backbone
from .heads import YoloHead


def yolo_loss(outputs: Dict[str, nn.Module], targets) -> nn.Module:  # type: ignore[override]
    """Placeholder YOLO loss for demonstration."""
    return sum(out.mean() for out in outputs.values()) if isinstance(outputs, dict) else outputs.mean()


class YOLO(nn.Module):
    """Simplified YOLO-style model with configurable head."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = build_resnet_backbone()
        self.head = YoloHead(in_channels=2048, num_classes=num_classes)

    def forward(self, images):
        features = self.backbone(images)
        return {"predictions": self.head(features)}
