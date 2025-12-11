"""Backbone factory for detection models."""

from __future__ import annotations

import torchvision
from torch import nn


def build_resnet_backbone(name: str = "resnet50", pretrained: bool = True) -> nn.Sequential:
    """Load a torchvision ResNet and drop classification layers."""
    backbone = getattr(torchvision.models, name)(weights="DEFAULT" if pretrained else None)
    return nn.Sequential(*list(backbone.children())[:-2])
