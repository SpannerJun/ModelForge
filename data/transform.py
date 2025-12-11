"""Composable transforms for training and evaluation."""

from __future__ import annotations

from typing import Iterable

from torchvision import transforms


def build_transforms(train: bool = True, extra: Iterable | None = None) -> transforms.Compose:
    """Create a torchvision transform pipeline.

    Args:
        train: Whether to include augmentation suitable for training.
        extra: Optional iterable of additional transforms to append.
    """

    augmentations = []
    if train:
        augmentations.extend([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(),
        ])

    base = [
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    if extra:
        base.extend(extra)

    return transforms.Compose([*augmentations, *base])
