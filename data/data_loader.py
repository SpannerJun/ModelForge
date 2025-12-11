"""DataLoader factory utilities."""

from __future__ import annotations

from typing import Dict, List

from torch.utils.data import DataLoader

from .dataset import Annotation, DetectionSample
from .transform import build_transforms


def create_dataloader(
    images_dir: str,
    annotations: Dict[str, List[Annotation]],
    batch_size: int = 4,
    num_workers: int = 4,
    train: bool = True,
) -> DataLoader:
    """Create a dataloader for detection tasks.

    Args:
        images_dir: Directory of images.
        annotations: Mapping of filenames to annotation lists.
        batch_size: Batch size for the dataloader.
        num_workers: Worker processes used by DataLoader.
        train: Whether to enable training augmentations.
    """

    dataset = DetectionSample(images_dir, annotations, transform=build_transforms(train=train))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)
