"""Dataset definitions for object detection and segmentation tasks.

This module provides dataset wrappers that convert raw annotation
structures into training-ready tensors. It is intentionally lightweight
so users can subclass and adapt to custom formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transform import build_transforms


@dataclass
class Annotation:
    """Represents a single object annotation.

    Attributes:
        bbox: Bounding box defined as (xmin, ymin, xmax, ymax).
        category_id: Integer category identifier.
    """

    bbox: Tuple[float, float, float, float]
    category_id: int


class DetectionSample(Dataset):
    """A minimal COCO-style dataset wrapper.

    Args:
        images_dir: Directory containing image files.
        annotations: Mapping from image filename to a list of annotations.
        transform: Optional callable applied to each sample.
    """

    def __init__(
        self,
        images_dir: Path,
        annotations: Dict[str, List[Annotation]],
        transform: Callable | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotations = annotations
        self.transform = transform or build_transforms()
        self._image_keys = sorted(self.annotations.keys())

    def __len__(self) -> int:
        return len(self._image_keys)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_id = self._image_keys[idx]
        image_path = self.images_dir / image_id
        image = Image.open(image_path).convert("RGB")

        target_annotations = self.annotations.get(image_id, [])
        boxes = torch.tensor([ann.bbox for ann in target_annotations], dtype=torch.float32)
        labels = torch.tensor([ann.category_id for ann in target_annotations], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": image_id}

        if self.transform:
            image = self.transform(image)

        return image, target
