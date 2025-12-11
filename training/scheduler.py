"""Scheduler helpers."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR


def build_scheduler(optimizer: Optimizer, step_size: int = 10, gamma: float = 0.1) -> StepLR:
    """Create a simple StepLR scheduler."""
    return StepLR(optimizer, step_size=step_size, gamma=gamma)
