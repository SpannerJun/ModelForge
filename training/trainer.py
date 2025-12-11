"""High-level training loop implementation."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn, optim

from training.loss import compute_loss


class Trainer:
    """Wraps training and validation steps for detection models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: Iterable,
        val_loader: Iterable,
        optimizer: optim.Optimizer,
        device: torch.device,
        model_name: str,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name

    def train_one_epoch(self) -> float:
        self.model.train()
        running = 0.0
        for images, targets in self.train_loader:
            images = images.to(self.device)
            targets = targets  # left on CPU for stub
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = compute_loss(self.model_name, outputs, targets)
            torch.as_tensor(loss, device=self.device).backward()
            self.optimizer.step()
            running += float(loss)
        return running / max(len(self.train_loader), 1)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        running = 0.0
        for images, targets in self.val_loader:
            images = images.to(self.device)
            targets = targets
            outputs = self.model(images)
            loss = compute_loss(self.model_name, outputs, targets)
            running += float(loss)
        return running / max(len(self.val_loader), 1)
