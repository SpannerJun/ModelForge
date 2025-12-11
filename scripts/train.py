"""Entry point for model training."""

from __future__ import annotations

import torch
from torch import optim

from data.data_loader import create_dataloader
from models.detr import DETR
from models.yolo import YOLO
from training.config import config
from training.scheduler import build_scheduler
from training.trainer import Trainer


def build_model(name: str):
    name = name.lower()
    if name == "detr":
        return DETR(num_classes=config["num_classes"], num_queries=config["num_queries"])
    if name.startswith("yolo"):
        return YOLO(num_classes=config["num_classes"])
    raise ValueError(f"Unknown model: {name}")


def main() -> None:
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = build_model(config["model_name"]).to(device)

    # Placeholder annotations and loaders for demonstration
    annotations = {"sample.jpg": []}
    train_loader = create_dataloader("data/train", annotations, batch_size=config["batch_size"], num_workers=config["num_workers"], train=True)
    val_loader = create_dataloader("data/val", annotations, batch_size=config["batch_size"], num_workers=config["num_workers"], train=False)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = build_scheduler(optimizer)

    trainer = Trainer(model, train_loader, val_loader, optimizer, device, config["model_name"])

    for epoch in range(config["num_epochs"]):
        train_loss = trainer.train_one_epoch()
        val_loss = trainer.validate()
        scheduler.step()
        print(f"Epoch {epoch:03d} | train: {train_loss:.4f} | val: {val_loss:.4f}")


if __name__ == "__main__":
    main()
