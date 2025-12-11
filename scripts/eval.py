"""Quick validation loop for saved checkpoints."""

from __future__ import annotations

import torch

from data.data_loader import create_dataloader
from models.detr import DETR
from models.utils import load_checkpoint
from training.config import config
from training.evaluator import evaluate


def main() -> None:
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = DETR(num_classes=config["num_classes"], num_queries=config["num_queries"]).to(device)
    try:
        load_checkpoint(model, "output/model_weights/model.pth")
    except FileNotFoundError:
        print("No checkpoint found; skipping weight loading.")
    model.eval()

    annotations = {"sample.jpg": []}
    val_loader = create_dataloader("data/val", annotations, batch_size=1, num_workers=0, train=False)

    for images, _ in val_loader:
        images = images.to(device)
        outputs = model(images)
        metrics = evaluate(outputs)
        print(metrics)


if __name__ == "__main__":
    main()
