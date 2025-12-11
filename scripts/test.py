"""Run evaluation on the test split."""

from __future__ import annotations

import torch

from data.data_loader import create_dataloader
from models.detr import DETR
from training.config import config
from training.evaluator import evaluate


def main() -> None:
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = DETR(num_classes=config["num_classes"], num_queries=config["num_queries"]).to(device)
    model.eval()

    annotations = {"sample.jpg": []}
    test_loader = create_dataloader("data/test", annotations, batch_size=1, num_workers=0, train=False)

    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        metrics = evaluate(outputs)
        print(metrics)


if __name__ == "__main__":
    main()
