"""Run inference on a single image."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models.detr import DETR
from models.utils import load_checkpoint, xyxy_to_cxcywh
from training.config import config


def load_image(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


def visualize_results(predictions):
    print("Predicted logits shape:", predictions["pred_logits"].shape)
    print("Predicted boxes shape:", predictions["pred_boxes"].shape)


def main(image_path: str) -> None:
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = DETR(num_classes=config["num_classes"], num_queries=config["num_queries"]).to(device)
    try:
        load_checkpoint(model, "output/model_weights/model.pth")
    except FileNotFoundError:
        print("No checkpoint found; using randomly initialized weights.")
    model.eval()

    image = load_image(image_path).to(device)
    with torch.no_grad():
        predictions = model(image)
    visualize_results(predictions)


if __name__ == "__main__":
    sample_image = Path("data/test/sample.jpg")
    main(str(sample_image))
