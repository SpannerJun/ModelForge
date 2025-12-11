"""Head modules for detection architectures."""

from __future__ import annotations

from torch import nn


class YoloHead(nn.Module):
    """A minimal detection head with adjustable channels."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, (num_classes + 5) * 3, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class TransformerHead(nn.Module):
    """Transformer output projection for DETR-like models."""

    def __init__(self, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, hs):
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        return {"pred_logits": outputs_class, "pred_boxes": outputs_coord}


class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        layers = []
        for i in range(num_layers):
            dim_in = input_dim if i == 0 else hidden_dim
            dim_out = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(dim_in, dim_out))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
