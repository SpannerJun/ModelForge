"""Lightweight DETR-style model skeleton."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import build_resnet_backbone
from .heads import TransformerHead


class MinimalTransformer(nn.Module):
    """Tiny transformer encoder-decoder stack for demonstration."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, num_layers: int = 3) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads)
        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.pos_embed = nn.Parameter(torch.rand(1, 1000, hidden_dim))

    def forward(self, src: torch.Tensor, query_embed: torch.Tensor) -> torch.Tensor:
        src = src.flatten(2).permute(2, 0, 1)
        memory = self.encoder(src + self.pos_embed[: src.size(0)])
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt + query_embed, memory)
        return hs.transpose(0, 1)


class DETR(nn.Module):
    """Minimal DETR wrapper with ResNet backbone and transformer head."""

    def __init__(self, num_classes: int, num_queries: int = 100, hidden_dim: int = 256) -> None:
        super().__init__()
        self.backbone = build_resnet_backbone()
        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.transformer = MinimalTransformer(hidden_dim=hidden_dim)
        self.head = TransformerHead(hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        proj = self.input_proj(features)
        hs = self.transformer(proj, self.query_embed.weight)
        return self.head(hs)


def detr_loss(outputs: Dict[str, torch.Tensor], targets: Any) -> torch.Tensor:
    """Stub loss mixing classification and bbox components."""
    logits = outputs["pred_logits"]
    boxes = outputs["pred_boxes"]
    cls_loss = F.cross_entropy(logits.flatten(0, 1), torch.zeros(logits.size(0) * logits.size(1), dtype=torch.long))
    l1_loss = F.l1_loss(boxes, torch.zeros_like(boxes))
    return cls_loss + l1_loss
