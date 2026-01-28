"""
DINO ViT-S/16 Model Implementation

This module provides the DINO Vision Transformer model pretrained on ImageNet,
adapted for CIFAR-100 classification.
"""

import torch
import torch.nn as nn
from typing import Dict


class DinoViTClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        freeze_backbone: bool = False,
        hidden_dim: int = 384,
        dropout: float = 0.1,
        freeze_layers: int = 0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self.dropout_rate = dropout

        self.backbone = torch.hub.load(
            "facebookresearch/dino:main", "dino_vits16", pretrained=True
        )

        # classification head with dropout for regularization
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        if freeze_backbone:
            self._freeze_backbone()
        elif freeze_layers > 0:
            self._freeze_early_layers(freeze_layers)

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _freeze_early_layers(self, num_layers: int):
        if hasattr(self.backbone, "patch_embed"):
            for param in self.backbone.patch_embed.parameters():
                param.requires_grad = False
        if hasattr(self.backbone, "cls_token"):
            self.backbone.cls_token.requires_grad = False
        if hasattr(self.backbone, "pos_embed"):
            self.backbone.pos_embed.requires_grad = False
        if hasattr(self.backbone, "blocks"):
            for i, block in enumerate(self.backbone.blocks):
                if i < num_layers:
                    for param in block.parameters():
                        param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_trainable_params(self) -> Dict[str, nn.Parameter]:
        return {name: p for name, p in self.named_parameters() if p.requires_grad}

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {"total": total, "trainable": trainable, "frozen": total - trainable}


def create_dino_vit(
    num_classes: int = 100,
    freeze_backbone: bool = False,
    dropout: float = 0.1,
    freeze_layers: int = 0,
    device: str = "cuda",
) -> DinoViTClassifier:
    model = DinoViTClassifier(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
        freeze_layers=freeze_layers,
    )

    return model.to(device)


def get_model_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.clone().detach() for k, v in model.state_dict().items()}


def set_model_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state_dict)


def compute_model_delta(
    model: nn.Module, reference_state: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    current_state = model.state_dict()

    delta = {}
    for key in reference_state:
        delta[key] = current_state[key] - reference_state[key]

    return delta


def apply_model_delta(
    model: nn.Module,
    reference_state: Dict[str, torch.Tensor],
    delta: Dict[str, torch.Tensor],
    scale: float = 1.0,
) -> None:
    new_state = {}
    for key in reference_state:
        new_state[key] = reference_state[key] + scale * delta[key]

    model.load_state_dict(new_state)
