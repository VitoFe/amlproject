"""
DINO ViT-S/16 Model Implementation

This module provides the DINO Vision Transformer model pretrained on ImageNet,
adapted for CIFAR-100 classification.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import copy


class DinoViTClassifier(nn.Module):
    """
    DINO ViT-S/16 with a classification head for CIFAR-100.
    
    The DINO pretrained backbone provides strong visual representations.
    We add a linear classification head for the downstream task.
    """
    
    def __init__(
        self,
        num_classes: int = 100,
        freeze_backbone: bool = False,
        hidden_dim: int = 384,  # ViT-S/16 embedding dimension
        dropout: float = 0.1,   # Dropout for regularization
        freeze_layers: int = 0   # Number of transformer blocks to freeze (0-12)
    ):
        """
        Initialize DINO ViT classifier.
        
        Args:
            num_classes: Number of output classes
            freeze_backbone: If True, freeze entire backbone (linear probe)
            hidden_dim: Hidden dimension of ViT (384 for ViT-S)
            dropout: Dropout probability for classifier head
            freeze_layers: Number of early transformer blocks to freeze (0 = none, 12 = all)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self.dropout_rate = dropout
        
        # Load DINO ViT-S/16 backbone
        self.backbone = torch.hub.load(
            'facebookresearch/dino:main',
            'dino_vits16',
            pretrained=True
        )
        
        # Classification head with dropout for regularization
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Initialize classifier
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        
        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()
        elif freeze_layers > 0:
            self._freeze_early_layers(freeze_layers)
    
    def _freeze_backbone(self):
        """Freeze entire backbone for linear probing."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _freeze_early_layers(self, num_layers: int):
        """
        Freeze early transformer blocks while keeping later ones trainable.
        
        This helps prevent overfitting by limiting the number of trainable parameters.
        
        Args:
            num_layers: Number of early blocks to freeze (1-12)
        """
        # Freeze patch embedding and positional embedding
        if hasattr(self.backbone, 'patch_embed'):
            for param in self.backbone.patch_embed.parameters():
                param.requires_grad = False
        
        # Freeze cls token and pos embed
        if hasattr(self.backbone, 'cls_token'):
            self.backbone.cls_token.requires_grad = False
        if hasattr(self.backbone, 'pos_embed'):
            self.backbone.pos_embed.requires_grad = False
        
        # Freeze specified number of transformer blocks
        if hasattr(self.backbone, 'blocks'):
            for i, block in enumerate(self.backbone.blocks):
                if i < num_layers:
                    for param in block.parameters():
                        param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            Logits [B, num_classes]
        """
        # Extract features from DINO backbone
        # DINO returns the [CLS] token embedding
        features = self.backbone(x)
        
        # Apply dropout and classify
        features = self.dropout(features)
        logits = self.classifier(features)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            Features [B, hidden_dim]
        """
        return self.backbone(x)
    
    def get_trainable_params(self) -> Dict[str, nn.Parameter]:
        """
        Get dictionary of trainable parameters.
        
        Returns:
            Dict mapping param name to parameter
        """
        return {name: p for name, p in self.named_parameters() if p.requires_grad}
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count total and trainable parameters.
        
        Returns:
            Dict with parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }


def create_dino_vit(
    num_classes: int = 100,
    freeze_backbone: bool = False,
    dropout: float = 0.1,
    freeze_layers: int = 0,
    device: str = 'cuda'
) -> DinoViTClassifier:
    """
    Factory function to create DINO ViT classifier.
    
    Args:
        num_classes: Number of output classes
        freeze_backbone: Whether to freeze entire backbone
        dropout: Dropout probability for classifier
        freeze_layers: Number of early transformer blocks to freeze (0-12)
        device: Device to place model on
    
    Returns:
        DinoViTClassifier model
    """
    model = DinoViTClassifier(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
        freeze_layers=freeze_layers
    )
    
    return model.to(device)


def get_model_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Get a detached copy of model state dict.
    
    Args:
        model: PyTorch model
    
    Returns:
        Detached state dict
    """
    return {k: v.clone().detach() for k, v in model.state_dict().items()}


def set_model_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor]
) -> None:
    """
    Load state dict into model.
    
    Args:
        model: PyTorch model
        state_dict: State dict to load
    """
    model.load_state_dict(state_dict)


def compute_model_delta(
    model: nn.Module,
    reference_state: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Compute the difference between current model state and reference.
    
    This is the "task vector" in task arithmetic terminology.
    
    Args:
        model: Current model
        reference_state: Reference state dict (e.g., pre-trained weights)
    
    Returns:
        Delta state dict (current - reference)
    """
    current_state = model.state_dict()
    
    delta = {}
    for key in reference_state:
        delta[key] = current_state[key] - reference_state[key]
    
    return delta


def apply_model_delta(
    model: nn.Module,
    reference_state: Dict[str, torch.Tensor],
    delta: Dict[str, torch.Tensor],
    scale: float = 1.0
) -> None:
    """
    Apply a scaled delta to reference state and load into model.
    
    new_state = reference + scale * delta
    
    Args:
        model: Model to update
        reference_state: Reference state dict
        delta: Delta to apply
        scale: Scaling factor for delta
    """
    new_state = {}
    for key in reference_state:
        new_state[key] = reference_state[key] + scale * delta[key]
    
    model.load_state_dict(new_state)
