"""Modèles spécialisés : un ResNet18 par attribut (age, genre, ethnicité)."""

import torch
import torch.nn as nn
from torchvision import models


class AgeModel(nn.Module):
    """ResNet18 pour régression d'âge (sortie scalaire)."""

    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(1)


class GenderModel(nn.Module):
    """ResNet18 pour classification de genre (2 classes)."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class EthnicityModel(nn.Module):
    """ResNet18 pour classification d'ethnicité (5 classes)."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
