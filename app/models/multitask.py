"""Modèle multitâche : EfficientNet-B0 backbone partagé, 3 têtes."""

import torch
import torch.nn as nn
from torchvision import models


class MultitaskModel(nn.Module):
    """EfficientNet-B0 avec 3 têtes : age (régression), genre (2), ethnicité (5)."""

    def __init__(self, num_genders: int = 2, num_ethnicities: int = 5):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Extraire les features (sans le classifier)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        in_features = backbone.classifier[1].in_features  # 1280

        self.age_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.gender_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_genders),
        )

        self.ethnicity_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_ethnicities),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.features(x)
        features = self.avgpool(features).flatten(1)

        age = self.age_head(features).squeeze(1)
        gender = self.gender_head(features)
        ethnicity = self.ethnicity_head(features)

        return {"age": age, "gender": gender, "ethnicity": ethnicity}
