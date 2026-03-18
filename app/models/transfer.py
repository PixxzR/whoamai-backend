"""Modèle transfer learning : ResNet18 avec couches early gelées, 3 têtes fine-tuned."""

import torch
import torch.nn as nn
from torchvision import models


class TransferModel(nn.Module):
    """ResNet18 avec les premières couches gelées et 3 têtes fine-tuned."""

    def __init__(self, num_genders: int = 2, num_ethnicities: int = 5, freeze_until: int = 6):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Extraire les couches du backbone
        self.early_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )
        self.late_layers = nn.Sequential(
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = resnet.avgpool
        in_features = resnet.fc.in_features  # 512

        # Geler les couches early
        for param in self.early_layers.parameters():
            param.requires_grad = False

        # Têtes de prédiction (fine-tuned)
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
        with torch.no_grad():
            features = self.early_layers(x)
        features = self.late_layers(features)
        features = self.avgpool(features).flatten(1)

        age = self.age_head(features).squeeze(1)
        gender = self.gender_head(features)
        ethnicity = self.ethnicity_head(features)

        return {"age": age, "gender": gender, "ethnicity": ethnicity}
