import io
import logging

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

TARGET_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Charge une image depuis des bytes bruts."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def validate_image(image_bytes: bytes, max_size_mb: int = 10) -> None:
    """Valide la taille de l'image."""
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"Image too large: {size_mb:.1f}MB (max {max_size_mb}MB)")


def normalize_face_tensor(face_tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalise un tensor visage (C, H, W) avec mean/std ImageNet.
    Le tensor en entrée est en [0, 255] (sortie MTCNN post_process=False).
    """
    # Convertir en [0, 1]
    tensor = face_tensor.float() / 255.0

    # Normaliser par canal
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    tensor = (tensor - mean) / std

    # Ajouter dimension batch
    return tensor.unsqueeze(0)
