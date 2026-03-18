import io
import logging

import torch
from PIL import Image

# Limite à 25 mégapixels pour éviter les bombes de décompression
Image.MAX_IMAGE_PIXELS = 25_000_000

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


def prepare_for_model(face_tensor: torch.Tensor) -> torch.Tensor:
    """
    Prépare un tensor visage MTCNN (C, H, W) en [0,255] pour inférence modèle.
    Resize 160->224 + normalisation ImageNet + batch dim.
    """
    # face_tensor est (C, H, W) en [0, 255]
    tensor = face_tensor.float() / 255.0

    # Resize de 160x160 vers 224x224
    tensor = tensor.unsqueeze(0)  # (1, C, H, W)
    tensor = torch.nn.functional.interpolate(tensor, size=TARGET_SIZE, mode="bilinear", align_corners=False)

    # Normaliser avec ImageNet stats
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor  # (1, C, 224, 224)
