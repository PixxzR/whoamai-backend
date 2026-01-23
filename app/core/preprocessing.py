import io
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

TARGET_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Charge une image depuis des bytes bruts."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def resize_image(image: Image.Image, size: tuple[int, int] = TARGET_SIZE) -> Image.Image:
    """Redimensionne l'image à la taille cible."""
    return image.resize(size, Image.BILINEAR)


def normalize(image_array: np.ndarray) -> np.ndarray:
    """
    Normalise un array image (H, W, C) avec mean/std ImageNet.

    Args:
        image_array: Array numpy float32 en [0, 1].

    Returns:
        Array normalisé.
    """
    mean = np.array(MEAN, dtype=np.float32)
    std = np.array(STD, dtype=np.float32)
    return (image_array - mean) / std


def preprocess_face(image: Image.Image) -> np.ndarray:
    """
    Pipeline complet de preprocessing pour un visage détecté.

    Args:
        image: Image PIL du visage croppé.

    Returns:
        Array numpy prêt pour l'inférence (1, C, H, W).
    """
    # Resize
    image = resize_image(image)

    # To float array [0, 1]
    image_array = np.array(image, dtype=np.float32) / 255.0

    # Normalize
    image_array = normalize(image_array)

    # HWC -> CHW
    image_array = np.transpose(image_array, (2, 0, 1))

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array
