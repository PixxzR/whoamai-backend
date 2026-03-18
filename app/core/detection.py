import logging

import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

logger = logging.getLogger(__name__)


class FaceDetector:
    """Détecteur de visages basé sur MTCNN."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.detector: MTCNN | None = None

    def load(self):
        """Charge le modèle MTCNN."""
        self.detector = MTCNN(
            image_size=160,
            margin=20,
            keep_all=False,
            device=self.device,
            post_process=False,
        )
        logger.info("MTCNN detector loaded on %s", self.device)

    def detect(self, image: Image.Image) -> dict | None:
        """
        Détecte un visage dans l'image.

        Returns:
            Dict avec 'box', 'confidence', 'face' (tensor)
            ou None si aucun visage détecté.
        """
        if self.detector is None:
            raise RuntimeError("Detector not loaded. Call load() first.")

        try:
            boxes, probs = self.detector.detect(image)
        except ValueError:
            # facenet_pytorch raises ValueError when no faces found
            return None

        if boxes is None or len(boxes) == 0:
            return None

        if probs is None:
            return None

        # keep_all=False -> une seule box
        box = boxes[0]
        confidence = float(probs[0]) if probs[0] is not None else 0.0

        # Extraire le visage aligné comme tensor
        face_tensor = self.detector.extract(image, np.array([box]), save_path=None)

        if face_tensor is None:
            return None

        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        return {
            "box": {
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
            },
            "confidence": confidence,
            "face_tensor": face_tensor[0],  # (C, H, W) tensor
        }
