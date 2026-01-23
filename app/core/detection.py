import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class MTCNNDetector:
    """Détecteur de visages basé sur MTCNN."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.detector = None
        # TODO: initialiser MTCNN depuis facenet_pytorch

    def load(self):
        """Charge le modèle MTCNN."""
        # TODO: from facenet_pytorch import MTCNN
        # self.detector = MTCNN(device=self.device, keep_all=False)
        logger.info("MTCNN detector loaded on %s", self.device)

    def detect(self, image: Image.Image) -> dict | None:
        """
        Détecte un visage dans l'image.

        Args:
            image: Image PIL en entrée.

        Returns:
            Dict avec 'box' (coordonnées), 'confidence', 'face' (crop PIL)
            ou None si aucun visage détecté.
        """
        # TODO: implémenter détection
        # boxes, probs = self.detector.detect(image)
        # if boxes is None:
        #     return None
        # box = boxes[0]
        # face = image.crop(box)
        # return {"box": box.tolist(), "confidence": float(probs[0]), "face": face}
        logger.warning("detect() not implemented, returning None")
        return None

    def detect_and_align(self, image: Image.Image) -> np.ndarray | None:
        """
        Détecte, aligne et retourne le visage sous forme de tensor.

        Args:
            image: Image PIL en entrée.

        Returns:
            Tensor du visage aligné ou None.
        """
        # TODO: implémenter détection + alignement
        # return self.detector(image)
        logger.warning("detect_and_align() not implemented, returning None")
        return None
