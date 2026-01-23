import logging
import random
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Labels pour les attributs faciaux
AGE_LABELS = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "60+"]
GENDER_LABELS = ["Male", "Female"]
ETHNICITY_LABELS = ["White", "Black", "Asian", "Indian", "Other"]
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


class ModelManager:
    """Gestionnaire de chargement et d'inférence des modèles PyTorch."""

    def __init__(self, models_dir: str = "./models", device: str = "cpu"):
        self.models_dir = Path(models_dir)
        self.device = device
        self.models: dict = {}
        self._demo_mode = False

    def load_all(self):
        """Charge tous les modèles disponibles ou active le mode démo."""
        self._load_specialized()
        self._load_multitask()
        self._load_transfer()

        if not self.models:
            logger.warning("No models found. Running in DEMO mode.")
            self._demo_mode = True
        else:
            logger.info("Loaded %d models", len(self.models))

    def _load_specialized(self):
        """Charge les modèles spécialisés (un par attribut)."""
        specialized_dir = self.models_dir / "specialized"
        if not specialized_dir.exists():
            return
        for model_path in specialized_dir.glob("*.pth"):
            try:
                model = torch.load(
                    model_path,
                    map_location=self.device,
                    weights_only=False,
                )
                if hasattr(model, "eval"):
                    model.eval()
                self.models[f"specialized_{model_path.stem}"] = model
                logger.info("Loaded specialized model: %s", model_path.stem)
            except Exception as e:
                logger.error("Failed to load %s: %s", model_path, e)

    def _load_multitask(self):
        """Charge le modèle multitâche."""
        multitask_dir = self.models_dir / "multitask"
        if not multitask_dir.exists():
            return
        for model_path in multitask_dir.glob("*.pth"):
            try:
                model = torch.load(
                    model_path,
                    map_location=self.device,
                    weights_only=False,
                )
                if hasattr(model, "eval"):
                    model.eval()
                self.models["multitask"] = model
                logger.info("Loaded multitask model: %s", model_path.name)
            except Exception as e:
                logger.error("Failed to load multitask model: %s", e)

    def _load_transfer(self):
        """Charge le modèle transfer learning."""
        transfer_dir = self.models_dir / "transfer"
        if not transfer_dir.exists():
            return
        for model_path in transfer_dir.glob("*.pth"):
            try:
                model = torch.load(
                    model_path,
                    map_location=self.device,
                    weights_only=False,
                )
                if hasattr(model, "eval"):
                    model.eval()
                self.models["transfer"] = model
                logger.info("Loaded transfer model: %s", model_path.name)
            except Exception as e:
                logger.error("Failed to load transfer model: %s", e)

    def predict_specialized(self, face_tensor: torch.Tensor) -> list[dict]:
        """Prédiction avec modèles spécialisés."""
        if self._demo_mode:
            return self._demo_predictions()
        # TODO: inférence réelle avec les modèles specialized_*
        return self._demo_predictions()

    def predict_multitask(self, face_tensor: torch.Tensor) -> list[dict]:
        """Prédiction avec modèle multitâche."""
        if self._demo_mode:
            return self._demo_predictions()
        # TODO: inférence réelle avec self.models["multitask"]
        return self._demo_predictions()

    def predict_transfer(self, face_tensor: torch.Tensor) -> list[dict]:
        """Prédiction avec modèle transfer learning."""
        if self._demo_mode:
            return self._demo_predictions()
        # TODO: inférence réelle avec self.models["transfer"]
        return self._demo_predictions()

    def _demo_predictions(self) -> list[dict]:
        """Génère des prédictions démo (mode sans modèles)."""
        age = random.choice(AGE_LABELS)
        gender = random.choice(GENDER_LABELS)
        ethnicity = random.choice(ETHNICITY_LABELS)
        emotion = random.choice(EMOTION_LABELS)

        return [
            {
                "label": "age",
                "value": age,
                "confidence": round(random.uniform(0.6, 0.95), 3),
            },
            {
                "label": "gender",
                "value": gender,
                "confidence": round(random.uniform(0.7, 0.99), 3),
            },
            {
                "label": "ethnicity",
                "value": ethnicity,
                "confidence": round(random.uniform(0.5, 0.9), 3),
            },
            {
                "label": "emotion",
                "value": emotion,
                "confidence": round(random.uniform(0.5, 0.92), 3),
            },
        ]

    @property
    def loaded_count(self) -> int:
        return len(self.models)

    @property
    def is_demo_mode(self) -> bool:
        return self._demo_mode
