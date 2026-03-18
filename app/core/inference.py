import logging
import random
from pathlib import Path

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

GENDER_LABELS = ["Male", "Female"]
ETHNICITY_LABELS = ["White", "Black", "Asian", "Indian", "Other"]


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
            logger.info("Loaded %d model(s): %s", len(self.models), list(self.models.keys()))

    def _load_specialized(self):
        """Charge les modèles spécialisés via architecture class + load_state_dict."""
        from app.models.specialized import AgeModel, EthnicityModel, GenderModel

        specialized_dir = self.models_dir / "specialized"
        if not specialized_dir.exists():
            return

        model_map = {
            "age_model": AgeModel,
            "gender_model": GenderModel,
            "ethnicity_model": EthnicityModel,
        }

        for stem, cls in model_map.items():
            model_path = specialized_dir / f"{stem}.pth"
            if not model_path.exists():
                continue
            try:
                model = cls()
                model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
                model.to(self.device).eval()
                self.models[f"specialized_{stem}"] = model
                logger.info("Loaded specialized model: %s", stem)
            except Exception as e:
                logger.error("Failed to load %s: %s", model_path, e)

    def _load_multitask(self):
        """Charge le modèle multitâche."""
        from app.models.multitask import MultitaskModel

        model_path = self.models_dir / "multitask" / "multitask_model.pth"
        if not model_path.exists():
            return
        try:
            model = MultitaskModel()
            model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            model.to(self.device).eval()
            self.models["multitask"] = model
            logger.info("Loaded multitask model")
        except Exception as e:
            logger.error("Failed to load multitask model: %s", e)

    def _load_transfer(self):
        """Charge le modèle transfer learning."""
        from app.models.transfer import TransferModel

        model_path = self.models_dir / "transfer" / "transfer_model.pth"
        if not model_path.exists():
            return
        try:
            model = TransferModel()
            model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            model.to(self.device).eval()
            self.models["transfer"] = model
            logger.info("Loaded transfer model")
        except Exception as e:
            logger.error("Failed to load transfer model: %s", e)

    def predict_specialized(self, face_tensor: torch.Tensor) -> dict:
        """Prédiction avec modèles spécialisés."""
        if self._demo_mode or not any(k.startswith("specialized_") for k in self.models):
            return self._demo_predictions()

        from app.core.preprocessing import prepare_for_model

        input_tensor = prepare_for_model(face_tensor).to(self.device)
        result = {}

        # Age
        age_model = self.models.get("specialized_age_model")
        if age_model:
            with torch.no_grad():
                age_pred = age_model(input_tensor)
            result["age"] = round(float(age_pred.item()), 1)
        else:
            result["age"] = round(random.uniform(1, 80), 1)

        # Gender
        gender_model = self.models.get("specialized_gender_model")
        if gender_model:
            with torch.no_grad():
                gender_logits = gender_model(input_tensor)
                gender_probs = F.softmax(gender_logits, dim=1)
            gender_idx = int(gender_probs.argmax(1).item())
            result["gender"] = GENDER_LABELS[gender_idx]
            result["gender_confidence"] = round(float(gender_probs[0, gender_idx].item()), 3)
        else:
            result["gender"] = random.choice(GENDER_LABELS)
            result["gender_confidence"] = round(random.uniform(0.7, 0.99), 3)

        # Ethnicity
        eth_model = self.models.get("specialized_ethnicity_model")
        if eth_model:
            with torch.no_grad():
                eth_logits = eth_model(input_tensor)
                eth_probs = F.softmax(eth_logits, dim=1)
            eth_idx = int(eth_probs.argmax(1).item())
            result["ethnicity"] = ETHNICITY_LABELS[eth_idx]
            result["ethnicity_confidence"] = round(float(eth_probs[0, eth_idx].item()), 3)
            result["ethnicity_class_id"] = eth_idx
        else:
            eth_idx = random.randint(0, len(ETHNICITY_LABELS) - 1)
            result["ethnicity"] = ETHNICITY_LABELS[eth_idx]
            result["ethnicity_confidence"] = round(random.uniform(0.5, 0.9), 3)
            result["ethnicity_class_id"] = eth_idx

        return result

    def _predict_multihead(self, face_tensor: torch.Tensor, model_key: str) -> dict:
        """Prédiction commune pour modèles multi-têtes (multitask/transfer)."""
        if self._demo_mode or model_key not in self.models:
            return self._demo_predictions()

        from app.core.preprocessing import prepare_for_model

        model = self.models[model_key]
        input_tensor = prepare_for_model(face_tensor).to(self.device)

        with torch.no_grad():
            outputs = model(input_tensor)

        age = round(float(outputs["age"].item()), 1)

        gender_probs = F.softmax(outputs["gender"], dim=1)
        gender_idx = int(gender_probs.argmax(1).item())

        eth_probs = F.softmax(outputs["ethnicity"], dim=1)
        eth_idx = int(eth_probs.argmax(1).item())

        return {
            "age": age,
            "gender": GENDER_LABELS[gender_idx],
            "gender_confidence": round(float(gender_probs[0, gender_idx].item()), 3),
            "ethnicity": ETHNICITY_LABELS[eth_idx],
            "ethnicity_confidence": round(float(eth_probs[0, eth_idx].item()), 3),
            "ethnicity_class_id": eth_idx,
        }

    def predict_multitask(self, face_tensor: torch.Tensor) -> dict:
        """Prédiction avec modèle multitâche."""
        return self._predict_multihead(face_tensor, "multitask")

    def predict_transfer(self, face_tensor: torch.Tensor) -> dict:
        """Prédiction avec modèle transfer learning."""
        return self._predict_multihead(face_tensor, "transfer")

    def _demo_predictions(self) -> dict:
        """Génère des prédictions démo (mode sans modèles)."""
        gender_idx = random.randint(0, 1)
        ethnicity_idx = random.randint(0, len(ETHNICITY_LABELS) - 1)

        return {
            "age": round(random.uniform(1, 80), 1),
            "gender": GENDER_LABELS[gender_idx],
            "gender_confidence": round(random.uniform(0.7, 0.99), 3),
            "ethnicity": ETHNICITY_LABELS[ethnicity_idx],
            "ethnicity_confidence": round(random.uniform(0.5, 0.9), 3),
            "ethnicity_class_id": ethnicity_idx,
        }

    @property
    def loaded_count(self) -> int:
        return len(self.models)

    @property
    def is_demo_mode(self) -> bool:
        return self._demo_mode
