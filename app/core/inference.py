import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelManager:
    """Gestionnaire de chargement et d'inférence des modèles PyTorch."""

    def __init__(self, models_dir: str = "./models", device: str = "cpu"):
        self.models_dir = Path(models_dir)
        self.device = device
        self.models: dict = {}

    def load_all(self):
        """Charge tous les modèles disponibles."""
        self._load_specialized()
        self._load_multitask()
        self._load_transfer()
        logger.info("Loaded %d models total", len(self.models))

    def _load_specialized(self):
        """Charge les modèles spécialisés (un par attribut)."""
        specialized_dir = self.models_dir / "specialized"
        if not specialized_dir.exists():
            logger.warning("Specialized models directory not found: %s", specialized_dir)
            return
        # TODO: charger chaque .pth dans specialized_dir
        # for model_path in specialized_dir.glob("*.pth"):
        #     model = torch.load(model_path, map_location=self.device)
        #     model.eval()
        #     self.models[f"specialized_{model_path.stem}"] = model
        logger.info("Specialized models: not implemented yet")

    def _load_multitask(self):
        """Charge le modèle multitâche."""
        multitask_dir = self.models_dir / "multitask"
        if not multitask_dir.exists():
            logger.warning("Multitask models directory not found: %s", multitask_dir)
            return
        # TODO: charger le modèle multitask
        logger.info("Multitask model: not implemented yet")

    def _load_transfer(self):
        """Charge le modèle transfer learning."""
        transfer_dir = self.models_dir / "transfer"
        if not transfer_dir.exists():
            logger.warning("Transfer models directory not found: %s", transfer_dir)
            return
        # TODO: charger le modèle transfer
        logger.info("Transfer model: not implemented yet")

    def predict(self, model_key: str, input_tensor) -> dict:
        """
        Exécute l'inférence sur un modèle chargé.

        Args:
            model_key: Clé du modèle dans self.models.
            input_tensor: Tensor d'entrée préprocessé.

        Returns:
            Dict avec les prédictions.
        """
        if model_key not in self.models:
            raise ValueError(f"Model '{model_key}' not loaded")
        # TODO: implémenter inférence
        # model = self.models[model_key]
        # with torch.no_grad():
        #     output = model(input_tensor)
        # return self._parse_output(output)
        return {}

    @property
    def loaded_count(self) -> int:
        """Nombre de modèles chargés."""
        return len(self.models)
