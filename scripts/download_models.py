"""
Script pour telecharger les modeles PyTorch depuis HuggingFace/Google Drive.
Usage: python scripts/download_models.py
"""

from pathlib import Path

MODELS_DIR = Path("./models")


def download_specialized_models():
    print("Downloading specialized models...")
    specialized_dir = MODELS_DIR / "specialized"
    specialized_dir.mkdir(parents=True, exist_ok=True)
    # TODO: implémenter download depuis HuggingFace ou Google Drive
    # Modèles attendus: age.pth, gender.pth, ethnicity.pth, emotion.pth
    print("  -> specialized/ (not implemented)")


def download_multitask_model():
    print("Downloading multitask model...")
    multitask_dir = MODELS_DIR / "multitask"
    multitask_dir.mkdir(parents=True, exist_ok=True)
    # TODO: implémenter download modèle multitask
    print("  -> multitask/ (not implemented)")


def download_transfer_model():
    print("Downloading transfer learning model...")
    transfer_dir = MODELS_DIR / "transfer"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    # TODO: implémenter download modèle transfer
    print("  -> transfer/ (not implemented)")


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    download_specialized_models()
    download_multitask_model()
    download_transfer_model()
    print("Done. (Models not yet available)")
