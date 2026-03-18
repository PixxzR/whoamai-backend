"""Évaluation des modèles : calcul des métriques et sauvegarde metrics.json."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.multitask import MultitaskModel
from app.models.specialized import AgeModel, EthnicityModel, GenderModel
from app.models.transfer import TransferModel
from training.dataset import UTKFaceDataset, get_val_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_specialized(test_loader, models_dir: Path, device: str) -> dict:
    """Évalue les 3 modèles spécialisés."""
    metrics = {"strategy": "specialized"}

    # Age
    age_path = models_dir / "age_model.pth"
    if age_path.exists():
        age_model = AgeModel()
        age_model.load_state_dict(torch.load(age_path, map_location=device, weights_only=True))
        age_model.to(device).eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                preds = age_model(images)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch["age"].numpy())

        metrics["age_mae"] = round(float(mean_absolute_error(all_targets, all_preds)), 2)
        metrics["age_mse"] = round(float(mean_squared_error(all_targets, all_preds)), 2)
        metrics["age_r2"] = round(float(r2_score(all_targets, all_preds)), 4)

    # Gender
    gender_path = models_dir / "gender_model.pth"
    if gender_path.exists():
        gender_model = GenderModel()
        gender_model.load_state_dict(torch.load(gender_path, map_location=device, weights_only=True))
        gender_model.to(device).eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                preds = gender_model(images).argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch["gender"].numpy())

        metrics["gender_accuracy"] = round(float(accuracy_score(all_targets, all_preds)), 4)

    # Ethnicity
    eth_path = models_dir / "ethnicity_model.pth"
    if eth_path.exists():
        eth_model = EthnicityModel()
        eth_model.load_state_dict(torch.load(eth_path, map_location=device, weights_only=True))
        eth_model.to(device).eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                preds = eth_model(images).argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch["race"].numpy())

        metrics["ethnicity_accuracy"] = round(float(accuracy_score(all_targets, all_preds)), 4)

    return metrics


def evaluate_multitask(test_loader, models_dir: Path, device: str) -> dict:
    """Évalue le modèle multitâche."""
    model_path = models_dir / "multitask_model.pth"
    if not model_path.exists():
        logger.warning("Multitask model not found at %s", model_path)
        return {"strategy": "multitask"}

    model = MultitaskModel()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    age_preds, age_targets = [], []
    gender_preds, gender_targets = [], []
    eth_preds, eth_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            outputs = model(images)

            age_preds.extend(outputs["age"].cpu().numpy())
            age_targets.extend(batch["age"].numpy())

            gender_preds.extend(outputs["gender"].argmax(1).cpu().numpy())
            gender_targets.extend(batch["gender"].numpy())

            eth_preds.extend(outputs["ethnicity"].argmax(1).cpu().numpy())
            eth_targets.extend(batch["race"].numpy())

    return {
        "strategy": "multitask",
        "age_mae": round(float(mean_absolute_error(age_targets, age_preds)), 2),
        "age_mse": round(float(mean_squared_error(age_targets, age_preds)), 2),
        "age_r2": round(float(r2_score(age_targets, age_preds)), 4),
        "gender_accuracy": round(float(accuracy_score(gender_targets, gender_preds)), 4),
        "ethnicity_accuracy": round(float(accuracy_score(eth_targets, eth_preds)), 4),
    }


def evaluate_transfer(test_loader, models_dir: Path, device: str) -> dict:
    """Évalue le modèle transfer learning."""
    model_path = models_dir / "transfer_model.pth"
    if not model_path.exists():
        logger.warning("Transfer model not found at %s", model_path)
        return {"strategy": "transfer"}

    model = TransferModel()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    age_preds, age_targets = [], []
    gender_preds, gender_targets = [], []
    eth_preds, eth_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            outputs = model(images)

            age_preds.extend(outputs["age"].cpu().numpy())
            age_targets.extend(batch["age"].numpy())

            gender_preds.extend(outputs["gender"].argmax(1).cpu().numpy())
            gender_targets.extend(batch["gender"].numpy())

            eth_preds.extend(outputs["ethnicity"].argmax(1).cpu().numpy())
            eth_targets.extend(batch["race"].numpy())

    return {
        "strategy": "transfer",
        "age_mae": round(float(mean_absolute_error(age_targets, age_preds)), 2),
        "age_mse": round(float(mean_squared_error(age_targets, age_preds)), 2),
        "age_r2": round(float(r2_score(age_targets, age_preds)), 4),
        "gender_accuracy": round(float(accuracy_score(gender_targets, gender_preds)), 4),
        "ethnicity_accuracy": round(float(accuracy_score(eth_targets, eth_preds)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate face attribute models")
    parser.add_argument("--strategy", choices=["specialized", "multitask", "transfer", "all"], default="all")
    parser.add_argument("--data_dir", default="./data/utkface", help="Path to UTKFace data")
    parser.add_argument("--models_dir", default="./models", help="Base models directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Charger le test set
    test_dir = Path(args.data_dir) / "test"
    if not test_dir.exists():
        logger.error("Test directory not found: %s", test_dir)
        return

    test_images = sorted(test_dir.glob("*.jpg")) + sorted(test_dir.glob("*.png"))
    test_dataset = UTKFaceDataset(test_images, transform=get_val_transforms())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    strategies = ["specialized", "multitask", "transfer"] if args.strategy == "all" else [args.strategy]
    models_base = Path(args.models_dir)

    for strategy in strategies:
        logger.info("Evaluating %s...", strategy)
        models_dir = models_base / strategy

        if strategy == "specialized":
            metrics = evaluate_specialized(test_loader, models_dir, args.device)
        elif strategy == "multitask":
            metrics = evaluate_multitask(test_loader, models_dir, args.device)
        elif strategy == "transfer":
            metrics = evaluate_transfer(test_loader, models_dir, args.device)

        # Sauvegarder metrics.json
        metrics_path = models_dir / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2))
        logger.info("Metrics saved to %s", metrics_path)
        logger.info("Results: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
