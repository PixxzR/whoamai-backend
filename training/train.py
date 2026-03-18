"""Script d'entraînement CLI configurable par stratégie."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.multitask import MultitaskModel
from app.models.specialized import AgeModel, EthnicityModel, GenderModel
from app.models.transfer import TransferModel
from training.dataset import UTKFaceDataset, get_train_transforms, get_val_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_data_loaders(data_dir: str, batch_size: int, image_size: int = 224) -> tuple:
    """Charge les DataLoaders train/val."""
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Train/val directories not found in {data_dir}. "
            "Run `python -m scripts.download_utkface` first."
        )

    train_images = sorted(train_dir.glob("*.jpg")) + sorted(train_dir.glob("*.png"))
    val_images = sorted(val_dir.glob("*.jpg")) + sorted(val_dir.glob("*.png"))

    train_dataset = UTKFaceDataset(train_images, transform=get_train_transforms(image_size))
    val_dataset = UTKFaceDataset(val_images, transform=get_val_transforms(image_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader


def train_specialized(train_loader, val_loader, epochs: int, lr: float, device: str, output_dir: Path):
    """Entraîne les 3 modèles spécialisés."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Age Model ---
    logger.info("Training AgeModel...")
    age_model = AgeModel().to(device)
    age_optimizer = torch.optim.Adam(age_model.parameters(), lr=lr)
    age_scheduler = ReduceLROnPlateau(age_optimizer, patience=3, factor=0.5)
    age_criterion = nn.MSELoss()
    best_age_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        age_model.train()
        total_loss = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            ages = batch["age"].to(device)
            age_optimizer.zero_grad()
            preds = age_model(images)
            loss = age_criterion(preds, ages)
            loss.backward()
            age_optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        age_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                ages = batch["age"].to(device)
                preds = age_model(images)
                val_loss += age_criterion(preds, ages).item()
        val_loss /= len(val_loader)
        age_scheduler.step(val_loss)

        logger.info("AgeModel Epoch %d/%d - train_loss: %.4f - val_loss: %.4f", epoch + 1, epochs, avg_loss, val_loss)

        if val_loss < best_age_loss:
            best_age_loss = val_loss
            torch.save(age_model.state_dict(), output_dir / "age_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                logger.info("Early stopping AgeModel at epoch %d", epoch + 1)
                break

    # --- Gender Model ---
    logger.info("Training GenderModel...")
    gender_model = GenderModel().to(device)
    gender_optimizer = torch.optim.Adam(gender_model.parameters(), lr=lr)
    gender_scheduler = ReduceLROnPlateau(gender_optimizer, patience=3, factor=0.5)
    gender_criterion = nn.CrossEntropyLoss()
    best_gender_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        gender_model.train()
        total_loss = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            genders = batch["gender"].to(device)
            gender_optimizer.zero_grad()
            preds = gender_model(images)
            loss = gender_criterion(preds, genders)
            loss.backward()
            gender_optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        gender_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                genders = batch["gender"].to(device)
                preds = gender_model(images)
                correct += (preds.argmax(1) == genders).sum().item()
                total += genders.size(0)
        val_acc = correct / total
        gender_scheduler.step(1 - val_acc)

        logger.info("GenderModel Epoch %d/%d - loss: %.4f - val_acc: %.4f", epoch + 1, epochs, avg_loss, val_acc)

        if val_acc > best_gender_acc:
            best_gender_acc = val_acc
            torch.save(gender_model.state_dict(), output_dir / "gender_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                logger.info("Early stopping GenderModel at epoch %d", epoch + 1)
                break

    # --- Ethnicity Model ---
    logger.info("Training EthnicityModel...")
    eth_model = EthnicityModel().to(device)
    eth_optimizer = torch.optim.Adam(eth_model.parameters(), lr=lr)
    eth_scheduler = ReduceLROnPlateau(eth_optimizer, patience=3, factor=0.5)
    eth_criterion = nn.CrossEntropyLoss()
    best_eth_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        eth_model.train()
        total_loss = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            races = batch["race"].to(device)
            eth_optimizer.zero_grad()
            preds = eth_model(images)
            loss = eth_criterion(preds, races)
            loss.backward()
            eth_optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        eth_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                races = batch["race"].to(device)
                preds = eth_model(images)
                correct += (preds.argmax(1) == races).sum().item()
                total += races.size(0)
        val_acc = correct / total
        eth_scheduler.step(1 - val_acc)

        logger.info("EthnicityModel Epoch %d/%d - loss: %.4f - val_acc: %.4f", epoch + 1, epochs, avg_loss, val_acc)

        if val_acc > best_eth_acc:
            best_eth_acc = val_acc
            torch.save(eth_model.state_dict(), output_dir / "ethnicity_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                logger.info("Early stopping EthnicityModel at epoch %d", epoch + 1)
                break

    logger.info("Specialized training complete.")


def train_multitask(train_loader, val_loader, epochs: int, lr: float, device: str, output_dir: Path):
    """Entraîne le modèle multitâche."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MultitaskModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    age_criterion = nn.MSELoss()
    gender_criterion = nn.CrossEntropyLoss()
    ethnicity_criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            ages = batch["age"].to(device)
            genders = batch["gender"].to(device)
            races = batch["race"].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss_age = age_criterion(outputs["age"], ages)
            loss_gender = gender_criterion(outputs["gender"], genders)
            loss_ethnicity = ethnicity_criterion(outputs["ethnicity"], races)
            loss = loss_age + loss_gender + loss_ethnicity

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                ages = batch["age"].to(device)
                genders = batch["gender"].to(device)
                races = batch["race"].to(device)

                outputs = model(images)
                loss = (
                    age_criterion(outputs["age"], ages)
                    + gender_criterion(outputs["gender"], genders)
                    + ethnicity_criterion(outputs["ethnicity"], races)
                )
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        logger.info("Multitask Epoch %d/%d - train_loss: %.4f - val_loss: %.4f", epoch + 1, epochs, avg_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "multitask_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    logger.info("Multitask training complete.")


def train_transfer(train_loader, val_loader, epochs: int, lr: float, device: str, output_dir: Path):
    """Entraîne le modèle transfer learning."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model = TransferModel().to(device)
    # N'optimiser que les paramètres non gelés
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    age_criterion = nn.MSELoss()
    gender_criterion = nn.CrossEntropyLoss()
    ethnicity_criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            ages = batch["age"].to(device)
            genders = batch["gender"].to(device)
            races = batch["race"].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss_age = age_criterion(outputs["age"], ages)
            loss_gender = gender_criterion(outputs["gender"], genders)
            loss_ethnicity = ethnicity_criterion(outputs["ethnicity"], races)
            loss = loss_age + loss_gender + loss_ethnicity

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                ages = batch["age"].to(device)
                genders = batch["gender"].to(device)
                races = batch["race"].to(device)

                outputs = model(images)
                loss = (
                    age_criterion(outputs["age"], ages)
                    + gender_criterion(outputs["gender"], genders)
                    + ethnicity_criterion(outputs["ethnicity"], races)
                )
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        logger.info("Transfer Epoch %d/%d - train_loss: %.4f - val_loss: %.4f", epoch + 1, epochs, avg_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "transfer_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    logger.info("Transfer training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train face attribute models")
    parser.add_argument("--strategy", choices=["specialized", "multitask", "transfer"], required=True)
    parser.add_argument("--data_dir", default="./data/utkface", help="Path to UTKFace data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda, mps)")
    parser.add_argument("--output_dir", default="./models", help="Base output directory")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.info("Strategy: %s | Device: %s | Epochs: %d | LR: %s", args.strategy, device, args.epochs, args.lr)

    train_loader, val_loader = get_data_loaders(args.data_dir, args.batch_size)
    output_dir = Path(args.output_dir) / args.strategy

    start = time.time()
    if args.strategy == "specialized":
        train_specialized(train_loader, val_loader, args.epochs, args.lr, device, output_dir)
    elif args.strategy == "multitask":
        train_multitask(train_loader, val_loader, args.epochs, args.lr, device, output_dir)
    elif args.strategy == "transfer":
        train_transfer(train_loader, val_loader, args.epochs, args.lr, device, output_dir)

    elapsed = time.time() - start
    logger.info("Training completed in %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
