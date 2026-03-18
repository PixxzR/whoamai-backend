"""Dataset UTKFace : parse filename [age]_[gender]_[race]_[date].jpg."""

import logging
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

ETHNICITY_MAP = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}


def parse_utkface_filename(filename: str) -> dict | None:
    """Parse un nom de fichier UTKFace : [age]_[gender]_[race]_[date].jpg.

    Returns:
        dict avec age (int), gender (int: 0=Male, 1=Female), race (int: 0-4)
        ou None si le parsing échoue.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    try:
        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])
        if age < 0 or age > 120 or gender not in (0, 1) or race not in range(5):
            return None
        return {"age": age, "gender": gender, "race": race}
    except (ValueError, IndexError):
        return None


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class UTKFaceDataset(Dataset):
    """Dataset UTKFace pour entraînement."""

    def __init__(self, image_paths: list[Path], transform=None):
        self.samples = []
        self.transform = transform

        skipped = 0
        for path in image_paths:
            parsed = parse_utkface_filename(path.name)
            if parsed is None:
                skipped += 1
                continue
            self.samples.append((path, parsed))

        if skipped > 0:
            logger.warning("Skipped %d files with invalid filenames", skipped)
        logger.info("Loaded %d samples", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, labels = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "age": float(labels["age"]),
            "gender": labels["gender"],
            "race": labels["race"],
        }
