"""Télécharger le dataset UTKFace et splitter train/val/test (70/15/15)."""

import argparse
import logging
import os
import random
import shutil
import tarfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

UTKFACE_URL = "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/"


def split_dataset(source_dir: Path, output_dir: Path, train_ratio: float = 0.70, val_ratio: float = 0.15, seed: int = 42):
    """Split les images en train/val/test."""
    random.seed(seed)

    images = sorted(list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png")) + list(source_dir.glob("*.chip.jpg")))
    # Filtrer les fichiers valides (nom parsable)
    valid_images = []
    for img in images:
        parts = img.stem.split("_")
        if len(parts) >= 3:
            try:
                int(parts[0])
                int(parts[1])
                int(parts[2])
                valid_images.append(img)
            except ValueError:
                continue

    random.shuffle(valid_images)
    total = len(valid_images)

    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    splits = {
        "train": valid_images[:train_end],
        "val": valid_images[train_end:val_end],
        "test": valid_images[val_end:],
    }

    for split_name, split_images in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for img_path in split_images:
            shutil.copy2(img_path, split_dir / img_path.name)
        logger.info("%s: %d images", split_name, len(split_images))

    logger.info("Total valid images: %d", total)


def main():
    parser = argparse.ArgumentParser(description="Download and prepare UTKFace dataset")
    parser.add_argument("--source_dir", required=True, help="Directory containing UTKFace images (downloaded manually)")
    parser.add_argument("--output_dir", default="./data/utkface", help="Output directory for train/val/test splits")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.source_dir)
    if not source.exists():
        logger.error(
            "Source directory not found: %s\n"
            "Please download UTKFace manually from:\n"
            "  https://susanqq.github.io/UTKFace/\n"
            "Then run:\n"
            "  python -m scripts.download_utkface --source_dir /path/to/UTKFace",
            source,
        )
        return

    output = Path(args.output_dir)
    logger.info("Splitting UTKFace from %s to %s", source, output)
    split_dataset(source, output, args.train_ratio, args.val_ratio, args.seed)
    logger.info("Done!")


if __name__ == "__main__":
    main()
