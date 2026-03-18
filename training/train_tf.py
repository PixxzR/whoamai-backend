"""Script d'entraînement TensorFlow/Keras pour modèles MobileNetV2 (TFLite mobile)."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.dataset import parse_utkface_filename
from training.tf_models import (
    create_age_model,
    create_ethnicity_model,
    create_gender_model,
    create_multitask_model,
    create_transfer_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def _parse_image_file(file_path: str) -> tuple | None:
    """Parse un fichier image UTKFace et retourne (image_bytes, age, gender, race)."""
    filename = tf.strings.split(file_path, sep="/")[-1]
    # On ne peut pas utiliser parse_utkface_filename directement dans tf.data,
    # donc on utilise tf.py_function
    return file_path


def _load_and_preprocess(file_path: str, augment: bool = False):
    """Charge et prétraite une image UTKFace."""
    raw = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0

    # Normalisation ImageNet
    mean = tf.constant(IMAGENET_MEAN, shape=[1, 1, 3])
    std = tf.constant(IMAGENET_STD, shape=[1, 1, 3])
    image = (image - mean) / std

    return image


def _augment_image(image: tf.Tensor) -> tf.Tensor:
    """Applique les augmentations d'entraînement."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.9, 1.1)
    # Rotation légère via crop/pad
    return image


def build_dataset(
    data_dir: Path,
    split: str,
    batch_size: int,
    augment: bool = False,
) -> tuple[tf.data.Dataset, int]:
    """Construit un tf.data.Dataset à partir d'un dossier UTKFace split."""
    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Directory not found: {split_dir}")

    image_paths = sorted(str(p) for p in split_dir.glob("*.jpg"))
    image_paths += sorted(str(p) for p in split_dir.glob("*.png"))

    # Filtrer et parser les labels depuis les noms de fichiers
    valid_paths = []
    ages = []
    genders = []
    races = []

    for path in image_paths:
        filename = Path(path).name
        parsed = parse_utkface_filename(filename)
        if parsed is None:
            continue
        valid_paths.append(path)
        ages.append(float(parsed["age"]))
        genders.append(parsed["gender"])
        races.append(parsed["race"])

    logger.info("Loaded %d valid samples from %s/%s", len(valid_paths), data_dir, split)

    # Créer le dataset
    path_ds = tf.data.Dataset.from_tensor_slices(valid_paths)
    age_ds = tf.data.Dataset.from_tensor_slices(np.array(ages, dtype=np.float32))
    gender_ds = tf.data.Dataset.from_tensor_slices(np.array(genders, dtype=np.int32))
    race_ds = tf.data.Dataset.from_tensor_slices(np.array(races, dtype=np.int32))

    def load_image(path, age, gender, race):
        image = _load_and_preprocess(path)
        return image, {"age": age, "gender": gender, "race": race}

    ds = tf.data.Dataset.zip((path_ds, age_ds, gender_ds, race_ds))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        def apply_augment(image, labels):
            return _augment_image(image), labels
        ds = ds.map(apply_augment, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.shuffle(buffer_size=min(len(valid_paths), 10000))

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, len(valid_paths)


# =============================================================================
# SPECIALIZED TRAINING
# =============================================================================

def train_specialized(data_dir: Path, epochs: int, batch_size: int, lr: float, output_dir: Path):
    """Entraîne les 3 modèles spécialisés MobileNetV2."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, _ = build_dataset(data_dir, "train", batch_size, augment=True)
    val_ds, _ = build_dataset(data_dir, "val", batch_size)

    # Reformater pour chaque tâche
    def extract_age(image, labels):
        return image, labels["age"]

    def extract_gender(image, labels):
        return image, labels["gender"]

    def extract_race(image, labels):
        return image, labels["race"]

    callbacks_base = lambda name: [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    ]

    # --- Age ---
    logger.info("Training AgeModel (MobileNetV2)...")
    age_model = create_age_model()
    age_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    age_model.fit(
        train_ds.map(extract_age),
        validation_data=val_ds.map(extract_age),
        epochs=epochs,
        callbacks=callbacks_base("age"),
    )
    age_model.save(str(output_dir / "age_mobilenetv2.keras"))
    logger.info("AgeModel saved.")

    # --- Gender ---
    logger.info("Training GenderModel (MobileNetV2)...")
    gender_model = create_gender_model()
    gender_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    gender_model.fit(
        train_ds.map(extract_gender),
        validation_data=val_ds.map(extract_gender),
        epochs=epochs,
        callbacks=callbacks_base("gender"),
    )
    gender_model.save(str(output_dir / "gender_mobilenetv2.keras"))
    logger.info("GenderModel saved.")

    # --- Ethnicity ---
    logger.info("Training EthnicityModel (MobileNetV2)...")
    eth_model = create_ethnicity_model()
    eth_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    eth_model.fit(
        train_ds.map(extract_race),
        validation_data=val_ds.map(extract_race),
        epochs=epochs,
        callbacks=callbacks_base("ethnicity"),
    )
    eth_model.save(str(output_dir / "ethnicity_mobilenetv2.keras"))
    logger.info("EthnicityModel saved.")

    logger.info("Specialized TF training complete.")


# =============================================================================
# MULTITASK TRAINING
# =============================================================================

def train_multitask(data_dir: Path, epochs: int, batch_size: int, lr: float, output_dir: Path):
    """Entraîne le modèle multitâche MobileNetV2."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, _ = build_dataset(data_dir, "train", batch_size, augment=True)
    val_ds, _ = build_dataset(data_dir, "val", batch_size)

    def reformat_multitask(image, labels):
        return image, {
            "age": labels["age"],
            "gender": labels["gender"],
            "ethnicity": labels["race"],
        }

    model = create_multitask_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={
            "age": "mse",
            "gender": "sparse_categorical_crossentropy",
            "ethnicity": "sparse_categorical_crossentropy",
        },
        metrics={
            "age": ["mae"],
            "gender": ["accuracy"],
            "ethnicity": ["accuracy"],
        },
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    ]

    model.fit(
        train_ds.map(reformat_multitask),
        validation_data=val_ds.map(reformat_multitask),
        epochs=epochs,
        callbacks=callbacks,
    )

    model.save(str(output_dir / "multitask_mobilenetv2.keras"))
    logger.info("Multitask TF training complete.")


# =============================================================================
# TRANSFER TRAINING
# =============================================================================

def train_transfer(data_dir: Path, epochs: int, batch_size: int, lr: float, output_dir: Path):
    """Entraîne le modèle transfer learning MobileNetV2 (couches early gelées)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, _ = build_dataset(data_dir, "train", batch_size, augment=True)
    val_ds, _ = build_dataset(data_dir, "val", batch_size)

    def reformat_multitask(image, labels):
        return image, {
            "age": labels["age"],
            "gender": labels["gender"],
            "ethnicity": labels["race"],
        }

    model = create_transfer_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={
            "age": "mse",
            "gender": "sparse_categorical_crossentropy",
            "ethnicity": "sparse_categorical_crossentropy",
        },
        metrics={
            "age": ["mae"],
            "gender": ["accuracy"],
            "ethnicity": ["accuracy"],
        },
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    ]

    model.fit(
        train_ds.map(reformat_multitask),
        validation_data=val_ds.map(reformat_multitask),
        epochs=epochs,
        callbacks=callbacks,
    )

    model.save(str(output_dir / "transfer_mobilenetv2.keras"))
    logger.info("Transfer TF training complete.")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train TensorFlow/Keras face attribute models (MobileNetV2)")
    parser.add_argument("--strategy", choices=["specialized", "multitask", "transfer", "all"], default="all")
    parser.add_argument("--data_dir", default="./data/utkface", help="Path to UTKFace data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output_dir", default="./models/tflite", help="Base output directory for TF models")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("Data directory not found: %s. Run `python cli.py download-dataset` first.", data_dir)
        sys.exit(1)

    strategies = ["specialized", "multitask", "transfer"] if args.strategy == "all" else [args.strategy]

    for strategy in strategies:
        output_dir = Path(args.output_dir) / strategy
        logger.info("=" * 60)
        logger.info("Training TF %s (epochs=%d, lr=%s)", strategy, args.epochs, args.lr)
        logger.info("=" * 60)

        start = time.time()
        if strategy == "specialized":
            train_specialized(data_dir, args.epochs, args.batch_size, args.lr, output_dir)
        elif strategy == "multitask":
            train_multitask(data_dir, args.epochs, args.batch_size, args.lr, output_dir)
        elif strategy == "transfer":
            train_transfer(data_dir, args.epochs, args.batch_size, args.lr, output_dir)

        elapsed = time.time() - start
        logger.info("TF %s training completed in %.1f seconds", strategy, elapsed)


if __name__ == "__main__":
    main()
