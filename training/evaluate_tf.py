"""Évaluation des modèles TensorFlow/Keras : métriques et sauvegarde metrics.json."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.train_tf import build_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_specialized(test_ds: tf.data.Dataset, models_dir: Path) -> dict:
    """Évalue les 3 modèles spécialisés TF."""
    metrics = {"strategy": "specialized", "framework": "tensorflow"}

    # Age
    age_path = models_dir / "age_mobilenetv2.keras"
    if age_path.exists():
        age_model = tf.keras.models.load_model(str(age_path))
        all_preds, all_targets = [], []
        for images, labels in test_ds:
            preds = age_model.predict(images, verbose=0)
            all_preds.extend(preds.flatten())
            all_targets.extend(labels["age"].numpy())
        metrics["age_mae"] = round(float(mean_absolute_error(all_targets, all_preds)), 2)
        metrics["age_mse"] = round(float(mean_squared_error(all_targets, all_preds)), 2)
        metrics["age_r2"] = round(float(r2_score(all_targets, all_preds)), 4)
    else:
        logger.warning("Age model not found: %s", age_path)

    # Gender
    gender_path = models_dir / "gender_mobilenetv2.keras"
    if gender_path.exists():
        gender_model = tf.keras.models.load_model(str(gender_path))
        all_preds, all_targets = [], []
        for images, labels in test_ds:
            preds = gender_model.predict(images, verbose=0)
            all_preds.extend(np.argmax(preds, axis=1))
            all_targets.extend(labels["gender"].numpy())
        metrics["gender_accuracy"] = round(float(accuracy_score(all_targets, all_preds)), 4)
    else:
        logger.warning("Gender model not found: %s", gender_path)

    # Ethnicity
    eth_path = models_dir / "ethnicity_mobilenetv2.keras"
    if eth_path.exists():
        eth_model = tf.keras.models.load_model(str(eth_path))
        all_preds, all_targets = [], []
        for images, labels in test_ds:
            preds = eth_model.predict(images, verbose=0)
            all_preds.extend(np.argmax(preds, axis=1))
            all_targets.extend(labels["race"].numpy())
        metrics["ethnicity_accuracy"] = round(float(accuracy_score(all_targets, all_preds)), 4)
    else:
        logger.warning("Ethnicity model not found: %s", eth_path)

    return metrics


def evaluate_multitask(test_ds: tf.data.Dataset, models_dir: Path) -> dict:
    """Évalue le modèle multitâche TF."""
    model_path = models_dir / "multitask_mobilenetv2.keras"
    if not model_path.exists():
        logger.warning("Multitask TF model not found: %s", model_path)
        return {"strategy": "multitask", "framework": "tensorflow"}

    model = tf.keras.models.load_model(str(model_path))

    age_preds, age_targets = [], []
    gender_preds, gender_targets = [], []
    eth_preds, eth_targets = [], []

    for images, labels in test_ds:
        outputs = model.predict(images, verbose=0)
        age_preds.extend(outputs["age"].flatten())
        age_targets.extend(labels["age"].numpy())
        gender_preds.extend(np.argmax(outputs["gender"], axis=1))
        gender_targets.extend(labels["gender"].numpy())
        eth_preds.extend(np.argmax(outputs["ethnicity"], axis=1))
        eth_targets.extend(labels["race"].numpy())

    return {
        "strategy": "multitask",
        "framework": "tensorflow",
        "age_mae": round(float(mean_absolute_error(age_targets, age_preds)), 2),
        "age_mse": round(float(mean_squared_error(age_targets, age_preds)), 2),
        "age_r2": round(float(r2_score(age_targets, age_preds)), 4),
        "gender_accuracy": round(float(accuracy_score(gender_targets, gender_preds)), 4),
        "ethnicity_accuracy": round(float(accuracy_score(eth_targets, eth_preds)), 4),
    }


def evaluate_transfer(test_ds: tf.data.Dataset, models_dir: Path) -> dict:
    """Évalue le modèle transfer learning TF."""
    model_path = models_dir / "transfer_mobilenetv2.keras"
    if not model_path.exists():
        logger.warning("Transfer TF model not found: %s", model_path)
        return {"strategy": "transfer", "framework": "tensorflow"}

    model = tf.keras.models.load_model(str(model_path))

    age_preds, age_targets = [], []
    gender_preds, gender_targets = [], []
    eth_preds, eth_targets = [], []

    for images, labels in test_ds:
        outputs = model.predict(images, verbose=0)
        age_preds.extend(outputs["age"].flatten())
        age_targets.extend(labels["age"].numpy())
        gender_preds.extend(np.argmax(outputs["gender"], axis=1))
        gender_targets.extend(labels["gender"].numpy())
        eth_preds.extend(np.argmax(outputs["ethnicity"], axis=1))
        eth_targets.extend(labels["race"].numpy())

    return {
        "strategy": "transfer",
        "framework": "tensorflow",
        "age_mae": round(float(mean_absolute_error(age_targets, age_preds)), 2),
        "age_mse": round(float(mean_squared_error(age_targets, age_preds)), 2),
        "age_r2": round(float(r2_score(age_targets, age_preds)), 4),
        "gender_accuracy": round(float(accuracy_score(gender_targets, gender_preds)), 4),
        "ethnicity_accuracy": round(float(accuracy_score(eth_targets, eth_preds)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate TensorFlow/Keras face attribute models")
    parser.add_argument("--strategy", choices=["specialized", "multitask", "transfer", "all"], default="all")
    parser.add_argument("--data_dir", default="./data/utkface", help="Path to UTKFace data")
    parser.add_argument("--models_dir", default="./models/tflite", help="Base TF models directory")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    test_ds, count = build_dataset(data_dir, "test", args.batch_size)
    logger.info("Test set: %d samples", count)

    strategies = ["specialized", "multitask", "transfer"] if args.strategy == "all" else [args.strategy]
    models_base = Path(args.models_dir)

    for strategy in strategies:
        logger.info("Evaluating TF %s...", strategy)
        models_dir = models_base / strategy

        if strategy == "specialized":
            metrics = evaluate_specialized(test_ds, models_dir)
        elif strategy == "multitask":
            metrics = evaluate_multitask(test_ds, models_dir)
        elif strategy == "transfer":
            metrics = evaluate_transfer(test_ds, models_dir)

        # Sauvegarder metrics.json
        metrics_path = models_dir / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2))
        logger.info("Metrics saved to %s", metrics_path)
        logger.info("Results: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
