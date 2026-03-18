"""Conversion directe TensorFlow/Keras → TFLite (sans passer par ONNX)."""

import argparse
import logging
import sys
from pathlib import Path

import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def convert_keras_to_tflite(keras_path: Path, tflite_path: Path):
    """Convertit un modèle Keras en TFLite avec quantification dynamique."""
    model = tf.keras.models.load_model(str(keras_path))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantification dynamique
    tflite_model = converter.convert()

    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)
    size_mb = len(tflite_model) / 1e6
    logger.info("Exported TFLite: %s (%.1f MB)", tflite_path, size_mb)


def convert_specialized(models_dir: Path, output_dir: Path):
    """Convertit les 3 modèles spécialisés."""
    model_map = {
        "age_mobilenetv2": "age_mobilenetv2",
        "gender_mobilenetv2": "gender_mobilenetv2",
        "ethnicity_mobilenetv2": "ethnicity_mobilenetv2",
    }

    for name, out_name in model_map.items():
        keras_path = models_dir / "specialized" / f"{name}.keras"
        if not keras_path.exists():
            logger.warning("Skipping %s (not found)", keras_path)
            continue
        tflite_path = output_dir / "specialized" / f"{out_name}.tflite"
        convert_keras_to_tflite(keras_path, tflite_path)


def convert_multitask(models_dir: Path, output_dir: Path):
    """Convertit le modèle multitâche."""
    keras_path = models_dir / "multitask" / "multitask_mobilenetv2.keras"
    if not keras_path.exists():
        logger.warning("Multitask model not found: %s", keras_path)
        return
    # Nom de sortie conforme à AssetConstants Flutter
    tflite_path = output_dir / "multitask" / "multitask_efficientnetb0.tflite"
    convert_keras_to_tflite(keras_path, tflite_path)


def convert_transfer(models_dir: Path, output_dir: Path):
    """Convertit le modèle transfer learning en 3 fichiers TFLite séparés.

    Note: Le modèle transfer a 3 sorties (age, gender, ethnicity).
    On exporte le modèle complet en un seul TFLite.
    Pour Flutter, on crée aussi des alias pour chaque tâche.
    """
    keras_path = models_dir / "transfer" / "transfer_mobilenetv2.keras"
    if not keras_path.exists():
        logger.warning("Transfer model not found: %s", keras_path)
        return

    model = tf.keras.models.load_model(str(keras_path))

    # Exporter le modèle complet (multi-output)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Sauvegarder sous les 3 noms attendus par Flutter (même fichier)
    for name in ["age_transfer", "gender_transfer", "ethnicity_transfer"]:
        tflite_path = output_dir / "transfer" / f"{name}.tflite"
        tflite_path.parent.mkdir(parents=True, exist_ok=True)
        tflite_path.write_bytes(tflite_model)
        size_mb = len(tflite_model) / 1e6
        logger.info("Exported TFLite: %s (%.1f MB)", tflite_path, size_mb)


def main():
    parser = argparse.ArgumentParser(description="Convert TF/Keras models to TFLite (direct, no ONNX)")
    parser.add_argument("--strategy", choices=["specialized", "multitask", "transfer", "all"], default="all")
    parser.add_argument("--models_dir", default="./models/tflite", help="Directory containing .keras models")
    parser.add_argument("--output_dir", default="./models/tflite", help="Output directory for .tflite files")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)

    strategies = ["specialized", "multitask", "transfer"] if args.strategy == "all" else [args.strategy]

    for strategy in strategies:
        logger.info("Converting TF %s → TFLite...", strategy)
        if strategy == "specialized":
            convert_specialized(models_dir, output_dir)
        elif strategy == "multitask":
            convert_multitask(models_dir, output_dir)
        elif strategy == "transfer":
            convert_transfer(models_dir, output_dir)

    logger.info("TFLite conversion complete!")


if __name__ == "__main__":
    main()
