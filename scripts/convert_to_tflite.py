"""Conversion PyTorch -> ONNX -> TFLite pour déploiement mobile."""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.multitask import MultitaskModel
from app.models.specialized import AgeModel, EthnicityModel, GenderModel
from app.models.transfer import TransferModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def convert_to_onnx(model: torch.nn.Module, output_path: Path, input_shape=(1, 3, 224, 224)):
    """Convertit un modèle PyTorch en ONNX."""
    model.eval()
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=13,
    )
    logger.info("Exported ONNX: %s", output_path)


def onnx_to_tflite(onnx_path: Path, tflite_path: Path):
    """Convertit un modèle ONNX en TFLite via onnx-tf."""
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)

        # Sauvegarder en SavedModel puis convertir en TFLite
        saved_model_dir = str(tflite_path.parent / "saved_model_temp")
        tf_rep.export_graph(saved_model_dir)

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        tflite_path.write_bytes(tflite_model)
        logger.info("Exported TFLite: %s (%.1f MB)", tflite_path, len(tflite_model) / 1e6)

        # Nettoyer
        import shutil
        shutil.rmtree(saved_model_dir, ignore_errors=True)
    except ImportError:
        logger.error(
            "onnx-tf and tensorflow are required for TFLite conversion.\n"
            "Install: pip install onnx onnx-tf tensorflow"
        )
        raise


def convert_specialized(models_dir: Path, output_dir: Path, device: str = "cpu"):
    """Convertit les 3 modèles spécialisés."""
    model_map = {
        "age_model": AgeModel,
        "gender_model": GenderModel,
        "ethnicity_model": EthnicityModel,
    }

    for name, cls in model_map.items():
        pth_path = models_dir / "specialized" / f"{name}.pth"
        if not pth_path.exists():
            logger.warning("Skipping %s (not found)", pth_path)
            continue

        model = cls()
        model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
        model.eval()

        onnx_path = output_dir / f"{name}.onnx"
        tflite_path = output_dir / f"{name}.tflite"

        convert_to_onnx(model, onnx_path)
        onnx_to_tflite(onnx_path, tflite_path)
        onnx_path.unlink()  # Supprimer l'intermédiaire ONNX


def convert_multitask(models_dir: Path, output_dir: Path, device: str = "cpu"):
    """Convertit le modèle multitâche."""
    pth_path = models_dir / "multitask" / "multitask_model.pth"
    if not pth_path.exists():
        logger.warning("Multitask model not found: %s", pth_path)
        return

    model = MultitaskModel()
    model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))

    onnx_path = output_dir / "multitask_model.onnx"
    tflite_path = output_dir / "multitask_model.tflite"

    convert_to_onnx(model, onnx_path)
    onnx_to_tflite(onnx_path, tflite_path)
    onnx_path.unlink()


def convert_transfer(models_dir: Path, output_dir: Path, device: str = "cpu"):
    """Convertit le modèle transfer learning."""
    pth_path = models_dir / "transfer" / "transfer_model.pth"
    if not pth_path.exists():
        logger.warning("Transfer model not found: %s", pth_path)
        return

    model = TransferModel()
    model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))

    onnx_path = output_dir / "transfer_model.onnx"
    tflite_path = output_dir / "transfer_model.tflite"

    convert_to_onnx(model, onnx_path)
    onnx_to_tflite(onnx_path, tflite_path)
    onnx_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch models to TFLite")
    parser.add_argument("--strategy", choices=["specialized", "multitask", "transfer", "all"], default="all")
    parser.add_argument("--models_dir", default="./models")
    parser.add_argument("--output_dir", default="./models/tflite")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies = ["specialized", "multitask", "transfer"] if args.strategy == "all" else [args.strategy]

    for strategy in strategies:
        logger.info("Converting %s...", strategy)
        if strategy == "specialized":
            convert_specialized(models_dir, output_dir, args.device)
        elif strategy == "multitask":
            convert_multitask(models_dir, output_dir, args.device)
        elif strategy == "transfer":
            convert_transfer(models_dir, output_dir, args.device)

    logger.info("Conversion complete!")


if __name__ == "__main__":
    main()
