"""CLI unifié pour le projet WhoAmAI - Dataset, Training, Evaluation, Test, Serveur."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def cmd_download_dataset(args):
    """Télécharge/prépare le dataset UTKFace."""
    subprocess.run(
        [sys.executable, "-m", "scripts.download_utkface", "--source_dir", args.source_dir, "--output_dir", args.output_dir],
        check=True,
    )


def cmd_train(args):
    """Lance l'entraînement."""
    if args.all:
        for strategy in ["specialized", "multitask", "transfer"]:
            print(f"\n{'='*60}")
            print(f"Training {strategy}...")
            print(f"{'='*60}")
            subprocess.run(
                [
                    sys.executable, "-m", "training.train",
                    "--strategy", strategy,
                    "--epochs", str(args.epochs),
                    "--batch_size", str(args.batch),
                    "--lr", str(args.lr),
                    "--device", args.device,
                    "--data_dir", args.data_dir,
                ],
                check=True,
            )
    else:
        subprocess.run(
            [
                sys.executable, "-m", "training.train",
                "--strategy", args.strategy,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch),
                "--lr", str(args.lr),
                "--device", args.device,
                "--data_dir", args.data_dir,
            ],
            check=True,
        )


def cmd_evaluate(args):
    """Évalue les modèles."""
    strategy = "all" if args.all else args.strategy
    subprocess.run(
        [
            sys.executable, "-m", "training.evaluate",
            "--strategy", strategy,
            "--data_dir", args.data_dir,
            "--models_dir", args.models_dir,
            "--device", args.device,
        ],
        check=True,
    )


def cmd_compare(args):
    """Compare les métriques des 3 stratégies côte à côte."""
    models_dir = Path(args.models_dir)
    strategies = ["specialized", "multitask", "transfer"]
    all_metrics = {}

    for strategy in strategies:
        metrics_file = models_dir / strategy / "metrics.json"
        if metrics_file.exists():
            all_metrics[strategy] = json.loads(metrics_file.read_text())
        else:
            all_metrics[strategy] = {"strategy": strategy, "note": "No metrics found"}

    # Affichage tableau
    print(f"\n{'Strategy':<15} {'Age MAE':<12} {'Gender Acc':<14} {'Ethnicity Acc':<15}")
    print("-" * 56)
    for strategy in strategies:
        m = all_metrics[strategy]
        age_mae = f"{m.get('age_mae', 'N/A')}"
        gender_acc = f"{m.get('gender_accuracy', 'N/A')}"
        eth_acc = f"{m.get('ethnicity_accuracy', 'N/A')}"
        print(f"{strategy:<15} {age_mae:<12} {gender_acc:<14} {eth_acc:<15}")
    print()


def cmd_test_image(args):
    """Teste une image avec les modèles."""
    import io
    from PIL import Image

    # Vérifier que l'image existe
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return

    # Lancer le serveur temporairement et tester
    import requests

    strategies = [args.strategy] if args.strategy else ["specialized", "multitask", "transfer"]
    base_url = f"http://localhost:{args.port}"

    for strategy in strategies:
        print(f"\n--- {strategy.upper()} ---")
        try:
            with open(image_path, "rb") as f:
                response = requests.post(
                    f"{base_url}/predict/{strategy}",
                    files={"file": (image_path.name, f, "image/jpeg")},
                    timeout=30,
                )
            if response.status_code == 200:
                data = response.json()
                pred = data.get("data", {})
                meta = data.get("metadata", {})
                print(f"  Age: {pred.get('age', 'N/A')}")
                print(f"  Gender: {pred.get('gender', 'N/A')} ({pred.get('gender_confidence', 0):.1%})")
                print(f"  Ethnicity: {pred.get('ethnicity', 'N/A')} ({pred.get('ethnicity_confidence', 0):.1%})")
                print(f"  Inference: {meta.get('inference_time_ms', 0):.1f}ms")
                print(f"  Demo mode: {meta.get('demo_mode', False)}")
            else:
                data = response.json()
                error = data.get("error", {})
                print(f"  Error ({response.status_code}): {error.get('code', 'UNKNOWN')} - {error.get('message', '')}")
        except requests.ConnectionError:
            print(f"  Error: Cannot connect to {base_url}. Is the server running?")
            print(f"  Start it with: python cli.py serve")
            break
        except Exception as e:
            print(f"  Error: {e}")


def cmd_convert_tflite(args):
    """Convertit les modèles PyTorch en TFLite (via ONNX)."""
    strategy = "all" if not args.strategy else args.strategy
    subprocess.run(
        [
            sys.executable, "-m", "scripts.convert_to_tflite",
            "--strategy", strategy,
            "--models_dir", args.models_dir,
        ],
        check=True,
    )


def cmd_train_tf(args):
    """Lance l'entraînement TensorFlow/Keras (MobileNetV2)."""
    if args.all:
        strategy = "all"
    else:
        strategy = args.strategy
    subprocess.run(
        [
            sys.executable, "-m", "training.train_tf",
            "--strategy", strategy,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch),
            "--lr", str(args.lr),
            "--data_dir", args.data_dir,
            "--output_dir", args.output_dir,
        ],
        check=True,
    )


def cmd_evaluate_tf(args):
    """Évalue les modèles TensorFlow/Keras."""
    strategy = "all" if args.all else args.strategy
    subprocess.run(
        [
            sys.executable, "-m", "training.evaluate_tf",
            "--strategy", strategy,
            "--data_dir", args.data_dir,
            "--models_dir", args.models_dir,
            "--batch_size", str(args.batch_size),
        ],
        check=True,
    )


def cmd_convert_tflite_direct(args):
    """Convertit les modèles TF/Keras directement en TFLite (sans ONNX)."""
    strategy = "all" if not args.strategy else args.strategy
    subprocess.run(
        [
            sys.executable, "-m", "scripts.convert_tf_to_tflite",
            "--strategy", strategy,
            "--models_dir", args.models_dir,
            "--output_dir", args.output_dir,
        ],
        check=True,
    )


def cmd_serve(args):
    """Lance le serveur API."""
    cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", str(args.port)]
    if args.prod:
        cmd.extend(["--workers", "4"])
    else:
        cmd.append("--reload")
    print(f"Starting server: {' '.join(cmd)}")
    subprocess.run(cmd)


def cmd_status(args):
    """Affiche l'état du projet."""
    models_dir = Path(args.models_dir)
    strategies = ["specialized", "multitask", "transfer"]

    print("\n=== WhoAmAI Status ===\n")

    # Modèles
    print("Models:")
    for strategy in strategies:
        strategy_dir = models_dir / strategy
        pth_files = list(strategy_dir.glob("*.pth")) if strategy_dir.exists() else []
        metrics_file = strategy_dir / "metrics.json"
        status = f"{len(pth_files)} .pth files"
        if metrics_file.exists():
            status += " + metrics.json"
        print(f"  {strategy:<15} {status}")

    # TFLite
    tflite_dir = models_dir / "tflite"
    tflite_files = list(tflite_dir.glob("*.tflite")) if tflite_dir.exists() else []
    print(f"\n  TFLite:         {len(tflite_files)} files")

    # Dataset
    data_dir = Path("./data/utkface")
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if split_dir.exists():
            count = len(list(split_dir.glob("*.jpg")))
            print(f"\n  Dataset {split}:   {count} images")
        else:
            print(f"\n  Dataset {split}:   Not found")

    print()


def main():
    parser = argparse.ArgumentParser(description="WhoAmAI CLI - Unified project management tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # download-dataset
    p = subparsers.add_parser("download-dataset", help="Prepare UTKFace dataset (split train/val/test)")
    p.add_argument("--source_dir", required=True, help="Directory with UTKFace images")
    p.add_argument("--output_dir", default="./data/utkface")
    p.set_defaults(func=cmd_download_dataset)

    # train
    p = subparsers.add_parser("train", help="Train models")
    p.add_argument("--strategy", choices=["specialized", "multitask", "transfer"], default="multitask")
    p.add_argument("--all", action="store_true", help="Train all 3 strategies")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--device", default="cpu")
    p.add_argument("--data_dir", default="./data/utkface")
    p.set_defaults(func=cmd_train)

    # evaluate
    p = subparsers.add_parser("evaluate", help="Evaluate models")
    p.add_argument("--strategy", choices=["specialized", "multitask", "transfer"], default="multitask")
    p.add_argument("--all", action="store_true", help="Evaluate all strategies")
    p.add_argument("--data_dir", default="./data/utkface")
    p.add_argument("--models_dir", default="./models")
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_evaluate)

    # compare
    p = subparsers.add_parser("compare", help="Compare metrics across strategies")
    p.add_argument("--models_dir", default="./models")
    p.set_defaults(func=cmd_compare)

    # test-image
    p = subparsers.add_parser("test-image", help="Test prediction on an image")
    p.add_argument("image", help="Path to image file")
    p.add_argument("--strategy", choices=["specialized", "multitask", "transfer"])
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_test_image)

    # convert-tflite (PyTorch → ONNX → TFLite)
    p = subparsers.add_parser("convert-tflite", help="Convert PyTorch models to TFLite (via ONNX)")
    p.add_argument("--strategy", choices=["specialized", "multitask", "transfer"])
    p.add_argument("--models_dir", default="./models")
    p.set_defaults(func=cmd_convert_tflite)

    # train-tf
    p = subparsers.add_parser("train-tf", help="Train TensorFlow/Keras models (MobileNetV2)")
    p.add_argument("--strategy", choices=["specialized", "multitask", "transfer"], default="multitask")
    p.add_argument("--all", action="store_true", help="Train all 3 strategies")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--data_dir", default="./data/utkface")
    p.add_argument("--output_dir", default="./models/tflite")
    p.set_defaults(func=cmd_train_tf)

    # evaluate-tf
    p = subparsers.add_parser("evaluate-tf", help="Evaluate TensorFlow/Keras models")
    p.add_argument("--strategy", choices=["specialized", "multitask", "transfer"], default="multitask")
    p.add_argument("--all", action="store_true", help="Evaluate all strategies")
    p.add_argument("--data_dir", default="./data/utkface")
    p.add_argument("--models_dir", default="./models/tflite")
    p.add_argument("--batch_size", type=int, default=32)
    p.set_defaults(func=cmd_evaluate_tf)

    # convert-tflite-direct (TF/Keras → TFLite direct)
    p = subparsers.add_parser("convert-tflite-direct", help="Convert TF/Keras models to TFLite (direct)")
    p.add_argument("--strategy", choices=["specialized", "multitask", "transfer"])
    p.add_argument("--models_dir", default="./models/tflite")
    p.add_argument("--output_dir", default="./models/tflite")
    p.set_defaults(func=cmd_convert_tflite_direct)

    # serve
    p = subparsers.add_parser("serve", help="Start the API server")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--prod", action="store_true", help="Production mode with multiple workers")
    p.set_defaults(func=cmd_serve)

    # status
    p = subparsers.add_parser("status", help="Show project status")
    p.add_argument("--models_dir", default="./models")
    p.set_defaults(func=cmd_status)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
