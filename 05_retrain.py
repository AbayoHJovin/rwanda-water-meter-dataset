#!/usr/bin/env python3
"""
05_retrain.py — Continue training from the latest best.pt checkpoint.

Finds the most recently modified best.pt under runs/detect/ and starts a
new training run from it.  Use this after an initial training run when you
want to fine-tune with more epochs or different settings.

Usage:
    python3 05_retrain.py

Custom:
    python3 05_retrain.py \
        --weights runs/detect/digit_model/weights/best.pt \
        --data    dataset/data.yaml \
        --epochs  20 \
        --imgsz   960 \
        --batch   8
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


# ── Defaults (match notebook layout) ─────────────────────────────────────────
TRAINING_ROOT = Path("runs/detect")
DATA_YAML     = Path("dataset/data.yaml")

DEFAULT_EPOCHS     = 20
DEFAULT_IMAGE_SIZE = 960
DEFAULT_BATCH_SIZE = 8
PROJECT_NAME       = "runs/detect"
RUN_PREFIX         = "digit_model_retrain"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain the digit model from the latest best.pt checkpoint."
    )
    parser.add_argument("--weights", default=None,
                        help="Path to best.pt (auto-detected from runs/detect/ if omitted)")
    parser.add_argument("--data",    default=str(DATA_YAML),
                        help="Path to data.yaml")
    parser.add_argument("--epochs",  type=int, default=DEFAULT_EPOCHS,
                        help="Additional training epochs")
    parser.add_argument("--imgsz",   type=int, default=DEFAULT_IMAGE_SIZE,
                        help="Input image size")
    parser.add_argument("--batch",   type=int, default=DEFAULT_BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--patience",type=int, default=30,
                        help="Early-stop patience (0 = off)")
    return parser.parse_args()


def find_latest_best_model(training_root: Path) -> Path:
    best_models = list(training_root.glob("*/weights/best.pt"))
    if not best_models:
        raise FileNotFoundError(f"No best.pt found inside: {training_root}")
    return max(best_models, key=lambda p: p.stat().st_mtime)


def make_run_name(prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def main() -> None:
    args = parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    model_path = Path(args.weights) if args.weights else find_latest_best_model(TRAINING_ROOT)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    run_name = make_run_name(RUN_PREFIX)

    print("=" * 60)
    print("  RWANDA WATER METER — DIGIT MODEL RETRAINING")
    print("=" * 60)
    print(f"  Checkpoint : {model_path}")
    print(f"  Data YAML  : {data_yaml}")
    print(f"  New run    : {PROJECT_NAME}/{run_name}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Batch size : {args.batch}")
    print(f"  Patience   : {args.patience}")
    print()

    model = YOLO(str(model_path))

    model.train(
        data     = str(data_yaml),
        epochs   = args.epochs,
        imgsz    = args.imgsz,
        batch    = args.batch,
        patience = args.patience,
        project  = PROJECT_NAME,
        name     = run_name,
        cache    = True,
        workers  = 4,

        # Keep the same augmentation as initial training
        degrees     = 45.0,
        scale       = 0.7,
        perspective = 0.0005,
        shear       = 5.0,
        translate   = 0.15,
        hsv_h       = 0.02,
        hsv_s       = 0.8,
        hsv_v       = 0.5,
        mosaic      = 1.0,
        flipud      = 0.1,
        fliplr      = 0.5,
        mixup       = 0.1,
        close_mosaic= 20,
    )

    weights_path = Path(PROJECT_NAME) / run_name / "weights" / "best.pt"
    print()
    print("=" * 60)
    print("  RETRAINING COMPLETE")
    print(f"  New best weights → {weights_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()