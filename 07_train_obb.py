#!/usr/bin/env python3
"""
07_train_obb.py — Train the OBB (oriented bounding box) window-rotation model.

Trains YOLOv8n-OBB to detect the meter reading window and output its rotation
angle.  This angle is used by predict_and_read.py to deskew the image before
digit detection.

Run 06_prepare_obb_dataset.py first to build obb_dataset/ and data_obb.yaml.

Usage:
    python3 07_train_obb.py

Custom:
    python3 07_train_obb.py \
        --data     obb_dataset/data_obb.yaml \
        --model    yolov8n-obb.pt \
        --epochs   100 \
        --imgsz    640 \
        --batch    8 \
        --patience 20 \
        --name     window_rotation
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8-OBB model for meter window angle detection."
    )
    parser.add_argument("--model",    default="yolov8n-obb.pt",        help="OBB base weights (downloads automatically)")
    parser.add_argument("--data",     default="obb_dataset/data_obb.yaml", help="Path to OBB data.yaml")
    parser.add_argument("--epochs",   type=int, default=100,           help="Training epochs")
    parser.add_argument("--patience", type=int, default=20,            help="Early-stop patience (0 = off)")
    parser.add_argument("--imgsz",    type=int, default=640,           help="Input image size")
    parser.add_argument("--batch",    type=int, default=8,             help="Batch size")
    parser.add_argument("--project",  default="runs/obb",              help="Output project folder")
    parser.add_argument("--name",     default="window_rotation",       help="Experiment name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"OBB data.yaml not found: {data_yaml}\n"
            "Run 06_prepare_obb_dataset.py first."
        )

    print("=" * 60)
    print("  RWANDA WATER METER — OBB MODEL TRAINING")
    print("=" * 60)
    print(f"  Base model : {args.model}")
    print(f"  Data YAML  : {data_yaml}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Patience   : {args.patience}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Batch size : {args.batch}")
    print(f"  Output     : {args.project}/{args.name}")
    print()
    print("  NOTE: yolov8n-obb.pt will download automatically on first run.")
    print()

    model = YOLO(args.model)

    model.train(
        data     = str(data_yaml),
        epochs   = args.epochs,
        imgsz    = args.imgsz,
        batch    = args.batch,
        patience = args.patience,
        project  = args.project,
        name     = args.name,
        cache    = True,
        workers  = 4,

        # ── Augmentation: windows arrive at ANY angle in the field ───────────
        # Heavy rotation is essential — the whole point of this model is to
        # detect and measure tilt, so it must see tilted examples during training.
        degrees  = 180.0,   # full 360° rotation coverage (±180°)
        scale    = 0.6,
        fliplr   = 0.5,
        flipud   = 0.3,
        hsv_v    = 0.4,
        mosaic   = 0.5,
    )

    weights_path = Path(args.project) / args.name / "weights" / "best.pt"
    print()
    print("=" * 60)
    print("  OBB TRAINING COMPLETE")
    print(f"  Best weights → {weights_path}")
    print()
    print("  Next step:")
    print("    python3 predict_and_read.py \\")
    print(f"        --digit-weights runs/detect/digit_model/weights/best.pt \\")
    print(f"        --obb-weights   {weights_path} \\")
    print("        --source        path/to/image.jpg")
    print("=" * 60)


if __name__ == "__main__":
    main()