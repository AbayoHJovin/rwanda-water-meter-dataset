#!/usr/bin/env python3
"""
03_train.py — Train the digit-detection YOLO model.

Trains YOLOv8-detect from scratch (or from any base weights) using the
prepared digit dataset.  Augmentation settings match the notebook so that
local training and Colab training produce identical results.

Usage:
    python3 03_train.py

Custom run:
    python3 03_train.py \
        --data   dataset/data.yaml \
        --model  yolov8n.pt \
        --epochs 209 \
        --imgsz  960 \
        --batch  8 \
        --patience 30 \
        --name   digit_model
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO digit-detection model for the Rwanda Water Meter system."
    )
    parser.add_argument("--model",    default="yolov8n.pt",           help="Base YOLO weights")
    parser.add_argument("--data",     default="dataset/data.yaml",    help="Path to data.yaml")
    parser.add_argument("--epochs",   type=int,   default=200,        help="Training epochs")
    parser.add_argument("--patience", type=int,   default=20,         help="Early-stop patience (0 = off)")
    parser.add_argument("--imgsz",    type=int,   default=960,        help="Input image size")
    parser.add_argument("--batch",    type=int,   default=8,          help="Batch size")
    parser.add_argument("--project",  default="runs/detect",          help="Output project folder")
    parser.add_argument("--name",     default="digit_model",          help="Experiment name")
    parser.add_argument("--cache",    action="store_true", default=True,
                        help="Cache images in RAM for faster training")
    parser.add_argument("--workers",  type=int,   default=4,          help="DataLoader workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    print("=" * 60)
    print("  RWANDA WATER METER — DIGIT MODEL TRAINING")
    print("=" * 60)
    print(f"  Base model : {args.model}")
    print(f"  Data YAML  : {data_yaml}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Patience   : {args.patience}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Batch size : {args.batch}")
    print(f"  Output     : {args.project}/{args.name}")
    print()

    model = YOLO(args.model)

    model.train(
        data    = str(data_yaml),
        epochs  = args.epochs,
        imgsz   = args.imgsz,
        batch   = args.batch,
        patience= args.patience,
        project = args.project,
        name    = args.name,
        cache   = args.cache,
        workers = args.workers,

        # ── Augmentation tuned for rotated field photos ──────────────────────
        # Meters arrive at all angles; heavy rotation + perspective teaches
        # the model to read digits regardless of how the phone was held.
        degrees     = 45.0,    # random rotation ±45°
        scale       = 0.7,     # zoom range 30%–170%
        perspective = 0.0005,  # mild phone-tilt distortion
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

    weights_path = Path(args.project) / args.name / "weights" / "best.pt"
    print()
    print("=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Best weights → {weights_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()