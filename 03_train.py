#!/usr/bin/env python3
"""
03_train.py

Train a YOLO model with augmentation tuned for water meters:
  - Rotation up to ±45 degrees  (meters are photographed at all angles)
  - Perspective/shear           (phone held at an angle)
  - Scale / zoom                (close-up vs far-away shots)
  - Mosaic, HSV, flips          (standard, already good defaults)

Install:
    pip install ultralytics

Usage:
    python3 scripts/03_train.py
    python3 scripts/03_train.py --data dataset/data.yaml --epochs 150
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO for water meter detection with rotation/zoom augmentation."
    )
    parser.add_argument("--model",   default="yolov8n.pt",             help="Base YOLO weights")
    parser.add_argument("--data",    default="dataset/data.yaml",       help="Path to data.yaml")
    parser.add_argument("--epochs",  type=int, default=150,             help="Training epochs")
    parser.add_argument("--imgsz",   type=int, default=640,             help="Image size")
    parser.add_argument("--batch",   type=int, default=8,               help="Batch size")
    parser.add_argument("--project", default="training_runs",           help="Output project folder")
    parser.add_argument("--name",    default="water_meter_augmented",   help="Experiment name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    model = YOLO(args.model)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,

        # ── Rotation & orientation ───────────────────────────────────────
        # Meters are photographed at all angles — this is the most important one.
        degrees=45.0,          # random rotation ±45°  (default: 0.0)

        # ── Zoom / scale ─────────────────────────────────────────────────
        # Handles close-up vs far-away shots. 0.7 means the object can
        # appear at 30%-170% of its original scale in the crop.
        scale=0.7,             # (default: 0.5 — raising it helps more)

        # ── Perspective / shear ──────────────────────────────────────────
        # Phone held at an angle creates trapezoid distortion.
        perspective=0.0005,    # subtle perspective warp  (default: 0.0)
        shear=5.0,             # shear ±5°               (default: 0.0)

        # ── Translation ──────────────────────────────────────────────────
        translate=0.15,        # shift up to 15% of image size (default: 0.1)

        # ── Color / lighting ─────────────────────────────────────────────
        # Outdoor meters get very different lighting through the day.
        hsv_h=0.02,            # hue shift       (default: 0.015)
        hsv_s=0.8,             # saturation      (default: 0.7)
        hsv_v=0.5,             # brightness      (default: 0.4)

        # ── Standard augmentations (good defaults, leaving as-is) ────────
        mosaic=1.0,            # mosaic 4-image blend
        flipud=0.1,            # vertical flip 10% of the time
        fliplr=0.5,            # horizontal flip 50%
        mixup=0.1,             # blend two images slightly

        # ── Training quality ─────────────────────────────────────────────
        # Close-up boxes are tiny — keep labels that get small after aug.
        close_mosaic=20,       # disable mosaic last 20 epochs for stability
        patience=30,           # early stopping if no improvement for 30 epochs
        cache=True,            # cache images in RAM for faster training
        workers=4,
    )


if __name__ == "__main__":
    main()