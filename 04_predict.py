#!/usr/bin/env python3
"""
04_predict.py — Batch predict on the test set using the latest trained digit model.

Automatically:
  - Finds the latest best.pt in runs/detect/
  - Runs inference on dataset/images/test
  - Saves annotated images and .txt prediction files

Usage:
    python3 04_predict.py

Custom paths:
    python3 04_predict.py \
        --weights runs/detect/digit_model/weights/best.pt \
        --source  dataset/images/test \
        --output  prediction_outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO


# ── Defaults (match notebook layout) ─────────────────────────────────────────
TRAINING_ROOT   = Path("runs/detect")
TEST_IMAGES_DIR = Path("dataset/images/test")
OUTPUT_PROJECT  = Path("prediction_outputs")
OUTPUT_NAME     = "test_predictions"
IMAGE_SIZE      = 960    # matches training imgsz
CONFIDENCE      = 0.25
IOU             = 0.35

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-predict on the test set using the best digit model."
    )
    parser.add_argument("--weights", default=None,
                        help="Path to best.pt (auto-detected from runs/detect/ if omitted)")
    parser.add_argument("--source",  default=str(TEST_IMAGES_DIR),
                        help="Folder of test images")
    parser.add_argument("--output",  default=str(OUTPUT_PROJECT),
                        help="Root folder for prediction output")
    parser.add_argument("--name",    default=OUTPUT_NAME,
                        help="Sub-folder name for this prediction run")
    parser.add_argument("--conf",    type=float, default=CONFIDENCE,
                        help="Confidence threshold")
    parser.add_argument("--iou",     type=float, default=IOU,
                        help="IoU threshold for NMS")
    parser.add_argument("--imgsz",   type=int,   default=IMAGE_SIZE,
                        help="Inference image size")
    return parser.parse_args()


def find_latest_best_model(training_root: Path) -> Path:
    best_models = list(training_root.glob("*/weights/best.pt"))
    if not best_models:
        raise FileNotFoundError(f"No best.pt found inside: {training_root}")
    return max(best_models, key=lambda p: p.stat().st_mtime)


def iter_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS:
            yield path


def main() -> None:
    args = parse_args()

    source_dir = Path(args.source)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    test_images = list(iter_images(source_dir))
    if not test_images:
        raise RuntimeError(f"No supported images found in: {source_dir}")

    model_path = Path(args.weights) if args.weights else find_latest_best_model(TRAINING_ROOT)
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  RWANDA WATER METER — BATCH PREDICTION (TEST SET)")
    print("=" * 60)
    print(f"  Model      : {model_path}")
    print(f"  Source     : {source_dir}")
    print(f"  Images     : {len(test_images)}")
    print(f"  Output     : {output_root / args.name}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Confidence : {args.conf}")
    print(f"  IoU thresh : {args.iou}")
    print()

    model = YOLO(str(model_path))

    results = model.predict(
        source     = str(source_dir),
        imgsz      = args.imgsz,
        conf       = args.conf,
        iou        = args.iou,
        save       = True,
        save_txt   = True,
        save_conf  = True,
        show_labels= True,
        show_conf  = False,
        project    = str(output_root),
        name       = args.name,
        exist_ok   = False,
    )

    output_dir = output_root / args.name
    print()
    print("Prediction complete.")
    print(f"  Annotated images → {output_dir}")
    print(f"  Label .txt files → {output_dir / 'labels'}")
    print()
    print("Per-image summary:")
    for img_path, result in zip(test_images, results):
        n = 0 if result.boxes is None else len(result.boxes)
        print(f"  {img_path.name}: {n} detection(s)")


if __name__ == "__main__":
    main()