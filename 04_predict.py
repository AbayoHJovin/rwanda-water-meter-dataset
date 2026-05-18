# #!/usr/bin/env python3
# """
# 04_predict.py

# Run YOLO inference on test images and save visualized predictions.

# Fully automatic:
# - Finds latest best.pt from training_runs/
# - Uses dataset/images/test
# - Saves annotated images
# - Saves prediction .txt files
# - Saves confidence scores

# Usage:
#     python3 scripts/04_predict.py
# """

# from __future__ import annotations

# from pathlib import Path
# from typing import Iterable

# from ultralytics import YOLO


# TRAINING_ROOT = Path("training_runs")
# TEST_IMAGES_DIR = Path("dataset/images/test")
# OUTPUT_PROJECT = Path("prediction_outputs")
# OUTPUT_NAME = "test_predictions"

# IMAGE_SIZE = 640
# CONFIDENCE = 0.25

# SAVE_TXT = True
# SAVE_CONF = True

# VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# def find_latest_best_model(training_root: Path) -> Path:
#     best_models = list(training_root.glob("*/weights/best.pt"))

#     if not best_models:
#         raise FileNotFoundError(
#             f"No best.pt model found inside: {training_root}"
#         )

#     return max(best_models, key=lambda p: p.stat().st_mtime)


# def iter_test_images(folder: Path) -> Iterable[Path]:
#     for path in sorted(folder.iterdir()):
#         if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS:
#             yield path


# def main() -> None:
#     model_path = find_latest_best_model(TRAINING_ROOT)

#     if not TEST_IMAGES_DIR.exists():
#         raise FileNotFoundError(
#             f"Test images directory not found: {TEST_IMAGES_DIR}"
#         )

#     test_images = list(iter_test_images(TEST_IMAGES_DIR))

#     if not test_images:
#         raise RuntimeError(f"No supported test images found in: {TEST_IMAGES_DIR}")

#     OUTPUT_PROJECT.mkdir(parents=True, exist_ok=True)

#     print("========== YOLO TEST PREDICTION ==========")
#     print(f"Model:       {model_path}")
#     print(f"Test folder: {TEST_IMAGES_DIR}")
#     print(f"Images:      {len(test_images)}")
#     print(f"Output root: {OUTPUT_PROJECT}")
#     print(f"Run name:    {OUTPUT_NAME}")
#     print(f"Image size:  {IMAGE_SIZE}")
#     print(f"Confidence:  {CONFIDENCE}")
#     print()

#     model = YOLO(str(model_path))

#     results = model.predict(
#         source=str(TEST_IMAGES_DIR),
#         imgsz=IMAGE_SIZE,
#         conf=CONFIDENCE,
#         save=True,
#         save_txt=SAVE_TXT,
#         save_conf=SAVE_CONF,
#         show_labels=True,
#         show_conf=False,
#         project=str(OUTPUT_PROJECT),
#         name=OUTPUT_NAME,
#         exist_ok=False,
#     )

#     output_dir = OUTPUT_PROJECT / OUTPUT_NAME
#     label_dir = output_dir / "labels"

#     print("\nPrediction completed. [√]")
#     print(f"Annotated images saved in: {output_dir}")
#     print(f"Prediction txt files saved in: {label_dir}")

#     print("\nPer-image summary:")
#     for image_path, result in zip(test_images, results):
#         num_boxes = 0 if result.boxes is None else len(result.boxes)
#         print(f"  - {image_path.name}: {num_boxes} detection(s)")


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
"""
04_predict.py

Run YOLO inference on test images with Test-Time Augmentation (TTA).

TTA = the model sees each image multiple times at different scales and flips,
then merges all the boxes. It costs ~3x inference time but improves accuracy,
especially on rotated or zoomed images.

Usage:
    python3 scripts/04_predict.py              # TTA on (default)
    python3 scripts/04_predict.py --no-tta     # faster, no TTA
    python3 scripts/04_predict.py --scales 0.8 1.0 1.2 1.5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO


TRAINING_ROOT   = Path("training_runs")
TEST_IMAGES_DIR = Path("dataset/images/test")
OUTPUT_PROJECT  = Path("prediction_outputs")
OUTPUT_NAME     = "test_predictions"

IMAGE_SIZE  = 640
CONFIDENCE  = 0.25

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO inference with optional TTA")
    parser.add_argument("--weights",  type=str, default=None,
                        help="Path to best.pt (auto-detected if not given)")
    parser.add_argument("--source",   type=str, default=str(TEST_IMAGES_DIR),
                        help="Image folder or single image path")
    parser.add_argument("--conf",     type=float, default=CONFIDENCE)
    parser.add_argument("--imgsz",    type=int,   default=IMAGE_SIZE)
    parser.add_argument("--no-tta",   action="store_true",
                        help="Disable Test-Time Augmentation (faster but less accurate)")
    parser.add_argument("--scales",   type=float, nargs="+",
                        default=[0.83, 1.0, 1.25],
                        help="TTA scale factors (ignored if --no-tta)")
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

    model_path = Path(args.weights) if args.weights else find_latest_best_model(TRAINING_ROOT)
    source     = Path(args.source)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    use_tta = not args.no_tta

    print("========== YOLO PREDICTION ==========")
    print(f"Model:   {model_path}")
    print(f"Source:  {source}")
    print(f"TTA:     {'ON  (scales=' + str(args.scales) + ')' if use_tta else 'OFF'}")
    print(f"Conf:    {args.conf}")
    print()

    model = YOLO(str(model_path))

    predict_kwargs = dict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        save=True,
        save_txt=True,
        save_conf=True,
        show_labels=True,
        show_conf=False,
        project=str(OUTPUT_PROJECT),
        name=OUTPUT_NAME,
        exist_ok=True,
    )

    if use_tta:
        # augment=True enables Ultralytics built-in TTA:
        # runs inference at multiple scales + flips, merges via WBF.
        predict_kwargs["augment"] = True

        # Pass custom scale list if Ultralytics version supports it.
        # (Supported in ultralytics >= 8.1. Silently ignored on older versions.)
        try:
            predict_kwargs["augment_scales"] = args.scales
        except Exception:
            pass

    results = model.predict(**predict_kwargs)

    output_dir = Path(OUTPUT_PROJECT) / OUTPUT_NAME
    print(f"\nDone. Annotated images → {output_dir}")
    print(f"Label txt files       → {output_dir / 'labels'}")

    print("\nPer-image summary:")
    image_list = list(iter_images(source)) if source.is_dir() else [source]
    for img_path, result in zip(image_list, results):
        n = 0 if result.boxes is None else len(result.boxes)
        print(f"  {img_path.name}: {n} detection(s)")


if __name__ == "__main__":
    main()