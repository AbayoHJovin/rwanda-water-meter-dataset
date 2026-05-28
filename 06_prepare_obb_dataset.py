#!/usr/bin/env python3
"""
06_prepare_obb_dataset.py — Prepare the OBB (oriented bounding box) dataset.

Converts a raw folder of OBB-labeled images into a clean train/val split
ready for YOLOv8-OBB training.  This script corresponds to Section 4 of the
Colab notebook and should be run before 07_train_obb.py.

Expected input layout:
    <obb_raw_dir>/
        obb_images/    ← meter photos (labeled with label_window.py)
        labels_obb/    ← .txt OBB label files written by label_window.py

Output layout:
    <obb_dataset>/
        images/
            train/
            val/
        labels/
            train/
            val/
    data_obb.yaml

Usage:
    python3 06_prepare_obb_dataset.py /path/to/Water-OBB-Dataset obb_dataset

Custom split:
    python3 06_prepare_obb_dataset.py /path/to/Water-OBB-Dataset obb_dataset \
        --val 0.15 --seed 0
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# OBB class names — must match label_window.py
OBB_CLASS_NAMES = {
    0: "meter",
    1: "window",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare OBB dataset (train/val split) for window-rotation model."
    )
    parser.add_argument("obb_raw_dir", type=Path,
                        help="Raw OBB dataset root (contains obb_images/ and labels_obb/)")
    parser.add_argument("obb_dataset", type=Path,
                        help="Output dataset directory")
    parser.add_argument("--val",  type=float, default=0.20,
                        help="Fraction of data used for validation (default: 0.20)")
    parser.add_argument("--seed", type=int,   default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def collect_pairs(img_dir: Path, lbl_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in VALID_IMAGE_EXTS:
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            print(f"  WARNING: no label for {img.name} — skipped")
    return pairs


def copy_split(
    pairs: list[tuple[Path, Path]],
    out_dir: Path,
    val_fraction: float,
    seed: int,
) -> dict[str, int]:
    random.seed(seed)
    random.shuffle(pairs)

    n_val   = max(1, int(len(pairs) * val_fraction))
    val_set = pairs[:n_val]
    trn_set = pairs[n_val:]

    for split, split_pairs in [("train", trn_set), ("val", val_set)]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, lbl in split_pairs:
            shutil.copy2(img, out_dir / "images" / split / img.name)
            shutil.copy2(lbl, out_dir / "labels" / split / lbl.name)

    return {"train": len(trn_set), "val": len(val_set), "total": len(pairs)}


def write_yaml(out_dir: Path, dataset_path: Path) -> Path:
    names_block = "\n".join(f"  {k}: {v}" for k, v in OBB_CLASS_NAMES.items())
    yaml_text = (
        f"path: {dataset_path.resolve()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "\n"
        "names:\n"
        f"{names_block}\n"
    )
    yaml_path = out_dir / "data_obb.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def main() -> None:
    args = parse_args()

    img_dir = args.obb_raw_dir / "obb_images"
    lbl_dir = args.obb_raw_dir / "labels_obb"

    if not img_dir.exists():
        raise FileNotFoundError(
            f"obb_images/ not found inside: {args.obb_raw_dir}\n"
            "Expected layout:\n"
            "  <obb_raw_dir>/\n"
            "      obb_images/\n"
            "      labels_obb/"
        )
    if not lbl_dir.exists():
        raise FileNotFoundError(
            f"labels_obb/ not found inside: {args.obb_raw_dir}\n"
            "Run label_window.py first to generate OBB labels."
        )

    print("=" * 60)
    print("  RWANDA WATER METER — OBB DATASET PREPARATION")
    print("=" * 60)
    print(f"  Raw images : {img_dir}")
    print(f"  Raw labels : {lbl_dir}")
    print(f"  Output dir : {args.obb_dataset}")
    print(f"  Val split  : {args.val:.0%}")
    print()

    pairs = collect_pairs(img_dir, lbl_dir)

    if not pairs:
        print("ERROR: No valid image+label pairs found.")
        print("  Check that labels_obb/ contains .txt files matching images in obb_images/")
        return

    print(f"Valid pairs found: {len(pairs)}")

    if args.obb_dataset.exists():
        shutil.rmtree(args.obb_dataset)

    counts = copy_split(pairs, args.obb_dataset, args.val, args.seed)
    yaml_path = write_yaml(args.obb_dataset, args.obb_dataset)

    print()
    print("OBB dataset prepared:")
    print(f"  Train  : {counts['train']} images")
    print(f"  Val    : {counts['val']} images")
    print(f"  Total  : {counts['total']} pairs")
    print(f"  YAML   : {yaml_path}")
    print()
    print("Next step:")
    print("  python3 07_train_obb.py")


if __name__ == "__main__":
    main()