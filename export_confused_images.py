#!/usr/bin/env python3
"""
export_confused_images.py

Export images where the model made mistakes (confused/misclassified).

Usage:
  python export_confused_images.py --weights best.pt --data data.yaml --output confused_images/

Output structure:
  confused_images/
    ├── false_negatives/
    ├── false_positives/
    ├── misclassified/
    └── summary.txt
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import yaml


CLASS_NAMES = [
    "meter", "window",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "unknown",
]

# Colors — BGR
COLOR_GT_MATCHED    = (34,  197,  94)   # green       — GT detected, class correct
COLOR_GT_WRONG_CLS  = (0,   165, 255)   # orange      — GT detected, class wrong
COLOR_GT_MISSED     = (60,  146, 251)   # blue-orange — GT missed entirely
COLOR_PRED_CORRECT  = (21,  204, 250)   # cyan/yellow — prediction, correct class
COLOR_PRED_WRONG    = (68,   68, 239)   # red         — prediction, wrong class
COLOR_FP            = (0,     0, 200)   # dark red    — false positive


def iou(box1, box2):
    x1 = max(box1[0], box2[0]);  y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]);  y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def parse_label_file(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            x1 = (xc - w/2) * img_w;  y1 = (yc - h/2) * img_h
            x2 = (xc + w/2) * img_w;  y2 = (yc + h/2) * img_h
            boxes.append({'class': cls, 'box': [x1, y1, x2, y2]})
    return boxes


def match_boxes(gt_boxes, pred_boxes, iou_thresh):
    """
    Greedy bipartite matching by descending IoU.
    Returns:
        gt_to_pred   : dict  gt_idx  -> pred_idx
        pred_to_gt   : dict  pred_idx -> gt_idx
        unmatched_gt : list of unmatched gt indices
        unmatched_pred: list of unmatched pred indices
    """
    pairs = []
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            v = iou(gt['box'], pred['box'])
            if v >= iou_thresh:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)

    gt_to_pred = {}
    pred_to_gt = {}
    for _, i, j in pairs:
        if i in gt_to_pred or j in pred_to_gt:
            continue
        gt_to_pred[i] = j
        pred_to_gt[j] = i

    unmatched_gt   = [i for i in range(len(gt_boxes))   if i not in gt_to_pred]
    unmatched_pred = [j for j in range(len(pred_boxes)) if j not in pred_to_gt]
    return gt_to_pred, pred_to_gt, unmatched_gt, unmatched_pred


def make_annotated_image(image, gt_boxes, pred_boxes, gt_to_pred, pred_to_gt,
                         unmatched_gt, unmatched_pred):
    img = image.copy()
    img_h, img_w = img.shape[:2]

    def draw_box(canvas, box, color, thickness=2):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

    # ── Draw GT boxes ────────────────────────────────────────────────────
    for i, gt in enumerate(gt_boxes):
        if i in gt_to_pred:
            correct = gt['class'] == pred_boxes[gt_to_pred[i]]['class']
            color = COLOR_GT_MATCHED if correct else COLOR_GT_WRONG_CLS
        else:
            color = COLOR_GT_MISSED
        draw_box(img, gt['box'], color, thickness=2)

    # ── Draw Pred boxes ──────────────────────────────────────────────────
    for j, pred in enumerate(pred_boxes):
        if j in pred_to_gt:
            correct = pred['class'] == gt_boxes[pred_to_gt[j]]['class']
            color = COLOR_PRED_CORRECT if correct else COLOR_PRED_WRONG
        else:
            color = COLOR_FP
        draw_box(img, pred['box'], color, thickness=2)

    # ── Build legend rows ────────────────────────────────────────────────
    rows = []

    # Matched pairs — sorted by GT index so order is consistent
    for i in sorted(gt_to_pred):
        j       = gt_to_pred[i]
        gt_name = CLASS_NAMES[gt_boxes[i]['class']]
        pr_name = CLASS_NAMES[pred_boxes[j]['class']]
        conf    = pred_boxes[j].get('conf', 0)
        correct = gt_boxes[i]['class'] == pred_boxes[j]['class']
        status  = "correct" if correct else "WRONG CLASS"
        dot     = COLOR_PRED_CORRECT if correct else COLOR_PRED_WRONG
        rows.append({'dot': dot,
                     'text': f"actual: {gt_name:<10} predicted: {pr_name:<10} conf: {conf:.2f}  {status}"})

    # False negatives (GT missed)
    for i in unmatched_gt:
        gt_name = CLASS_NAMES[gt_boxes[i]['class']]
        rows.append({'dot': COLOR_GT_MISSED,
                     'text': f"actual: {gt_name:<10} predicted: {'—':<10}                MISSED"})

    # False positives (extra predictions)
    for j in unmatched_pred:
        pr_name = CLASS_NAMES[pred_boxes[j]['class']]
        conf    = pred_boxes[j].get('conf', 0)
        rows.append({'dot': COLOR_FP,
                     'text': f"actual: {'—':<10} predicted: {pr_name:<10} conf: {conf:.2f}  FALSE POSITIVE"})

    # ── Render legend panel ──────────────────────────────────────────────
    font     = cv2.FONT_HERSHEY_SIMPLEX
    fscale   = 0.50
    fthick   = 1
    row_h    = 26
    pad      = 14
    dot_r    = 6
    header_h = 28

    legend_h = pad * 2 + header_h + row_h * max(len(rows), 1)
    legend   = np.full((legend_h, img_w, 3), 25, dtype=np.uint8)

    # Header
    cv2.putText(legend, "DETECTION RESULTS", (pad, pad + 14),
                font, 0.58, (180, 180, 180), 1, cv2.LINE_AA)

    # Color key (top-right corner, small)
    key_items = [
        (COLOR_GT_MATCHED,   "GT ok"),
        (COLOR_GT_WRONG_CLS, "GT wrong cls"),
        (COLOR_GT_MISSED,    "GT missed"),
        (COLOR_PRED_CORRECT, "Pred ok"),
        (COLOR_PRED_WRONG,   "Pred wrong"),
        (COLOR_FP,           "False pos"),
    ]
    kx = img_w - 8
    ky = pad + 4
    for col, label in reversed(key_items):
        (tw, _), _ = cv2.getTextSize(label, font, 0.38, 1)
        tx = kx - tw
        cv2.putText(legend, label, (tx, ky + 8), font, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.circle(legend, (tx - 9, ky + 5), 5, col, -1, cv2.LINE_AA)
        kx = tx - 20

    # Rows
    for idx, row in enumerate(rows):
        y  = pad + header_h + idx * row_h
        cx = pad + dot_r
        cy = y + row_h // 2
        cv2.circle(legend, (cx, cy), dot_r, row['dot'], -1, cv2.LINE_AA)
        cv2.putText(legend, row['text'], (cx + dot_r + 8, cy + 5),
                    font, fscale, (215, 215, 215), fthick, cv2.LINE_AA)

    return np.vstack([img, legend])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    type=str, required=True)
    parser.add_argument("--data",       type=str, required=True)
    parser.add_argument("--output",     type=str, default="confused_images")
    parser.add_argument("--conf",       type=float, default=0.25)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--split",      type=str, default="val",
                        choices=["train", "val", "test"])
    args = parser.parse_args()

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    data_root = Path(args.data).parent
    image_dir = data_root / "images" / args.split
    label_dir = data_root / "labels" / args.split

    if not image_dir.exists():
        print(f"ERROR: {image_dir} not found"); return

    output_dir   = Path(args.output)
    fn_dir       = output_dir / "false_negatives"
    fp_dir       = output_dir / "false_positives"
    misclass_dir = output_dir / "misclassified"
    for d in [fn_dir, fp_dir, misclass_dir]:
        d.mkdir(parents=True, exist_ok=True)

    stats         = defaultdict(int)
    error_details = defaultdict(list)

    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    print(f"Found {len(image_files)} images in '{args.split}' split\n")

    for idx, img_path in enumerate(image_files):
        print(f"[{idx+1}/{len(image_files)}] {img_path.name}", end='\r')

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        img_h, img_w = image.shape[:2]

        label_path = label_dir / f"{img_path.stem}.txt"
        gt_boxes   = parse_label_file(str(label_path), img_w, img_h)

        results    = model.predict(str(img_path), conf=args.conf, verbose=False)[0]
        pred_boxes = []
        if results.boxes is not None:
            for box in results.boxes:
                pred_boxes.append({
                    'class': int(box.cls[0].cpu()),
                    'box':   box.xyxy[0].cpu().numpy().tolist(),
                    'conf':  float(box.conf[0].cpu()),
                })

        # ── Single source-of-truth matching ──────────────────────────────
        gt_to_pred, pred_to_gt, unmatched_gt, unmatched_pred = \
            match_boxes(gt_boxes, pred_boxes, args.iou_thresh)

        def save(subdir):
            subdir.mkdir(parents=True, exist_ok=True)
            ann = make_annotated_image(image, gt_boxes, pred_boxes,
                                       gt_to_pred, pred_to_gt,
                                       unmatched_gt, unmatched_pred)
            cv2.imwrite(str(subdir / img_path.name), ann)

        # Misclassifications (matched but wrong class)
        for i, j in gt_to_pred.items():
            if gt_boxes[i]['class'] != pred_boxes[j]['class']:
                stats['misclassified'] += 1
                key = (f"predicted_{CLASS_NAMES[pred_boxes[j]['class']]}"
                       f"_actual_{CLASS_NAMES[gt_boxes[i]['class']]}")
                error_details[key].append(img_path.name)
                save(misclass_dir / key)

        # False negatives
        for i in unmatched_gt:
            stats['false_negatives'] += 1
            cname = CLASS_NAMES[gt_boxes[i]['class']]
            error_details[f"fn_{cname}"].append(img_path.name)
            save(fn_dir / cname)

        # False positives
        for j in unmatched_pred:
            stats['false_positives'] += 1
            cname = CLASS_NAMES[pred_boxes[j]['class']]
            error_details[f"fp_{cname}"].append(img_path.name)
            save(fp_dir / cname)

    print(f"\n\n{'─'*50}")
    print(f"  False Negatives (missed): {stats['false_negatives']}")
    print(f"  False Positives (extra):  {stats['false_positives']}")
    print(f"  Misclassified:            {stats['misclassified']}")
    print(f"{'─'*50}")

    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("CONFUSED IMAGES SUMMARY\n" + "="*50 + "\n\n")
        f.write(f"Model:      {args.weights}\n")
        f.write(f"Split:      {args.split}\n")
        f.write(f"Conf:       {args.conf}\n")
        f.write(f"IoU thresh: {args.iou_thresh}\n\n")
        f.write(f"False Negatives: {stats['false_negatives']}\n")
        f.write(f"False Positives: {stats['false_positives']}\n")
        f.write(f"Misclassified:   {stats['misclassified']}\n\n")

        f.write("── FALSE NEGATIVES ──\n")
        for key in sorted(k for k in error_details if k.startswith("fn_")):
            imgs = error_details[key]
            f.write(f"\n  {key[3:]}: {len(imgs)}\n")
            for n in imgs[:10]: f.write(f"    - {n}\n")
            if len(imgs) > 10: f.write(f"    ... +{len(imgs)-10} more\n")

        f.write("\n── MISCLASSIFICATIONS ──\n")
        for key in sorted(k for k in error_details if k.startswith("predicted_")):
            imgs = error_details[key]
            f.write(f"\n  {key}: {len(imgs)}\n")
            for n in imgs[:10]: f.write(f"    - {n}\n")
            if len(imgs) > 10: f.write(f"    ... +{len(imgs)-10} more\n")

        f.write(f"\nOutput: {output_dir.absolute()}\n")

    print(f"\nSummary → {summary_path}")
    print(f"Images  → {output_dir.absolute()}")


if __name__ == "__main__":
    main()