#!/usr/bin/env python3
"""
diagnose_dataset.py

Run ALL diagnostics on your dataset + model to pinpoint exactly
why the model is missing boxes. No guessing.

Tests:
  1. Label audit        — are all digits actually labeled? counts per image
  2. Box size audit     — are digit boxes suspiciously small or huge?
  3. Box overlap audit  — do GT boxes overlap each other? (labeling mistake)
  4. Class distribution — which digits are rare? (undertrained classes)
  5. Model confidence   — what confidence does the model give missed digits?
  6. IoU distribution   — what IoU do predictions have with GT? (label alignment)
  7. Edge digit audit   — are rightmost/leftmost digits being missed more?
  8. Per-class recall   — which digit classes get missed the most?

Usage:
    python diagnose_dataset.py --weights best.pt --data dataset/data.yaml
    python diagnose_dataset.py --weights best.pt --data dataset/data.yaml --split train
    python diagnose_dataset.py --weights best.pt --data dataset/data.yaml --fix-report
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = [
    "meter", "window",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "unknown",
]
DIGIT_CLASS_IDS = set(range(2, 12))  # classes 2–11 = digits 0–9


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
    if not label_path.exists():
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
            boxes.append({
                'class': cls,
                'box': [x1, y1, x2, y2],
                'w_norm': w, 'h_norm': h,
                'xc_norm': xc, 'yc_norm': yc,
            })
    return boxes


def nms_cross_class(pred_boxes, iou_thresh=0.45):
    if not pred_boxes:
        return pred_boxes
    boxes  = np.array([p['box']  for p in pred_boxes], dtype=np.float32)
    scores = np.array([p['conf'] for p in pred_boxes], dtype=np.float32)
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        union=areas[i]+areas[order[1:]]-inter
        order=order[1:][inter/np.maximum(union,1e-6)<iou_thresh]
    return [pred_boxes[k] for k in keep]


def sep(title="", width=60):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*pad}")
    else:
        print("─" * width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    required=True)
    parser.add_argument("--data",       required=True)
    parser.add_argument("--split",      default="val", choices=["train","val","test"])
    parser.add_argument("--conf",       type=float, default=0.10,
                        help="Low conf on purpose — we want to see what model almost detects")
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--fix-report", action="store_true",
                        help="Save a fix_report.txt listing every image+box that needs attention")
    args = parser.parse_args()

    data_root = Path(args.data).parent
    image_dir = data_root / "images" / args.split
    label_dir = data_root / "labels" / args.split

    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    print(f"\nLoading model: {args.weights}")
    model = YOLO(args.weights)
    print(f"Dataset split: {args.split}  ({len(image_files)} images)")
    print(f"Conf threshold: {args.conf} (intentionally low to catch weak detections)")
    print(f"IoU match threshold: {args.iou_thresh}")

    # ── Accumulators ────────────────────────────────────────────────────
    class_gt_count      = defaultdict(int)   # GT counts per class
    class_detected      = defaultdict(int)   # correctly matched
    class_missed        = defaultdict(int)   # GT with no pred match
    class_wrong         = defaultdict(int)   # matched but wrong class

    box_widths          = defaultdict(list)  # GT box widths (normalized) per class
    box_heights         = defaultdict(list)
    gt_overlap_cases    = []                 # images where GT boxes overlap each other
    edge_missed         = defaultdict(int)   # leftmost/rightmost digit missed counts
    edge_total          = defaultdict(int)

    iou_of_missed       = []  # best IoU a missed GT had with ANY prediction (any class)
    conf_of_missed      = []  # conf of that best prediction
    images_missing_label= []  # images with 0 digit labels

    fix_report_lines    = []

    for idx, img_path in enumerate(image_files):
        print(f"  [{idx+1}/{len(image_files)}] {img_path.name}", end='\r')

        image = cv2.imread(str(img_path))
        if image is None: continue
        img_h, img_w = image.shape[:2]

        label_path = label_dir / f"{img_path.stem}.txt"
        gt_boxes   = parse_label_file(label_path, img_w, img_h)

        digit_gt = [b for b in gt_boxes if b['class'] in DIGIT_CLASS_IDS]

        if len(digit_gt) == 0:
            images_missing_label.append(img_path.name)

        # ── Test 2: box size audit ───────────────────────────────────────
        for b in gt_boxes:
            box_widths[b['class']].append(b['w_norm'])
            box_heights[b['class']].append(b['h_norm'])
            if b['w_norm'] < 0.01 or b['h_norm'] < 0.01:
                fix_report_lines.append(
                    f"TINY_BOX  {img_path.name}  class={CLASS_NAMES[b['class']]}  "
                    f"w={b['w_norm']:.4f} h={b['h_norm']:.4f}")
            if b['w_norm'] > 0.5 or b['h_norm'] > 0.5:
                fix_report_lines.append(
                    f"HUGE_BOX  {img_path.name}  class={CLASS_NAMES[b['class']]}  "
                    f"w={b['w_norm']:.4f} h={b['h_norm']:.4f}")

        # ── Test 3: GT box overlap audit ─────────────────────────────────
        for i in range(len(digit_gt)):
            for j in range(i+1, len(digit_gt)):
                overlap = iou(digit_gt[i]['box'], digit_gt[j]['box'])
                if overlap > 0.3:
                    gt_overlap_cases.append(
                        f"{img_path.name}: GT[{CLASS_NAMES[digit_gt[i]['class']]}] "
                        f"and GT[{CLASS_NAMES[digit_gt[j]['class']]}] overlap IoU={overlap:.2f}")

        # ── Class distribution ───────────────────────────────────────────
        for b in gt_boxes:
            class_gt_count[b['class']] += 1

        # ── Run model ────────────────────────────────────────────────────
        results = model.predict(str(img_path), conf=args.conf, iou=0.3, verbose=False)[0]
        pred_boxes = []
        if results.boxes is not None:
            for box in results.boxes:
                pred_boxes.append({
                    'class': int(box.cls[0].cpu()),
                    'box':   box.xyxy[0].cpu().numpy().tolist(),
                    'conf':  float(box.conf[0].cpu()),
                })
        pred_boxes = nms_cross_class(pred_boxes, 0.45)

        # ── Greedy match ─────────────────────────────────────────────────
        pairs = sorted(
            [(iou(g['box'], p['box']), gi, pi)
             for gi, g in enumerate(gt_boxes)
             for pi, p in enumerate(pred_boxes)
             if iou(g['box'], p['box']) >= args.iou_thresh],
            reverse=True)
        gt_matched = {}; pred_matched = {}
        for v, gi, pi in pairs:
            if gi in gt_matched or pi in pred_matched: continue
            gt_matched[gi] = pi; pred_matched[pi] = gi

        for gi, gt in enumerate(gt_boxes):
            cls = gt['class']
            if gi in gt_matched:
                pi = gt_matched[gi]
                if pred_boxes[pi]['class'] == cls:
                    class_detected[cls] += 1
                else:
                    class_wrong[cls] += 1
            else:
                class_missed[cls] += 1
                # What was the best IoU this missed GT had with any pred?
                best_iou  = 0.0
                best_conf = 0.0
                for p in pred_boxes:
                    v = iou(gt['box'], p['box'])
                    if v > best_iou:
                        best_iou  = v
                        best_conf = p['conf']
                iou_of_missed.append(best_iou)
                conf_of_missed.append(best_conf)
                if best_iou > 0.05:
                    fix_report_lines.append(
                        f"NEAR_MISS  {img_path.name}  "
                        f"GT={CLASS_NAMES[cls]}  best_pred_iou={best_iou:.3f}  "
                        f"best_pred_conf={best_conf:.2f}  "
                        f"→ {'label misaligned' if best_iou < args.iou_thresh else 'low conf only'}")
                else:
                    fix_report_lines.append(
                        f"INVISIBLE  {img_path.name}  "
                        f"GT={CLASS_NAMES[cls]}  best_pred_iou={best_iou:.3f}  "
                        f"→ model has no idea this exists (need more training data)")

        # ── Test 7: edge digit audit ─────────────────────────────────────
        if digit_gt:
            sorted_by_x = sorted(digit_gt, key=lambda b: b['box'][0])
            for pos_label, b in [("leftmost", sorted_by_x[0]), ("rightmost", sorted_by_x[-1])]:
                cls = b['class']
                matched = any(
                    gi in gt_matched and gt_boxes[gi] is b
                    for gi in range(len(gt_boxes))
                )
                # find index of b in gt_boxes
                for gi, g in enumerate(gt_boxes):
                    if g is b:
                        edge_total[pos_label] += 1
                        if gi not in gt_matched:
                            edge_missed[pos_label] += 1
                        break

    print(f"\n\nDone processing {len(image_files)} images.\n")

    # ════════════════════════════════════════════════════════════════
    sep("TEST 1 — LABEL COMPLETENESS")
    print(f"  Images with ZERO digit labels: {len(images_missing_label)}")
    if images_missing_label:
        print("  ⚠ These images have no digit GT boxes at all:")
        for n in images_missing_label[:20]:
            print(f"    - {n}")
        if len(images_missing_label) > 20:
            print(f"    ... and {len(images_missing_label)-20} more")
        print("  → GO BACK AND LABEL THESE. The model can never learn from unlabeled images.")
    else:
        print("  ✓ All images have at least one digit label.")

    sep("TEST 2 — BOX SIZE DISTRIBUTION")
    print(f"  {'Class':<12} {'Count':>6}  {'Avg W':>7}  {'Avg H':>7}  {'Min W':>7}  {'Max W':>7}")
    for cls_id in sorted(class_gt_count):
        ws = box_widths[cls_id]
        hs = box_heights[cls_id]
        if not ws: continue
        print(f"  {CLASS_NAMES[cls_id]:<12} {len(ws):>6}  "
              f"{np.mean(ws):>7.3f}  {np.mean(hs):>7.3f}  "
              f"{np.min(ws):>7.3f}  {np.max(ws):>7.3f}")
    print("\n  ⚠ If digit avg width varies a lot across classes, some were labeled too big/small.")

    sep("TEST 3 — GT BOX OVERLAPS (labeling mistakes)")
    if gt_overlap_cases:
        print(f"  Found {len(gt_overlap_cases)} cases where two GT digit boxes overlap > 0.3 IoU:")
        for c in gt_overlap_cases[:15]:
            print(f"    ⚠ {c}")
        if len(gt_overlap_cases) > 15:
            print(f"    ... and {len(gt_overlap_cases)-15} more")
        print("  → These are labeling errors. Two boxes shouldn't overlap that much.")
    else:
        print("  ✓ No significant GT box overlaps found.")

    sep("TEST 4 — CLASS DISTRIBUTION (are some digits rare?)")
    total_gt = sum(class_gt_count.values())
    print(f"  {'Class':<12} {'GT count':>9}  {'% of total':>11}")
    for cls_id in sorted(class_gt_count):
        n   = class_gt_count[cls_id]
        pct = 100 * n / total_gt if total_gt else 0
        warn = "  ⚠ RARE — model undertrained on this" if pct < 3.0 and cls_id in DIGIT_CLASS_IDS else ""
        print(f"  {CLASS_NAMES[cls_id]:<12} {n:>9}  {pct:>10.1f}%{warn}")

    sep("TEST 5 — PER-CLASS RECALL (which digits get missed?)")
    print(f"  {'Class':<12} {'GT':>5}  {'Detected':>9}  {'Wrong cls':>10}  {'Missed':>7}  {'Recall':>8}")
    for cls_id in sorted(class_gt_count):
        if cls_id not in DIGIT_CLASS_IDS:
            continue
        gt  = class_gt_count[cls_id]
        det = class_detected[cls_id]
        mis = class_missed[cls_id]
        wrg = class_wrong[cls_id]
        rec = det / gt if gt else 0
        warn = "  ⚠ LOW" if rec < 0.7 else ""
        print(f"  {CLASS_NAMES[cls_id]:<12} {gt:>5}  {det:>9}  {wrg:>10}  {mis:>7}  {rec:>7.1%}{warn}")

    sep("TEST 6 — IOY DISTRIBUTION OF MISSED BOXES")
    if iou_of_missed:
        bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01]
        labels = ["0–0.05 (INVISIBLE)", "0.05–0.1", "0.1–0.2", "0.2–0.3",
                  "0.3–0.5 (near miss)", "0.5+ (label offset only)"]
        arr = np.array(iou_of_missed)
        print("  Best IoU a missed GT box had with ANY prediction:\n")
        for i, lbl in enumerate(labels):
            count = np.sum((arr >= bins[i]) & (arr < bins[i+1]))
            bar   = "█" * int(30 * count / len(arr))
            print(f"  {lbl:<30} {count:>4}  {bar}")
        print(f"\n  Interpretation:")
        invisible = np.sum(arr < 0.05)
        near_miss = np.sum((arr >= 0.1) & (arr < args.iou_thresh))
        print(f"  • {invisible} boxes are INVISIBLE to the model → need more training data / better labels")
        print(f"  • {near_miss} boxes are NEAR MISSES → label boxes are slightly misaligned, fix the labels")
        print(f"  • {np.sum(arr >= args.iou_thresh)} boxes matched but below conf threshold → lower --conf or train more")
    else:
        print("  No missed boxes found — model is perfect or all GT matched.")

    sep("TEST 7 — EDGE DIGIT ANALYSIS")
    for pos in ["leftmost", "rightmost"]:
        total  = edge_total[pos]
        missed = edge_missed[pos]
        rate   = missed/total if total else 0
        warn   = "  ⚠ HIGH MISS RATE — boxes at edge of display window are cut off" if rate > 0.2 else "  ✓"
        print(f"  {pos} digit: {missed}/{total} missed ({rate:.0%}){warn}")

    sep("SUMMARY & WHAT TO DO")
    total_missed = sum(class_missed[c] for c in DIGIT_CLASS_IDS)
    total_gt_d   = sum(class_gt_count[c] for c in DIGIT_CLASS_IDS)
    overall_miss_rate = total_missed / total_gt_d if total_gt_d else 0

    print(f"\n  Overall digit miss rate: {overall_miss_rate:.1%}  ({total_missed}/{total_gt_d} digits missed)\n")

    if images_missing_label:
        print(f"  [CRITICAL] {len(images_missing_label)} images have NO digit labels → label them first")

    if gt_overlap_cases:
        print(f"  [HIGH]     {len(gt_overlap_cases)} overlapping GT boxes → fix those labels")

    if iou_of_missed:
        invisible = sum(1 for v in iou_of_missed if v < 0.05)
        near      = sum(1 for v in iou_of_missed if 0.05 <= v < args.iou_thresh)
        if invisible > near:
            print(f"  [HIGH]     Most missed boxes are INVISIBLE ({invisible} vs {near} near-misses)")
            print(f"             → Your model needs more training data on the missed classes")
            print(f"             → Check class distribution — are those classes underrepresented?")
        else:
            print(f"  [HIGH]     Most missed boxes are NEAR MISSES ({near} vs {invisible} invisible)")
            print(f"             → Your GT label boxes are slightly misaligned from where the model detects")
            print(f"             → Re-label those digit boxes more tightly around the digit only")

    rare_classes = [CLASS_NAMES[c] for c in DIGIT_CLASS_IDS
                    if class_gt_count[c] < 0.03 * total_gt_d and class_gt_count[c] > 0]
    if rare_classes:
        print(f"  [MEDIUM]   Rare digit classes: {rare_classes}")
        print(f"             → Collect/label more images containing these digits")

    sep()

    if args.fix_report:
        report_path = Path("fix_report.txt")
        with open(report_path, 'w') as f:
            f.write("FIX REPORT — images and boxes that need attention\n")
            f.write("="*70 + "\n\n")
            f.write("NEAR_MISS  = model almost detected it, label box is probably slightly off\n")
            f.write("INVISIBLE  = model has zero awareness of this box, need more training data\n")
            f.write("TINY_BOX   = GT box is suspiciously small, may be a labeling error\n")
            f.write("HUGE_BOX   = GT box is suspiciously large, may be a labeling error\n\n")
            for line in fix_report_lines:
                f.write(line + "\n")
        print(f"\n  Fix report saved → {report_path.absolute()}")
        print(f"  ({len(fix_report_lines)} issues found)")


if __name__ == "__main__":
    main()