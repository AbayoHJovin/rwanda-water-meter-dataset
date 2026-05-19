#!/usr/bin/env python3
"""
fix_weak_classes.py

Since the issue is wrong-class predictions (not missing boxes),
this script does two things:

1. CONFUSION MATRIX — shows exactly which digits get confused with which
   e.g. "2 gets predicted as 3 forty times" tells you the model
   can't tell them apart → you need more 2s in training data

2. EXPORT WEAK CLASS CROPS — saves zoomed-in crops of every digit
   the model got wrong, so you can:
   a) verify the labels are correct
   b) use them to understand what more data you need to collect

Usage:
    python fix_weak_classes.py --weights best.pt --data dataset/data.yaml
    python fix_weak_classes.py --weights best.pt --data dataset/data.yaml --split train
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
DIGIT_IDS = set(range(2, 13))  # 0–9 + unknown


def iou(box1, box2):
    x1=max(box1[0],box2[0]); y1=max(box1[1],box2[1])
    x2=min(box1[2],box2[2]); y2=min(box1[3],box2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(box1[2]-box1[0])*(box1[3]-box1[1])
    a2=(box2[2]-box2[0])*(box2[3]-box2[1])
    union=a1+a2-inter
    return inter/union if union>0 else 0


def parse_labels(path, img_w, img_h):
    boxes = []
    if not Path(path).exists(): return boxes
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p)!=5: continue
            cls=int(p[0]); xc,yc,w,h=map(float,p[1:])
            x1=(xc-w/2)*img_w; y1=(yc-h/2)*img_h
            x2=(xc+w/2)*img_w; y2=(yc+h/2)*img_h
            boxes.append({'class':cls,'box':[x1,y1,x2,y2]})
    return boxes


def nms_all(preds, thresh=0.45):
    if not preds: return preds
    boxes=np.array([p['box'] for p in preds],dtype=np.float32)
    scores=np.array([p['conf'] for p in preds],dtype=np.float32)
    x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas=(x2-x1)*(y2-y1); order=scores.argsort()[::-1]; keep=[]
    while order.size>0:
        i=order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        union=areas[i]+areas[order[1:]]-inter
        order=order[1:][inter/np.maximum(union,1e-6)<thresh]
    return [preds[k] for k in keep]


def crop_digit(image, box, pad_frac=0.3):
    """Crop a digit from the image with some padding."""
    x1,y1,x2,y2 = map(int, box)
    h,w = image.shape[:2]
    pw = int((x2-x1)*pad_frac); ph = int((y2-y1)*pad_frac)
    x1=max(0,x1-pw); y1=max(0,y1-ph)
    x2=min(w,x2+pw); y2=min(h,y2+ph)
    crop = image[y1:y2, x1:x2]
    # Resize to standard size for easy visual comparison
    if crop.size > 0:
        crop = cv2.resize(crop, (80, 100))
    return crop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    required=True)
    parser.add_argument("--data",       required=True)
    parser.add_argument("--split",      default="val", choices=["train","val","test"])
    parser.add_argument("--conf",       type=float, default=0.10)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--output",     default="weak_class_analysis")
    args = parser.parse_args()

    data_root = Path(args.data).parent
    image_dir = data_root / "images" / args.split
    label_dir = data_root / "labels" / args.split
    out_dir   = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    model = YOLO(args.weights)

    print(f"Analyzing {len(image_files)} images...\n")

    # confusion[actual][predicted] = count
    confusion     = defaultdict(lambda: defaultdict(int))
    wrong_crops   = defaultdict(list)   # actual_class -> list of (crop, pred_class, conf, filename)
    class_gt_count= defaultdict(int)

    for idx, img_path in enumerate(image_files):
        print(f"  [{idx+1}/{len(image_files)}] {img_path.name}", end='\r')
        image = cv2.imread(str(img_path))
        if image is None: continue
        img_h, img_w = image.shape[:2]

        gt_boxes   = parse_labels(label_dir/f"{img_path.stem}.txt", img_w, img_h)
        results    = model.predict(str(img_path), conf=args.conf, iou=0.3, verbose=False)[0]
        pred_boxes = []
        if results.boxes is not None:
            for box in results.boxes:
                pred_boxes.append({
                    'class': int(box.cls[0].cpu()),
                    'box':   box.xyxy[0].cpu().numpy().tolist(),
                    'conf':  float(box.conf[0].cpu()),
                })
        pred_boxes = nms_all(pred_boxes, 0.45)

        # Greedy match
        pairs = sorted(
            [(iou(g['box'],p['box']),gi,pi)
             for gi,g in enumerate(gt_boxes)
             for pi,p in enumerate(pred_boxes)
             if iou(g['box'],p['box'])>=args.iou_thresh],
            reverse=True)
        gt_matched={}; pred_matched={}
        for v,gi,pi in pairs:
            if gi in gt_matched or pi in pred_matched: continue
            gt_matched[gi]=pi; pred_matched[pi]=gi

        for gi, gt in enumerate(gt_boxes):
            cls = gt['class']
            if cls not in DIGIT_IDS: continue
            class_gt_count[cls] += 1

            if gi in gt_matched:
                pi       = gt_matched[gi]
                pred_cls = pred_boxes[pi]['class']
                conf     = pred_boxes[pi]['conf']
                confusion[cls][pred_cls] += 1

                if pred_cls != cls:
                    crop = crop_digit(image, gt['box'])
                    wrong_crops[cls].append({
                        'crop': crop,
                        'pred': pred_cls,
                        'conf': conf,
                        'file': img_path.name,
                    })

    print(f"\n\nDone.\n")

    # ── 1. Print confusion matrix ────────────────────────────────────────
    digit_ids = sorted(c for c in class_gt_count if c in DIGIT_IDS)
    names     = [CLASS_NAMES[c] for c in digit_ids]

    print("CONFUSION MATRIX  (rows=actual, cols=predicted, numbers=count)")
    print("Diagonal = correct. Off-diagonal = wrong class.\n")

    col_w = 6
    header = f"  {'actual':>8}  " + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    print("  " + "─"*(10 + col_w*len(names)))

    for actual_id in digit_ids:
        row = f"  {CLASS_NAMES[actual_id]:>8}  "
        for pred_id in digit_ids:
            count = confusion[actual_id][pred_id]
            if actual_id == pred_id:
                cell = f"\033[32m{count:>{col_w}}\033[0m"   # green = correct
            elif count > 0:
                cell = f"\033[31m{count:>{col_w}}\033[0m"   # red = wrong
            else:
                cell = f"{'':>{col_w}}"
            row += cell
        gt_total = class_gt_count[actual_id]
        correct  = confusion[actual_id][actual_id]
        recall   = correct/gt_total if gt_total else 0
        wrong_sum= sum(confusion[actual_id][p] for p in digit_ids if p!=actual_id)
        row += f"   GT:{gt_total:>3}  recall:{recall:>5.1%}  wrong:{wrong_sum:>3}"
        print(row)

    print()

    # ── 2. Print top confusions ──────────────────────────────────────────
    print("TOP CONFUSIONS (pairs most often mixed up):\n")
    pair_counts = []
    for actual_id in digit_ids:
        for pred_id in digit_ids:
            if actual_id != pred_id and confusion[actual_id][pred_id] > 0:
                pair_counts.append((confusion[actual_id][pred_id], actual_id, pred_id))
    pair_counts.sort(reverse=True)

    for count, actual_id, pred_id in pair_counts[:15]:
        actual_name = CLASS_NAMES[actual_id]
        pred_name   = CLASS_NAMES[pred_id]
        gt_total    = class_gt_count[actual_id]
        pct         = count/gt_total if gt_total else 0
        bar         = "█" * count
        print(f"  actual {actual_name} → predicted {pred_name}:  {count:>3}x  ({pct:.0%} of all {actual_name}s)  {bar}")

    # ── 3. Explain each weak class and what to do ────────────────────────
    print("\n\nWHAT TO DO — per weak class:\n")
    WEAK_THRESHOLD = 0.75
    for actual_id in digit_ids:
        gt_total = class_gt_count[actual_id]
        correct  = confusion[actual_id][actual_id]
        recall   = correct/gt_total if gt_total else 0
        if recall >= WEAK_THRESHOLD:
            continue

        print(f"  ── Digit '{CLASS_NAMES[actual_id]}'  recall={recall:.0%}  ({gt_total} GT samples) ──")

        # What does it get confused with most?
        wrong = sorted(
            [(confusion[actual_id][p], CLASS_NAMES[p]) for p in digit_ids if p!=actual_id],
            reverse=True)
        top_wrong = [(c,n) for c,n in wrong if c>0][:3]

        for count, pred_name in top_wrong:
            print(f"    • confused as '{pred_name}' {count} times")

        if gt_total < 50:
            print(f"    → COLLECT MORE DATA: only {gt_total} GT samples. Need at least 100+.")
            print(f"       Go photograph meters that show '{CLASS_NAMES[actual_id]}' and label them.")
        else:
            print(f"    → DATA IS ENOUGH ({gt_total} samples) but model still confuses it.")
            print(f"       Check if labels for '{CLASS_NAMES[actual_id]}' are consistently drawn.")
            print(f"       Run: --fix-report to see specific images")
        print()

    # ── 4. Save visual crops of wrong predictions ────────────────────────
    print("Saving visual comparison sheets of wrong predictions...")
    for cls_id, items in wrong_crops.items():
        if not items: continue
        cls_name  = CLASS_NAMES[cls_id]
        crops_dir = out_dir / f"wrong_{cls_name}"
        crops_dir.mkdir(exist_ok=True)

        # Save individual crops
        for i, item in enumerate(items[:50]):
            if item['crop'].size == 0: continue
            fname = f"{cls_name}_predicted_as_{CLASS_NAMES[item['pred']]}_{i:03d}_{item['file']}"
            cv2.imwrite(str(crops_dir/fname), item['crop'])

        # Save a comparison sheet — all wrong crops on one image
        sheet_crops = [item['crop'] for item in items if item['crop'].size > 0][:40]
        if sheet_crops:
            # Pad all to same size
            H, W = 100, 80
            sheet_crops = [cv2.resize(c, (W, H)) if c.shape[:2]!=(H,W) else c
                           for c in sheet_crops]
            cols = min(10, len(sheet_crops))
            rows = (len(sheet_crops) + cols - 1) // cols
            sheet = np.zeros((rows*(H+30), cols*(W+4), 3), dtype=np.uint8)
            for i, crop in enumerate(sheet_crops):
                r, c = divmod(i, cols)
                y = r*(H+30); x = c*(W+4)
                sheet[y:y+H, x:x+W] = crop
                pred_name = CLASS_NAMES[items[i]['pred']]
                cv2.putText(sheet, f"→{pred_name}", (x, y+H+18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,255), 1)

            # Header
            header_img = np.zeros((40, sheet.shape[1], 3), dtype=np.uint8)
            cv2.putText(header_img,
                        f"Actual digit: '{cls_name}'  —  all {len(sheet_crops)} wrong predictions",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            final = np.vstack([header_img, sheet])
            cv2.imwrite(str(out_dir/f"sheet_wrong_{cls_name}.jpg"), final)
            print(f"  Saved sheet for '{cls_name}': {len(sheet_crops)} wrong crops → {out_dir}/sheet_wrong_{cls_name}.jpg")

    # ── 5. Data collection priority list ─────────────────────────────────
    print("\n\nDATA COLLECTION PRIORITY (sorted by urgency):\n")
    urgency = []
    for actual_id in digit_ids:
        gt    = class_gt_count[actual_id]
        rec   = confusion[actual_id][actual_id] / gt if gt else 0
        score = (1 - rec) * 100 + max(0, 100 - gt)  # low recall + low count = urgent
        urgency.append((score, actual_id, gt, rec))
    urgency.sort(reverse=True)

    print(f"  {'Priority':<10} {'Digit':<8} {'GT count':>9}  {'Recall':>8}  Action")
    print("  " + "─"*65)
    for rank, (score, cid, gt, rec) in enumerate(urgency, 1):
        if rec >= WEAK_THRESHOLD and gt >= 50:
            action = "✓ ok"
        elif gt < 50:
            action = f"⚠ COLLECT MORE — need {max(0, 100-gt)} more labeled examples"
        elif gt < 100:
            action = f"⚠ COLLECT MORE — ideally get to 150+ examples"
        else:
            action = "⚠ CHECK LABELS — enough data but model still confused"
        print(f"  #{rank:<9} {CLASS_NAMES[cid]:<8} {gt:>9}  {rec:>7.1%}  {action}")


if __name__ == "__main__":
    main()