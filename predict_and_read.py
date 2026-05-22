#!/usr/bin/env python3
"""
predict_and_read.py

Run prediction and overlay:
  - Bounding boxes around each detected digit
  - The digit value printed above each box
  - A clean "METER READING: 0 2 3 4 5" banner at the top

Usage:
    python predict_and_read.py --weights best.pt --source image.jpg
    python predict_and_read.py --weights best.pt --source images_folder/
    python predict_and_read.py --weights best.pt --source image.jpg --output result.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = [
    "meter", "window",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "unknown",
]
DIGIT_CLASS_IDS = set(range(2, 13))   # classes 2–12 = digits 0–9 + unknown

# Colors BGR
COLOR_BOX    = (0, 220, 0)      # green box around each digit
COLOR_LABEL  = (0, 0, 0)        # black text on label
COLOR_BG     = (0, 220, 0)      # green label background
COLOR_BANNER = (20, 20, 20)     # dark banner background
COLOR_READING= (0, 230, 100)    # reading text color


def nms_all(preds, thresh=0.45):
    if not preds:
        return preds
    boxes  = np.array([p['box']  for p in preds], dtype=np.float32)
    scores = np.array([p['conf'] for p in preds], dtype=np.float32)
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        union=areas[i]+areas[order[1:]]-inter
        order=order[1:][inter/np.maximum(union,1e-6)<thresh]
    return [preds[k] for k in keep]


def draw_label(img, text, x1, y1, font, font_scale, thickness):
    """Draw a filled rectangle with text above a box."""
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 3
    # Label background
    lx1 = x1
    ly1 = max(0, y1 - th - baseline - pad*2)
    lx2 = x1 + tw + pad*2
    ly2 = max(th + baseline, y1)
    cv2.rectangle(img, (lx1, ly1), (lx2, ly2), COLOR_BG, -1)
    cv2.putText(img, text, (lx1 + pad, ly2 - baseline - pad),
                font, font_scale, COLOR_LABEL, thickness, cv2.LINE_AA)


def render_result(image_path: Path, model: YOLO, conf: float, output_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  Cannot read image: {image_path}")
        return
    img_h, img_w = image.shape[:2]

    # ── Run prediction ───────────────────────────────────────────────────
    results = model.predict(str(image_path), conf=conf, iou=0.45, verbose=False)[0]

    preds = []
    if results.boxes is not None:
        for box in results.boxes:
            preds.append({
                'class': int(box.cls[0].cpu()),
                'box':   box.xyxy[0].cpu().numpy().tolist(),
                'conf':  float(box.conf[0].cpu()),
            })
    preds = nms_all(preds, thresh=0.45)

    # ── Separate digits from meter/window ───────────────────────────────
    digits   = [p for p in preds if p['class'] in DIGIT_CLASS_IDS]
    non_digit= [p for p in preds if p['class'] not in DIGIT_CLASS_IDS]

    # Sort digits left→right by x-center to get reading order
    digits.sort(key=lambda p: (p['box'][0] + p['box'][2]) / 2)

    # ── Build the meter reading string ──────────────────────────────────
    reading_parts = [CLASS_NAMES[d['class']] for d in digits]
    reading       = " ".join(reading_parts) if reading_parts else "— no digits detected —"

    # ── Draw on image ────────────────────────────────────────────────────
    canvas = image.copy()
    font        = cv2.FONT_HERSHEY_SIMPLEX
    box_thick   = 2
    label_scale = 0.55
    label_thick = 1

    # Draw meter/window boxes subtly (thin, no label clutter)
    for p in non_digit:
        x1,y1,x2,y2 = map(int, p['box'])
        cv2.rectangle(canvas, (x1,y1), (x2,y2), (200,200,50), 1)

    # Draw digit boxes + labels
    for d in digits:
        x1,y1,x2,y2 = map(int, d['box'])
        name = CLASS_NAMES[d['class']]
        conf_val = d['conf']

        # Box
        cv2.rectangle(canvas, (x1,y1), (x2,y2), COLOR_BOX, box_thick)

        # Label above box: digit + confidence
        label_text = f"{name} {conf_val:.2f}"
        draw_label(canvas, label_text, x1, y1, font, label_scale, label_thick)

    # ── Reading banner at top ────────────────────────────────────────────
    banner_h = 52
    banner   = np.full((banner_h, img_w, 3), 20, dtype=np.uint8)

    # Left: "READING:"
    cv2.putText(banner, "METER READING:", (12, 34),
                font, 0.65, (160,160,160), 1, cv2.LINE_AA)

    # Right: the actual digits, large and green
    reading_display = " ".join(reading_parts) if reading_parts else "—"
    (rw, _), _ = cv2.getTextSize(reading_display, font, 1.1, 2)
    rx = min(200, img_w - rw - 12)
    cv2.putText(banner, reading_display, (rx, 38),
                font, 1.1, COLOR_READING, 2, cv2.LINE_AA)

    final = np.vstack([banner, canvas])

    cv2.imwrite(str(output_path), final)
    print(f"  Reading : {reading_display}")
    print(f"  Saved   : {output_path}")
    return reading_display


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True,          help="Path to best.pt")
    parser.add_argument("--source",  required=True,          help="Image file or folder")
    parser.add_argument("--output",  default=None,           help="Output path (file or folder). Default: adds _reading suffix")
    parser.add_argument("--conf",    type=float, default=0.25)
    args = parser.parse_args()

    model  = YOLO(args.weights)
    source = Path(args.source)

    VALID = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if source.is_file():
        images = [source]
    elif source.is_dir():
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in VALID)
    else:
        print(f"Source not found: {source}"); return

    print(f"Model  : {args.weights}")
    print(f"Images : {len(images)}\n")

    for img_path in images:
        if args.output:
            out_path = Path(args.output)
            if len(images) > 1:
                out_path = out_path / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = img_path.parent / f"{img_path.stem}_reading{img_path.suffix}"

        render_result(img_path, model, args.conf, out_path)


if __name__ == "__main__":
    main()