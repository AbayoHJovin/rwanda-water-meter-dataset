# #!/usr/bin/env python3
# """
# predict_and_read.py  —  Water-meter digit reader (two-model pipeline)
# ======================================================================

# Pipeline (mirrors the production app.py logic):
#   1. Decode image safely (handles RGBA / grayscale PNG)
#   2. OBB model  → detect tilt angle of the meter window
#   3. Deskew     → rotate to upright (white border, no ghost digits)
#   4. Try 0° and 180° → pick the orientation that scores higher
#   5. Digit model → detect and sort digits left → right
#   6. Annotate   → draw boxes, labels, and a reading banner

# Usage:
#     python predict_and_read.py \\
#         --digit-weights digit_model.pt \\
#         --obb-weights   obb_model.pt \\
#         --source        image.jpg

#     python predict_and_read.py \\
#         --digit-weights digit_model.pt \\
#         --obb-weights   obb_model.pt \\
#         --source        images_folder/ \\
#         --output        results/

#     # Skip OBB deskewing (single-model mode, no angle correction):
#     python predict_and_read.py --digit-weights digit_model.pt --source image.jpg
# """

# from __future__ import annotations

# import argparse
# import math
# from pathlib import Path

# import cv2
# import numpy as np
# from ultralytics import YOLO

# # ── Class map (must match training labels) ────────────────────────────────────
# CLASS_NAMES = [
#     "meter", "window",
#     "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
#     "unknown",
# ]
# DIGIT_CLASS_IDS = set(range(2, 12))   # indices 2–11  →  digits "0"–"9"
# #  index 12 ("unknown") is intentionally excluded from the reading

# # ── Visual constants (BGR) ─────────────────────────────────────────────────────
# COLOR_BOX    = (0, 215, 80)    # green box around each digit
# COLOR_LABEL  = (0, 215, 80)    # green label background
# COLOR_BANNER = (18, 18, 18)    # near-black banner background
# COLOR_READING= (0, 215, 80)    # large reading text


# # ══════════════════════════════════════════════════════════════════════════════
# #  IMAGE DECODING
# # ══════════════════════════════════════════════════════════════════════════════

# def decode_image(path: Path) -> np.ndarray:
#     """
#     Load an image to a solid BGR array.

#     Why not plain cv2.imread?
#     PNG files with an alpha channel read via IMREAD_COLOR get their alpha
#     dropped and transparent areas become black (value 0).  On a cropped meter
#     shot those large dark regions can be mis-detected as meter content.

#     Fix: read with IMREAD_UNCHANGED, detect 4-channel (BGRA) images, and
#     composite them onto a plain white background using the alpha as a mask.
#     """
#     arr   = np.fromfile(str(path), dtype=np.uint8)
#     image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

#     if image is None:
#         raise ValueError(f"Cannot read image: {path}")

#     if image.ndim == 3 and image.shape[2] == 4:       # BGRA / RGBA
#         bgr   = image[:, :, :3].astype(np.float32)
#         alpha = image[:, :, 3:4].astype(np.float32) / 255.0
#         white = np.full_like(bgr, 255.0)
#         image = (bgr * alpha + white * (1.0 - alpha)).astype(np.uint8)
#         print("    [decode] RGBA → composited onto white background")
#     elif image.ndim == 2:                               # grayscale
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

#     return image


# # ══════════════════════════════════════════════════════════════════════════════
# #  OBB: ANGLE DETECTION & DESKEWING
# # ══════════════════════════════════════════════════════════════════════════════

# def get_window_angle(image: np.ndarray, obb_model: YOLO,
#                      obb_conf: float = 0.25) -> tuple[float, float]:
#     """
#     Run the OBB model to find the tilt angle of the meter window.

#     Returns (angle_degrees, best_confidence).
#     Returns (0.0, 0.0) when nothing is detected.

#     The OBB model is expected to have class 1 = 'window'.
#     We take the highest-confidence 'window' detection and read its rotation
#     from xywhr[:, 4] (radians).
#     """
#     import tempfile, os
#     fd, tmp = tempfile.mkstemp(suffix=".jpg")
#     os.close(fd)
#     cv2.imwrite(tmp, image)

#     try:
#         results = obb_model.predict(tmp, conf=obb_conf, verbose=False)[0]
#     finally:
#         os.unlink(tmp)

#     if results.obb is None or len(results.obb) == 0:
#         return 0.0, 0.0

#     best_conf  = -1.0
#     best_angle = 0.0
#     for box in results.obb:
#         cls_id = int(box.cls[0].cpu())
#         conf_v = float(box.conf[0].cpu())
#         if cls_id == 1 and conf_v > best_conf:        # class 1 = 'window'
#             best_conf  = conf_v
#             xywhr      = box.xywhr[0].cpu().numpy()
#             best_angle = math.degrees(float(xywhr[4]))

#     return best_angle, best_conf


# def deskew(image: np.ndarray, angle_deg: float) -> np.ndarray:
#     """
#     Rotate image by -angle_deg, expanding canvas to avoid clipping.

#     Border mode is BORDER_CONSTANT (white).
#     Using BORDER_REFLECT_101 mirrors meter content into the new border strips,
#     producing ghost digit detections.  A white border is invisible to the model.
#     """
#     if abs(angle_deg) < 0.5:
#         return image

#     h, w   = image.shape[:2]
#     cx, cy = w / 2.0, h / 2.0
#     M      = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)

#     cos_a = abs(M[0, 0])
#     sin_a = abs(M[0, 1])
#     new_w = int(h * sin_a + w * cos_a)
#     new_h = int(h * cos_a + w * sin_a)
#     M[0, 2] += (new_w - w) / 2.0
#     M[1, 2] += (new_h - h) / 2.0

#     return cv2.warpAffine(
#         image, M, (new_w, new_h),
#         flags      = cv2.INTER_LINEAR,
#         borderMode = cv2.BORDER_CONSTANT,
#         borderValue= (255, 255, 255),
#     )


# # ══════════════════════════════════════════════════════════════════════════════
# #  NMS (cross-class, confidence-ranked)
# # ══════════════════════════════════════════════════════════════════════════════

# def nms_cross_class(preds: list[dict], iou_thresh: float = 0.40) -> list[dict]:
#     """
#     Suppress overlapping boxes across all classes, keeping higher-confidence
#     detections.  This prevents the same dial digit from being counted twice
#     when two classes fire on the same region.
#     """
#     if not preds:
#         return preds

#     boxes  = np.array([p["xyxy"] for p in preds], dtype=np.float32)
#     scores = np.array([p["conf"] for p in preds], dtype=np.float32)
#     x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
#     areas  = (x2 - x1) * (y2 - y1)
#     order  = scores.argsort()[::-1]
#     keep   = []

#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])
#         inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
#         union = areas[i] + areas[order[1:]] - inter
#         order = order[1:][inter / np.maximum(union, 1e-6) < iou_thresh]

#     return [preds[k] for k in keep]


# # ══════════════════════════════════════════════════════════════════════════════
# #  DIGIT DETECTION
# # ══════════════════════════════════════════════════════════════════════════════

# def read_digits(image: np.ndarray, digit_model: YOLO,
#                 conf: float = 0.25, iou: float = 0.35) -> list[dict]:
#     """
#     Run the digit model on *image* and return detections sorted left → right.

#     Each detection dict:
#         { cls_id, name, conf, xyxy, x_center, y_center }
#     """
#     import tempfile, os
#     fd, tmp = tempfile.mkstemp(suffix=".jpg")
#     os.close(fd)
#     cv2.imwrite(tmp, image)

#     try:
#         results = digit_model.predict(tmp, conf=conf, iou=iou, verbose=False)[0]
#     finally:
#         os.unlink(tmp)

#     if results.boxes is None:
#         return []

#     raw = []
#     for box in results.boxes:
#         cls_id = int(box.cls[0].cpu())
#         if cls_id not in DIGIT_CLASS_IDS:
#             continue
#         conf_v   = float(box.conf[0].cpu())
#         xyxy     = box.xyxy[0].cpu().numpy().tolist()
#         x_center = (xyxy[0] + xyxy[2]) / 2.0
#         y_center = (xyxy[1] + xyxy[3]) / 2.0
#         raw.append({
#             "cls_id":   cls_id,
#             "name":     CLASS_NAMES[cls_id],
#             "conf":     round(conf_v, 4),
#             "xyxy":     xyxy,
#             "x_center": x_center,
#             "y_center": y_center,
#         })

#     raw = nms_cross_class(raw, iou_thresh=0.40)
#     raw.sort(key=lambda d: d["x_center"])
#     return raw


# # ══════════════════════════════════════════════════════════════════════════════
# #  ORIENTATION SELECTION  (0° vs 180°)
# # ══════════════════════════════════════════════════════════════════════════════

# def orientation_score(digits: list[dict]) -> float:
#     """
#     Score how well-oriented a digit set is.  Higher = better.

#     Rewards:
#       • More digits detected                 (+10 per digit)
#       • Higher average confidence            (+5 × avg_conf)

#     Penalises:
#       • High Y-spread (digits not on a horizontal line)
#         A meter read upside-down produces digits scattered vertically.
#                                              (−8 × normalised Y-std)
#     """
#     if not digits:
#         return -1.0

#     y_centers  = np.array([d["y_center"] for d in digits])
#     heights    = np.array([d["xyxy"][3] - d["xyxy"][1] for d in digits])
#     avg_h      = float(np.mean(heights)) or 1.0
#     y_std_norm = float(np.std(y_centers)) / avg_h
#     avg_conf   = float(np.mean([d["conf"] for d in digits]))
#     n          = len(digits)

#     return n * 10.0 + avg_conf * 5.0 - y_std_norm * 8.0


# def best_orientation(image: np.ndarray, digit_model: YOLO,
#                      conf: float, iou: float) -> tuple[list[dict], np.ndarray, bool]:
#     """
#     Try the image at 0° and 180°.  Return whichever orientation scores higher.

#     Returns: (digits, image_used, was_flipped)
#     """
#     digits_0  = read_digits(image, digit_model, conf, iou)
#     score_0   = orientation_score(digits_0)

#     image_180  = cv2.rotate(image, cv2.ROTATE_180)
#     digits_180 = read_digits(image_180, digit_model, conf, iou)
#     score_180  = orientation_score(digits_180)

#     print(f"    [orient] 0°  score={score_0:.2f}  digits={len(digits_0)}")
#     print(f"    [orient] 180° score={score_180:.2f}  digits={len(digits_180)}")

#     if score_180 > score_0:
#         print("    [orient] → chose 180° flip")
#         return digits_180, image_180, True

#     print("    [orient] → chose 0° (no flip)")
#     return digits_0, image, False


# # ══════════════════════════════════════════════════════════════════════════════
# #  ANNOTATION
# # ══════════════════════════════════════════════════════════════════════════════

# def annotate(image: np.ndarray, digits: list[dict]) -> np.ndarray:
#     """Draw green bounding boxes and digit+confidence labels onto *image*."""
#     canvas = image.copy()
#     font   = cv2.FONT_HERSHEY_SIMPLEX

#     for d in digits:
#         x1, y1, x2, y2 = map(int, d["xyxy"])
#         label = f"{d['name']} {d['conf']:.2f}"

#         # Bounding box
#         cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_BOX, 2)

#         # Label background + text above box
#         (tw, th), baseline = cv2.getTextSize(label, font, 0.52, 1)
#         pad = 4
#         ly  = max(th + baseline + pad, y1)
#         cv2.rectangle(canvas,
#                       (x1, ly - th - baseline - pad),
#                       (x1 + tw + pad * 2, ly),
#                       COLOR_LABEL, -1)
#         cv2.putText(canvas, label,
#                     (x1 + pad, ly - baseline - pad // 2),
#                     font, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

#     return canvas


# def build_banner(width: int, reading: str,
#                  angle: float, obb_conf: float, flipped: bool) -> np.ndarray:
#     """
#     Build the top status banner showing the reading and pipeline metadata.
#     """
#     h      = 58
#     banner = np.full((h, width, 3), COLOR_BANNER, dtype=np.uint8)
#     font   = cv2.FONT_HERSHEY_SIMPLEX

#     # Left: large green reading
#     display = reading if reading else "—"
#     cv2.putText(banner, f"METER READING:  {display}",
#                 (16, 38), font, 0.95, COLOR_READING, 2, cv2.LINE_AA)

#     # Right: angle / flip metadata (small grey text)
#     parts = []
#     if obb_conf > 0:
#         parts.append(f"angle {angle:+.1f}°  (obb conf {obb_conf:.2f})")
#     else:
#         parts.append("no OBB rotation detected")
#     if flipped:
#         parts.append("180° flip applied")
#     info = "  |  ".join(parts)

#     (tw, _), _ = cv2.getTextSize(info, font, 0.42, 1)
#     cv2.putText(banner, info, (width - tw - 16, 38),
#                 font, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

#     return banner


# # ══════════════════════════════════════════════════════════════════════════════
# #  MAIN RENDER FUNCTION
# # ══════════════════════════════════════════════════════════════════════════════

# def render_result(image_path: Path,
#                   digit_model: YOLO,
#                   obb_model:   YOLO | None,
#                   conf:        float,
#                   iou:         float,
#                   obb_conf:    float,
#                   output_path: Path) -> str:
#     """
#     Full pipeline for a single image:
#       decode → (OBB deskew) → orientation select → digit detect → annotate → save
#     """
#     print(f"\n  Image : {image_path.name}")

#     # ── 1. Decode (RGBA / grayscale safe) ────────────────────────────────
#     try:
#         image = decode_image(image_path)
#     except ValueError as e:
#         print(f"  ERROR: {e}")
#         return ""

#     # ── 2. OBB deskew ────────────────────────────────────────────────────
#     angle     = 0.0
#     obb_conf_val = 0.0
#     if obb_model is not None:
#         angle, obb_conf_val = get_window_angle(image, obb_model, obb_conf)
#         print(f"    [obb]  angle={angle:+.1f}°  conf={obb_conf_val:.3f}")
#         image = deskew(image, angle)
#     else:
#         print("    [obb]  skipped (no OBB model supplied)")

#     # ── 3. Orientation selection (0° vs 180°) ────────────────────────────
#     digits, final_image, was_flipped = best_orientation(image, digit_model, conf, iou)

#     # ── 4. Build reading string ───────────────────────────────────────────
#     reading = "".join(d["name"] for d in digits)
#     if not reading:
#         reading_display = "— no digits —"
#     else:
#         reading_display = " ".join(d["name"] for d in digits)

#     # ── 5. Annotate image ────────────────────────────────────────────────
#     annotated = annotate(final_image, digits)
#     banner    = build_banner(annotated.shape[1], reading_display,
#                              angle, obb_conf_val, was_flipped)

#     # Ensure banner width matches (handles edge case after deskew)
#     if banner.shape[1] != annotated.shape[1]:
#         banner = cv2.resize(banner, (annotated.shape[1], banner.shape[0]))

#     final = np.vstack([banner, annotated])

#     # ── 6. Save ───────────────────────────────────────────────────────────
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     cv2.imwrite(str(output_path), final)

#     print(f"  Reading : {reading_display}")
#     print(f"  Saved   : {output_path}")
#     return reading_display


# # ══════════════════════════════════════════════════════════════════════════════
# #  CLI ENTRY POINT
# # ══════════════════════════════════════════════════════════════════════════════

# def main():
#     parser = argparse.ArgumentParser(
#         description="Water-meter digit reader — two-model pipeline (OBB + digit)"
#     )
#     parser.add_argument(
#         "--digit-weights", required=True,
#         help="Path to digit detection model weights  (e.g. digit_model.pt)"
#     )
#     parser.add_argument(
#         "--obb-weights", default=None,
#         help="Path to OBB (oriented bounding box) model weights  (e.g. obb_model.pt). "
#              "Optional — skip to disable auto-rotation."
#     )
#     parser.add_argument(
#         "--source", required=True,
#         help="Path to a single image file or a folder of images"
#     )
#     parser.add_argument(
#         "--output", default=None,
#         help="Output file (single image) or folder (multiple images). "
#              "Default: <source_stem>_reading.<ext>"
#     )
#     parser.add_argument(
#         "--conf", type=float, default=0.25,
#         help="Confidence threshold for the digit model  (default: 0.25)"
#     )
#     parser.add_argument(
#         "--iou", type=float, default=0.35,
#         help="IoU threshold for digit-model NMS  (default: 0.35)"
#     )
#     parser.add_argument(
#         "--obb-conf", type=float, default=0.25,
#         help="Confidence threshold for the OBB model  (default: 0.25)"
#     )
#     args = parser.parse_args()

#     # ── Load models ───────────────────────────────────────────────────────
#     print(f"\nLoading digit model : {args.digit_weights}")
#     digit_model = YOLO(args.digit_weights)

#     obb_model = None
#     if args.obb_weights:
#         print(f"Loading OBB model   : {args.obb_weights}")
#         obb_model = YOLO(args.obb_weights)
#     else:
#         print("OBB model           : not provided — skipping angle correction")

#     # ── Collect images ────────────────────────────────────────────────────
#     VALID = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
#     source = Path(args.source)

#     if source.is_file():
#         images = [source]
#     elif source.is_dir():
#         images = sorted(p for p in source.iterdir() if p.suffix.lower() in VALID)
#     else:
#         print(f"ERROR: source not found: {source}")
#         return

#     if not images:
#         print("No valid images found.")
#         return

#     print(f"\nImages  : {len(images)}")
#     print("-" * 52)

#     # ── Process ───────────────────────────────────────────────────────────
#     for img_path in images:
#         if args.output:
#             out_path = Path(args.output)
#             if len(images) > 1 or out_path.is_dir():
#                 out_path = out_path / img_path.name
#         else:
#             out_path = img_path.parent / f"{img_path.stem}_reading{img_path.suffix}"

#         render_result(
#             image_path  = img_path,
#             digit_model = digit_model,
#             obb_model   = obb_model,
#             conf        = args.conf,
#             iou         = args.iou,
#             obb_conf    = args.obb_conf,
#             output_path = out_path,
#         )

#     print("\nDone.")


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
"""
predict_and_read.py  —  Water-meter digit reader (two-model pipeline)
======================================================================

Pipeline:
  1. Decode image safely          (handles RGBA / grayscale PNG)
  2. OBB model                    → detect tilt angle of the meter window
  3. Deskew                       → rotate to upright (white border, no ghost digits)
  4. Four-orientation sweep       → 0°, 90°, 180°, 270° scored with asymmetric-digit
                                    confidence bonus; OBB tiebreak when scores are close
  5. Digit model                  → detect and sort digits left → right
  6. Annotate                     → draw boxes, labels, and a reading banner

Usage:
    python predict_and_read.py \\
        --digit-weights digit_model.pt \\
        --obb-weights   obb_model.pt \\
        --source        image.jpg

    python predict_and_read.py \\
        --digit-weights digit_model.pt \\
        --obb-weights   obb_model.pt \\
        --source        images_folder/ \\
        --output        results/

    # Skip OBB deskewing (single-model mode, no angle correction):
    python predict_and_read.py --digit-weights digit_model.pt --source image.jpg
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Class map (must match training labels) ────────────────────────────────────
CLASS_NAMES = [
    "meter", "window",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "unknown",
]
DIGIT_CLASS_IDS = set(range(2, 12))   # indices 2–11  →  digits "0"–"9"
#  index 12 ("unknown") intentionally excluded from the reading

# Digits that look distinctly different upright vs. upside-down.
# When the model is confident about one of these it strongly implies
# the orientation is correct (e.g. a 9 upside-down looks like a 6
# but with much lower confidence).
#   cls_id 3  → "1"
#   cls_id 6  → "4"
#   cls_id 8  → "6"
#   cls_id 9  → "7"
#   cls_id 11 → "9"
ASYMMETRIC_CLS = {3, 6, 8, 9, 11}

# ── Visual constants (BGR) ─────────────────────────────────────────────────────
COLOR_BOX    = (0, 215, 80)
COLOR_LABEL  = (0, 215, 80)
COLOR_BANNER = (18, 18, 18)
COLOR_READING= (0, 215, 80)


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE DECODING
# ══════════════════════════════════════════════════════════════════════════════

def decode_image(path: Path) -> np.ndarray:
    """
    Load an image to a solid BGR array.

    PNG files with an alpha channel read via IMREAD_COLOR have their alpha
    dropped and transparent areas become black (0).  On a cropped meter shot
    those dark regions can be mis-detected as meter content.

    Fix: read IMREAD_UNCHANGED, detect 4-channel (BGRA), composite onto white.
    """
    arr   = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    if image.ndim == 3 and image.shape[2] == 4:       # BGRA / RGBA
        bgr   = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        white = np.full_like(bgr, 255.0)
        image = (bgr * alpha + white * (1.0 - alpha)).astype(np.uint8)
        print("    [decode] RGBA → composited onto white background")
    elif image.ndim == 2:                               # grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


# ══════════════════════════════════════════════════════════════════════════════
#  TEMP FILE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _tmp_write(image: np.ndarray) -> str:
    """Write image to a temp JPEG and return the path. Caller must unlink."""
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    cv2.imwrite(path, image)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  OBB: ANGLE DETECTION & DESKEWING
# ══════════════════════════════════════════════════════════════════════════════

def get_window_angle(image: np.ndarray, obb_model: YOLO,
                     obb_conf: float = 0.25) -> tuple[float, float]:
    """
    Run the OBB model to find the tilt angle of the meter window.

    Returns (angle_degrees, best_confidence).
    Returns (0.0, 0.0) when nothing is detected.

    The OBB model is expected to have class 1 = 'window'.
    We take the highest-confidence 'window' detection and read its rotation
    from xywhr[:, 4] (radians).
    """
    tmp = _tmp_write(image)
    try:
        results = obb_model.predict(tmp, conf=obb_conf, verbose=False)[0]
    finally:
        os.unlink(tmp)

    if results.obb is None or len(results.obb) == 0:
        return 0.0, 0.0

    best_conf  = -1.0
    best_angle = 0.0
    for box in results.obb:
        cls_id = int(box.cls[0].cpu())
        conf_v = float(box.conf[0].cpu())
        if cls_id == 1 and conf_v > best_conf:
            best_conf  = conf_v
            xywhr      = box.xywhr[0].cpu().numpy()
            best_angle = math.degrees(float(xywhr[4]))

    return best_angle, best_conf


def deskew(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate image by -angle_deg, expanding canvas to avoid clipping.
    No-op when |angle| < 0.5°.

    Border is white (BORDER_CONSTANT).  BORDER_REFLECT_101 mirrors meter
    content into the new strips → ghost digit detections.
    """
    if abs(angle_deg) < 0.5:
        return image

    h, w   = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M      = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)

    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0

    return cv2.warpAffine(
        image, M, (new_w, new_h),
        flags      = cv2.INTER_LINEAR,
        borderMode = cv2.BORDER_CONSTANT,
        borderValue= (255, 255, 255),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  NMS (cross-class, confidence-ranked)
# ══════════════════════════════════════════════════════════════════════════════

def nms_cross_class(preds: list[dict], iou_thresh: float = 0.40) -> list[dict]:
    """
    Suppress overlapping boxes across all classes, keeping higher-confidence
    detections.  Prevents the same dial digit from being counted twice when
    two classes fire on the same region.
    """
    if not preds:
        return preds

    boxes  = np.array([p["xyxy"] for p in preds], dtype=np.float32)
    scores = np.array([p["conf"] for p in preds], dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        order = order[1:][inter / np.maximum(union, 1e-6) < iou_thresh]

    return [preds[k] for k in keep]


# ══════════════════════════════════════════════════════════════════════════════
#  DIGIT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def read_digits(image: np.ndarray, digit_model: YOLO,
                conf: float = 0.25, iou: float = 0.35) -> list[dict]:
    """
    Run the digit model on *image* and return detections sorted left → right.

    Each detection dict:
        { cls_id, name, conf, xyxy, x_center, y_center }
    """
    tmp = _tmp_write(image)
    try:
        results = digit_model.predict(tmp, conf=conf, iou=iou, verbose=False)[0]
    finally:
        os.unlink(tmp)

    if results.boxes is None:
        return []

    raw = []
    for box in results.boxes:
        cls_id = int(box.cls[0].cpu())
        if cls_id not in DIGIT_CLASS_IDS:
            continue
        conf_v   = float(box.conf[0].cpu())
        xyxy     = box.xyxy[0].cpu().numpy().tolist()
        x_center = (xyxy[0] + xyxy[2]) / 2.0
        y_center = (xyxy[1] + xyxy[3]) / 2.0
        raw.append({
            "cls_id":   cls_id,
            "name":     CLASS_NAMES[cls_id],
            "conf":     round(conf_v, 4),
            "xyxy":     xyxy,
            "x_center": x_center,
            "y_center": y_center,
        })

    raw = nms_cross_class(raw, iou_thresh=0.40)
    raw.sort(key=lambda d: d["x_center"])
    return raw


# ══════════════════════════════════════════════════════════════════════════════
#  ORIENTATION SCORING
# ══════════════════════════════════════════════════════════════════════════════

def orientation_score(digits: list[dict], image_h: int) -> float:
    """
    Score how well-oriented a digit set is.  Higher = better.

    Components:
      +10  per digit detected
           More detections = stronger reading.

      + 3  × total confidence sum
           Rewards both count and per-digit quality.  An upside-down 9
           gets read as a low-confidence 6; flipping the meter lifts its
           confidence noticeably — this term captures that.

      + 4  × confidence, per asymmetric digit  (1, 4, 6, 7, 9)
           These digits look very different upright vs. inverted.  A high-
           confidence 9 strongly implies the orientation is correct; a low-
           confidence 6 that is actually an upside-down 9 loses points here.
           This is the primary fix for the 6-vs-9 misread.

      + 2  × fraction of digits in the upper 60 % of the image
           Meter windows typically sit in the upper portion of the meter face.
           After correct orientation the digit row should not be in the bottom
           third.

      - 8  × normalised Y-spread
           Digits on a single meter window sit on the same horizontal band.
           High vertical scatter implies the model is struggling with the image.
    """
    if not digits:
        return -1.0

    n          = len(digits)
    confs      = np.array([d["conf"] for d in digits])
    y_centers  = np.array([d["y_center"] for d in digits])
    heights    = np.array([d["xyxy"][3] - d["xyxy"][1] for d in digits])
    avg_h      = float(np.mean(heights)) or 1.0
    y_std_norm = float(np.std(y_centers)) / avg_h
    sum_conf   = float(np.sum(confs))

    asym_bonus = sum(
        d["conf"] * 4.0 for d in digits if d["cls_id"] in ASYMMETRIC_CLS
    )

    top_bias = 0.0
    if image_h > 0:
        frac_upper = float(np.mean(y_centers < image_h * 0.60))
        top_bias   = frac_upper * 2.0

    return (
        n         * 10.0
        + sum_conf * 3.0
        + asym_bonus
        + top_bias
        - y_std_norm * 8.0
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ORIENTATION SELECTION  (four-way + OBB tiebreak)
# ══════════════════════════════════════════════════════════════════════════════

def best_orientation(
    image:       np.ndarray,
    digit_model: YOLO,
    conf:        float,
    iou:         float,
    obb_model:   YOLO | None = None,
    obb_conf:    float       = 0.25,
) -> tuple[list[dict], np.ndarray, str]:
    """
    Find the correct reading orientation using a two-pass strategy.

    WHY FOUR ORIENTATIONS?
    ──────────────────────
    The OBB deskew removes the measured tilt angle, but the OBB angle is only
    accurate modulo 180° (a symmetric window looks the same upside-down).
    A meter mounted at 160° gets deskewed by -160°  →  tilt gone, content
    upside-down.  Trying all four cardinal angles catches:
      • 180° flip  — the common upside-down case (your 6-vs-9 failure)
      • 90°/270°   — portrait-mounted meters

    PASS 1 — digit-model sweep
    ──────────────────────────
    Run read_digits() at 0°, 90°, 180°, 270° and score each with
    orientation_score().  The asymmetric-digit confidence bonus usually
    breaks ties decisively (an upside-down 9 scores far worse than an
    upright 9 even when both are detected).

    PASS 2 — OBB tiebreak  (only when top-2 scores are within TIEBREAK_GAP)
    ─────────────────────────────────────────────────────────────────────────
    When the digit model is uncertain (all digits are symmetric like 00000000),
    run the OBB model on the two closest candidates.  The window that appears
    most horizontal (angle closest to 0°) wins — the OBB can reliably tell
    landscape from portrait even when it can't tell 0° from 180°.

    Returns: (digits, image_used, orientation_label)
      orientation_label ∈ {"0°", "90°", "180°", "270°"}
    """
    TIEBREAK_GAP = 8.0

    h = image.shape[0]
    candidates = [
        ("0°",   image),
        ("90°",  cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
        ("180°", cv2.rotate(image, cv2.ROTATE_180)),
        ("270°", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    scored: list[tuple[str, np.ndarray, list[dict], float]] = []

    for label, candidate in candidates:
        digits   = read_digits(candidate, digit_model, conf, iou)
        score    = orientation_score(digits, h)
        sum_conf = sum(d["conf"] for d in digits)
        asym_n   = sum(1 for d in digits if d["cls_id"] in ASYMMETRIC_CLS)
        print(
            f"    [orient] {label:<5}  score={score:7.2f}  "
            f"digits={len(digits)}  conf_sum={sum_conf:.3f}  asym={asym_n}"
        )
        scored.append((label, candidate, digits, score))

    scored.sort(key=lambda r: r[3], reverse=True)
    best   = scored[0]
    second = scored[1]

    # ── Pass 2: OBB tiebreak ─────────────────────────────────────────────
    gap = best[3] - second[3]
    if obb_model is not None and gap < TIEBREAK_GAP and best[3] > -1.0:
        print(
            f"    [orient] scores close ({best[3]:.2f} vs {second[3]:.2f}, "
            f"gap={gap:.2f}) — OBB tiebreak between {best[0]} and {second[0]}"
        )
        top_two      = [best, second]
        obb_winner   = None
        best_straight = 999.0

        for label, candidate, digits, score in top_two:
            angle, oconf = get_window_angle(candidate, obb_model, obb_conf)
            # Straightness: window angle closest to 0° = most upright
            straightness = abs(angle) if oconf > 0.0 else 45.0
            print(
                f"    [tiebreak] {label:<5}  window_angle={angle:+.1f}°  "
                f"obb_conf={oconf:.3f}  straightness={straightness:.1f}"
            )
            if straightness < best_straight:
                best_straight = straightness
                obb_winner    = (label, candidate, digits, score)

        if obb_winner is not None:
            best = obb_winner
            print(f"    [tiebreak] → chose {best[0]}")
    else:
        reason = (
            f"gap={gap:.2f} ≥ {TIEBREAK_GAP}" if gap >= TIEBREAK_GAP
            else "no OBB model"
        )
        print(f"    [orient] tiebreak skipped ({reason})")

    print(f"    [orient] → final: {best[0]}  (score={best[3]:.2f})")
    return best[2], best[1], best[0]


# ══════════════════════════════════════════════════════════════════════════════
#  ANNOTATION
# ══════════════════════════════════════════════════════════════════════════════

def annotate(image: np.ndarray, digits: list[dict]) -> np.ndarray:
    """Draw green bounding boxes and digit+confidence labels onto *image*."""
    canvas = image.copy()
    font   = cv2.FONT_HERSHEY_SIMPLEX

    for d in digits:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        label = f"{d['name']} {d['conf']:.2f}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_BOX, 2)

        (tw, th), baseline = cv2.getTextSize(label, font, 0.52, 1)
        pad = 4
        ly  = max(th + baseline + pad, y1)
        cv2.rectangle(canvas,
                      (x1, ly - th - baseline - pad),
                      (x1 + tw + pad * 2, ly),
                      COLOR_LABEL, -1)
        cv2.putText(canvas, label,
                    (x1 + pad, ly - baseline - pad // 2),
                    font, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas


def build_banner(
    width:             int,
    reading:           str,
    angle:             float,
    obb_conf:          float,
    orientation_label: str,
) -> np.ndarray:
    """Build the top status banner showing the reading and pipeline metadata."""
    h      = 58
    banner = np.full((h, width, 3), COLOR_BANNER, dtype=np.uint8)
    font   = cv2.FONT_HERSHEY_SIMPLEX

    display = reading if reading else "—"
    cv2.putText(banner, f"METER READING:  {display}",
                (16, 38), font, 0.95, COLOR_READING, 2, cv2.LINE_AA)

    parts = []
    if obb_conf > 0:
        parts.append(f"deskew {angle:+.1f}°  (obb conf {obb_conf:.2f})")
    else:
        parts.append("no OBB rotation detected")
    if orientation_label != "0°":
        parts.append(f"orientation fix: {orientation_label}")

    info = "  |  ".join(parts)
    (tw, _), _ = cv2.getTextSize(info, font, 0.42, 1)
    cv2.putText(banner, info, (width - tw - 16, 38),
                font, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

    return banner


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RENDER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def render_result(
    image_path:  Path,
    digit_model: YOLO,
    obb_model:   YOLO | None,
    conf:        float,
    iou:         float,
    obb_conf:    float,
    output_path: Path,
) -> str:
    """
    Full pipeline for a single image:
      decode → OBB deskew → four-orientation sweep → digit detect → annotate → save
    """
    print(f"\n  Image : {image_path.name}")

    # ── 1. Decode (RGBA / grayscale safe) ────────────────────────────────
    try:
        image = decode_image(image_path)
    except ValueError as e:
        print(f"  ERROR: {e}")
        return ""

    # ── 2. OBB deskew ────────────────────────────────────────────────────
    angle        = 0.0
    obb_conf_val = 0.0
    if obb_model is not None:
        angle, obb_conf_val = get_window_angle(image, obb_model, obb_conf)
        print(f"    [obb]  angle={angle:+.1f}°  conf={obb_conf_val:.3f}")
        image = deskew(image, angle)
    else:
        print("    [obb]  skipped (no OBB model supplied)")

    # ── 3. Four-orientation sweep + OBB tiebreak ─────────────────────────
    digits, final_image, orientation_label = best_orientation(
        image, digit_model, conf, iou,
        obb_model=obb_model, obb_conf=obb_conf,
    )

    # ── 4. Build reading string ───────────────────────────────────────────
    reading         = "".join(d["name"] for d in digits)
    reading_display = " ".join(d["name"] for d in digits) if reading else "— no digits —"

    # ── 5. Annotate ───────────────────────────────────────────────────────
    annotated = annotate(final_image, digits)
    banner    = build_banner(
        annotated.shape[1], reading_display,
        angle, obb_conf_val, orientation_label,
    )
    if banner.shape[1] != annotated.shape[1]:
        banner = cv2.resize(banner, (annotated.shape[1], banner.shape[0]))

    final = np.vstack([banner, annotated])

    # ── 6. Save ───────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), final)

    print(f"  Reading     : {reading_display}")
    print(f"  Orientation : {orientation_label}")
    print(f"  Saved       : {output_path}")
    return reading_display


# ══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Water-meter digit reader — two-model pipeline (OBB + digit)"
    )
    parser.add_argument(
        "--digit-weights", required=True,
        help="Path to digit detection model weights  (e.g. digit_model.pt)"
    )
    parser.add_argument(
        "--obb-weights", default=None,
        help="Path to OBB model weights  (e.g. obb_model.pt). "
             "Optional — omit to disable auto-rotation."
    )
    parser.add_argument(
        "--source", required=True,
        help="Path to a single image file or a folder of images"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file (single image) or folder (multiple images). "
             "Default: <source_stem>_reading.<ext>"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold for the digit model  (default: 0.25)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.35,
        help="IoU threshold for digit-model NMS  (default: 0.35)"
    )
    parser.add_argument(
        "--obb-conf", type=float, default=0.25,
        help="Confidence threshold for the OBB model  (default: 0.25)"
    )
    args = parser.parse_args()

    # ── Load models ───────────────────────────────────────────────────────
    print(f"\nLoading digit model : {args.digit_weights}")
    digit_model = YOLO(args.digit_weights)

    obb_model = None
    if args.obb_weights:
        print(f"Loading OBB model   : {args.obb_weights}")
        obb_model = YOLO(args.obb_weights)
    else:
        print("OBB model           : not provided — skipping angle correction")

    # ── Collect images ────────────────────────────────────────────────────
    VALID  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    source = Path(args.source)

    if source.is_file():
        images = [source]
    elif source.is_dir():
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in VALID)
    else:
        print(f"ERROR: source not found: {source}")
        return

    if not images:
        print("No valid images found.")
        return

    print(f"\nImages  : {len(images)}")
    print("-" * 56)

    # ── Process ───────────────────────────────────────────────────────────
    for img_path in images:
        if args.output:
            out_path = Path(args.output)
            if len(images) > 1 or out_path.is_dir():
                out_path = out_path / img_path.name
        else:
            out_path = img_path.parent / f"{img_path.stem}_reading{img_path.suffix}"

        render_result(
            image_path  = img_path,
            digit_model = digit_model,
            obb_model   = obb_model,
            conf        = args.conf,
            iou         = args.iou,
            obb_conf    = args.obb_conf,
            output_path = out_path,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()