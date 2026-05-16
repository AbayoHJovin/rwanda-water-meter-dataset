#!/usr/bin/env python3
"""
export_confused_images.py

Export images where the model made mistakes (confused/misclassified).
This helps you identify which images need better labels or more training data.

Usage:
  python export_confused_images.py --weights best.pt --data data.yaml --output confused_images/

What it does:
  1. Runs validation on your val/test dataset
  2. Compares predictions vs ground truth
  3. Exports images where:
     - Model missed boxes (False Negatives)
     - Model drew wrong boxes (False Positives)
     - Model got the class wrong (Misclassification)
  4. Saves them organized by error type and class

Output structure:
  confused_images/
    ├── false_negatives/      # missed boxes
    │   ├── digit_1/
    │   ├── digit_6/
    │   └── ...
    ├── false_positives/      # wrong boxes
    │   └── ...
    ├── misclassified/        # wrong class
    │   ├── predicted_1_actual_7/
    │   ├── predicted_6_actual_9/
    │   └── ...
    └── summary.txt           # statistics
"""

import os
import shutil
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


def iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def parse_label_file(label_path, img_w, img_h):
    """Parse YOLO format label file and return list of boxes."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            cls = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            
            # Convert to pixel coords
            x1 = (x_center - width / 2) * img_w
            y1 = (y_center - height / 2) * img_h
            x2 = (x_center + width / 2) * img_w
            y2 = (y_center + height / 2) * img_h
            
            boxes.append({
                'class': cls,
                'box': [x1, y1, x2, y2],
                'matched': False
            })
    
    return boxes


def draw_boxes_on_image(image, gt_boxes, pred_boxes, matched_gt, matched_pred):
    """Draw ground truth (green) and predictions (red/yellow) on image with clean labels."""
    img_h, img_w = image.shape[:2]
    
    # Create a taller canvas to add legend at the bottom
    legend_height = 200
    canvas = np.ones((img_h + legend_height, img_w, 3), dtype=np.uint8) * 240
    canvas[:img_h, :] = image.copy()
    
    # Number each box with a small circle
    box_counter = 1
    gt_labels = []
    pred_labels = []
    
    # Draw ground truth boxes in GREEN with numbered circles
    for i, gt in enumerate(gt_boxes):
        x1, y1, x2, y2 = map(int, gt['box'])
        is_matched = matched_gt[i]
        
        # Box color: green if matched, orange if missed
        box_color = (0, 200, 0) if is_matched else (0, 140, 255)
        thickness = 2 if is_matched else 3
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)
        
        # Draw number circle in top-left corner of box
        circle_center = (x1 + 15, y1 + 15)
        cv2.circle(canvas, circle_center, 12, box_color, -1)
        cv2.putText(canvas, str(box_counter), (x1 + 10, y1 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Store label info for legend
        status = "✓" if is_matched else "MISSED"
        gt_labels.append(f"#{box_counter} GT: {CLASS_NAMES[gt['class']]:8s} {status}")
        box_counter += 1
    
    # Draw predictions in YELLOW (correct) or RED (wrong) with numbered circles
    for i, pred in enumerate(pred_boxes):
        x1, y1, x2, y2 = map(int, pred['box'])
        is_matched = matched_pred[i]
        conf = pred.get('conf', 0)
        
        # Box color: yellow if matched, red if false positive
        box_color = (0, 200, 200) if is_matched else (0, 0, 255)
        thickness = 1 if is_matched else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)
        
        # Draw number circle in bottom-right corner of box
        circle_center = (x2 - 15, y2 - 15)
        cv2.circle(canvas, circle_center, 12, box_color, -1)
        cv2.putText(canvas, str(box_counter), (x2 - 20, y2 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Store label info for legend
        status = "✓" if is_matched else "FALSE+"
        pred_labels.append(f"#{box_counter} Pred: {CLASS_NAMES[pred['class']]:8s} ({conf:.2f}) {status}")
        box_counter += 1
    
    # Draw legend panel at the bottom
    legend_y = img_h + 10
    
    # Title
    cv2.putText(canvas, "LEGEND:", (10, legend_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    legend_y += 25
    
    # Color key
    cv2.rectangle(canvas, (10, legend_y), (30, legend_y + 15), (0, 200, 0), -1)
    cv2.putText(canvas, "= Ground Truth (Matched)", (35, legend_y + 12), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    
    cv2.rectangle(canvas, (10, legend_y + 20), (30, legend_y + 35), (0, 140, 255), -1)
    cv2.putText(canvas, "= Ground Truth (MISSED)", (35, legend_y + 32), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    
    cv2.rectangle(canvas, (10, legend_y + 40), (30, legend_y + 55), (0, 200, 200), -1)
    cv2.putText(canvas, "= Prediction (Correct)", (35, legend_y + 52), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    
    cv2.rectangle(canvas, (10, legend_y + 60), (30, legend_y + 75), (0, 0, 255), -1)
    cv2.putText(canvas, "= Prediction (Wrong)", (35, legend_y + 72), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    
    # Box details in two columns
    details_y = legend_y + 95
    col1_x = 10
    col2_x = img_w // 2 + 10
    
    # Ground truth column
    cv2.putText(canvas, "GROUND TRUTH:", (col1_x, details_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 2)
    details_y += 20
    for label in gt_labels:
        cv2.putText(canvas, label, (col1_x, details_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        details_y += 18
    
    # Predictions column
    details_y = legend_y + 95
    cv2.putText(canvas, "PREDICTIONS:", (col2_x, details_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 100), 2)
    details_y += 20
    for label in pred_labels:
        cv2.putText(canvas, label, (col2_x, details_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        details_y += 18
    
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Export confused/misclassified images")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model weights (best.pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--output", type=str, default="confused_images", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for matching")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], 
                        help="Which split to analyze")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.weights}...")
    model = YOLO(args.weights)
    
    # Load data.yaml to get image paths
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    
    data_root = Path(args.data).parent
    image_dir = data_root / "images" / args.split
    label_dir = data_root / "labels" / args.split
    
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        return
    
    print(f"Analyzing images from: {image_dir}")
    print(f"Using labels from: {label_dir}")
    
    # Create output directories
    output_dir = Path(args.output)
    fn_dir = output_dir / "false_negatives"      # model missed these
    fp_dir = output_dir / "false_positives"      # model wrongly detected these
    misclass_dir = output_dir / "misclassified"  # model got class wrong
    
    for d in [fn_dir, fp_dir, misclass_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = defaultdict(int)
    error_details = defaultdict(list)
    
    # Process each image
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    print(f"Found {len(image_files)} images to analyze...")
    
    for img_idx, img_path in enumerate(image_files):
        print(f"[{img_idx+1}/{len(image_files)}] Processing {img_path.name}...", end='\r')
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        img_h, img_w = image.shape[:2]
        
        # Get ground truth labels
        label_path = label_dir / f"{img_path.stem}.txt"
        gt_boxes = parse_label_file(str(label_path), img_w, img_h)
        
        # Get predictions
        results = model.predict(str(img_path), conf=args.conf, verbose=False)[0]
        pred_boxes = []
        
        if results.boxes is not None:
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                pred_boxes.append({
                    'class': cls,
                    'box': xyxy.tolist(),
                    'conf': conf,
                    'matched': False
                })
        
        # Match predictions to ground truth
        matched_gt = [False] * len(gt_boxes)
        matched_pred = [False] * len(pred_boxes)
        
        for i, gt in enumerate(gt_boxes):
            best_iou = 0
            best_pred_idx = -1
            
            for j, pred in enumerate(pred_boxes):
                if matched_pred[j]:
                    continue
                
                iou_val = iou(gt['box'], pred['box'])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_pred_idx = j
            
            if best_iou >= args.iou_thresh:
                matched_gt[i] = True
                if best_pred_idx >= 0:
                    matched_pred[best_pred_idx] = True
                    
                    # Check if class is correct
                    if gt['class'] != pred_boxes[best_pred_idx]['class']:
                        stats['misclassified'] += 1
                        pred_class = pred_boxes[best_pred_idx]['class']
                        gt_class = gt['class']
                        
                        error_key = f"predicted_{CLASS_NAMES[pred_class]}_actual_{CLASS_NAMES[gt_class]}"
                        error_details[error_key].append(img_path.name)
                        
                        # Save misclassified image
                        save_dir = misclass_dir / error_key
                        save_dir.mkdir(exist_ok=True)
                        
                        annotated = draw_boxes_on_image(image, gt_boxes, pred_boxes, matched_gt, matched_pred)
                        cv2.imwrite(str(save_dir / img_path.name), annotated)
        
        # Find false negatives (missed boxes)
        for i, is_matched in enumerate(matched_gt):
            if not is_matched:
                stats['false_negatives'] += 1
                cls = gt_boxes[i]['class']
                class_name = CLASS_NAMES[cls]
                
                error_details[f"fn_{class_name}"].append(img_path.name)
                
                # Save false negative image
                save_dir = fn_dir / class_name
                save_dir.mkdir(exist_ok=True)
                
                annotated = draw_boxes_on_image(image, gt_boxes, pred_boxes, matched_gt, matched_pred)
                cv2.imwrite(str(save_dir / img_path.name), annotated)
        
        # Find false positives (wrong boxes)
        for i, is_matched in enumerate(matched_pred):
            if not is_matched:
                stats['false_positives'] += 1
                cls = pred_boxes[i]['class']
                class_name = CLASS_NAMES[cls]
                
                error_details[f"fp_{class_name}"].append(img_path.name)
                
                # Save false positive image
                save_dir = fp_dir / class_name
                save_dir.mkdir(exist_ok=True)
                
                annotated = draw_boxes_on_image(image, gt_boxes, pred_boxes, matched_gt, matched_pred)
                cv2.imwrite(str(save_dir / img_path.name), annotated)
    
    print("\n\nAnalysis complete!")
    print(f"\nTotal errors found:")
    print(f"  False Negatives (missed):  {stats['false_negatives']}")
    print(f"  False Positives (wrong):   {stats['false_positives']}")
    print(f"  Misclassified:             {stats['misclassified']}")
    
    # Write summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("CONFUSED IMAGES ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Model: {args.weights}\n")
        f.write(f"Dataset split: {args.split}\n")
        f.write(f"Confidence threshold: {args.conf}\n")
        f.write(f"IoU threshold: {args.iou_thresh}\n\n")
        
        f.write(f"Total Errors:\n")
        f.write(f"  False Negatives: {stats['false_negatives']}\n")
        f.write(f"  False Positives: {stats['false_positives']}\n")
        f.write(f"  Misclassified:   {stats['misclassified']}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("FALSE NEGATIVES (Model Missed These)\n")
        f.write("=" * 60 + "\n")
        for key in sorted(error_details.keys()):
            if key.startswith("fn_"):
                class_name = key[3:]
                count = len(error_details[key])
                f.write(f"\n{class_name}: {count} missed\n")
                for img_name in error_details[key][:10]:  # show first 10
                    f.write(f"  - {img_name}\n")
                if count > 10:
                    f.write(f"  ... and {count - 10} more\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("MISCLASSIFICATIONS (Wrong Class)\n")
        f.write("=" * 60 + "\n")
        for key in sorted(error_details.keys()):
            if key.startswith("predicted_"):
                count = len(error_details[key])
                f.write(f"\n{key}: {count} cases\n")
                for img_name in error_details[key][:10]:
                    f.write(f"  - {img_name}\n")
                if count > 10:
                    f.write(f"  ... and {count - 10} more\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Images saved to: {output_dir.absolute()}\n")
        f.write("=" * 60 + "\n")
    
    print(f"\nFull summary saved to: {summary_path}")
    print(f"Confused images saved to: {output_dir.absolute()}")
    print("\nReview these images to:")
    print("  1. Fix incorrect labels")
    print("  2. Identify missing training data for weak classes")
    print("  3. Understand model confusion patterns (e.g. 6 vs 9, 1 vs 7)")


if __name__ == "__main__":
    main()