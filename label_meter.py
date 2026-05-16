#!/usr/bin/env python3
"""
01_label_images.py

Water Meter YOLO Labeler
- Regular YOLO boxes for all classes except window
- OBB (Oriented Bounding Box) for window class (class_id == 1)
  Saved as:  1 x1 y1 x2 y2 x3 y3 x4 y4  (normalised image coords, 4 corners)
  Loaded back and displayed as a rotated rectangle
- At inference: compute angle from the 4 corners → rotate crop horizontal → read digits
"""

import math
import os
import sys
import shutil
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QAction, QColor, QImage, QKeySequence, QPainter, QPen,
    QPixmap, QBrush, QWheelEvent, QPolygonF, QTransform,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QInputDialog,
    QDialog,
    QSlider,
    QFrame,
    QScrollArea,
)


CLASS_NAMES = [
    "meter", "window",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "unknown",
]

CLASS_COLORS = {
    0: QColor(0, 255, 0),
    1: QColor(255, 0, 0),       # window → red
    2: QColor(255, 255, 0),
    3: QColor(255, 165, 0),
    4: QColor(0, 255, 255),
    5: QColor(255, 0, 255),
    6: QColor(128, 255, 0),
    7: QColor(0, 128, 255),
    8: QColor(255, 128, 128),
    9: QColor(128, 0, 255),
    10: QColor(200, 200, 255),
    11: QColor(255, 200, 200),
    12: QColor(180, 180, 180),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

HANDLE_SIZE = 10
MIN_BOX_SIZE = 5

ZOOM_MIN = 0.1
ZOOM_MAX = 10.0
ZOOM_STEP = 0.15

WINDOW_CLASS_ID = 1   # the only class that uses OBB

# How many pixels from the box centre the rotation handle sits (in widget pixels)
ROTATE_HANDLE_DIST = 28


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_image_files(folder: str) -> List[str]:
    return sorted([
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ])


def unique_destination_path(destination_path: str) -> str:
    if not os.path.exists(destination_path):
        return destination_path
    folder = os.path.dirname(destination_path)
    filename = os.path.basename(destination_path)
    stem, ext = os.path.splitext(filename)
    counter = 1
    while True:
        new_path = os.path.join(folder, f"{stem}_removed_{counter}{ext}")
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def rotate_image_keep_size_crop_edges(image, angle_degrees: float):
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def crop_image(image, x1, y1, x2, y2):
    h, w = image.shape[:2]
    x_min = int(max(0, min(x1, x2)))
    y_min = int(max(0, min(y1, y2)))
    x_max = int(min(w, max(x1, x2)))
    y_max = int(min(h, max(y1, y2)))
    if x_max <= x_min or y_max <= y_min:
        return None
    return image[y_min:y_max, x_min:x_max]


def safe_message(parent, title: str, message: str, button_text: str = "OK") -> None:
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setModal(True)
    dialog.resize(520, 170)
    label = QLabel(message)
    label.setWordWrap(True)
    ok_btn = QPushButton(button_text)
    ok_btn.clicked.connect(dialog.accept)
    buttons = QHBoxLayout()
    buttons.addStretch()
    buttons.addWidget(ok_btn)
    layout = QVBoxLayout()
    layout.addWidget(label)
    layout.addStretch()
    layout.addLayout(buttons)
    dialog.setLayout(layout)
    dialog.exec()


def safe_confirm(parent, title: str, message: str,
                 yes_text: str = "Yes", no_text: str = "Cancel") -> bool:
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setModal(True)
    dialog.resize(560, 200)
    label = QLabel(message)
    label.setWordWrap(True)
    yes_btn = QPushButton(yes_text)
    no_btn = QPushButton(no_text)
    yes_btn.clicked.connect(dialog.accept)
    no_btn.clicked.connect(dialog.reject)
    buttons = QHBoxLayout()
    buttons.addStretch()
    buttons.addWidget(yes_btn)
    buttons.addWidget(no_btn)
    layout = QVBoxLayout()
    layout.addWidget(label)
    layout.addStretch()
    layout.addLayout(buttons)
    dialog.setLayout(layout)
    return dialog.exec() == QDialog.Accepted


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Box:
    """Regular axis-aligned bounding box (all classes except window)."""
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    def copy(self) -> "Box":
        return Box(self.class_id, self.x1, self.y1, self.x2, self.y2)

    def normalized(self, img_w, img_h):
        x_min = min(self.x1, self.x2)
        y_min = min(self.y1, self.y2)
        x_max = max(self.x1, self.x2)
        y_max = max(self.y1, self.y2)
        x_center = ((x_min + x_max) / 2.0) / img_w
        y_center = ((y_min + y_max) / 2.0) / img_h
        width = (x_max - x_min) / img_w
        height = (y_max - y_min) / img_h
        return x_center, y_center, width, height

    @staticmethod
    def from_normalized(class_id, x_center, y_center, width, height,
                        img_w, img_h) -> "Box":
        bw = width * img_w
        bh = height * img_h
        cx = x_center * img_w
        cy = y_center * img_h
        return Box(class_id, cx - bw / 2.0, cy - bh / 2.0,
                   cx + bw / 2.0, cy + bh / 2.0)

    def rect(self):
        return (min(self.x1, self.x2), min(self.y1, self.y2),
                max(self.x1, self.x2), max(self.y1, self.y2))

    def contains(self, x, y) -> bool:
        x_min, y_min, x_max, y_max = self.rect()
        return x_min <= x <= x_max and y_min <= y <= y_max

    def width(self):
        return abs(self.x2 - self.x1)

    def height(self):
        return abs(self.y2 - self.y1)

    def is_too_small(self, min_size=MIN_BOX_SIZE) -> bool:
        return self.width() < min_size or self.height() < min_size

    def move(self, dx, dy, img_w, img_h) -> None:
        x_min, y_min, x_max, y_max = self.rect()
        box_w = x_max - x_min
        box_h = y_max - y_min
        new_x_min = max(0, min(img_w - box_w, x_min + dx))
        new_y_min = max(0, min(img_h - box_h, y_min + dy))
        self.x1 = new_x_min
        self.y1 = new_y_min
        self.x2 = new_x_min + box_w
        self.y2 = new_y_min + box_h

    def clamp(self, img_w, img_h) -> None:
        self.x1 = max(0, min(img_w - 1, self.x1))
        self.y1 = max(0, min(img_h - 1, self.y1))
        self.x2 = max(0, min(img_w - 1, self.x2))
        self.y2 = max(0, min(img_h - 1, self.y2))


@dataclass
class OBBBox:
    """
    Oriented bounding box for the window class.

    Internally stored as:
        cx, cy   — centre in image pixels
        w, h     — full width and height in image pixels
                   (w = long side = along the digit strip)
        angle    — rotation in degrees, CCW from the positive-x axis
                   (0° means the long side is horizontal/pointing right)

    On disk (YOLO-OBB format, normalised):
        1  x1 y1  x2 y2  x3 y3  x4 y4
    Corner order: top-left → top-right → bottom-right → bottom-left
    relative to the *unrotated* rectangle, then rotated by `angle`.
    """
    class_id: int          # always WINDOW_CLASS_ID
    cx: float
    cy: float
    w: float               # long dimension (along digit row)
    h: float               # short dimension
    angle: float = 0.0     # degrees, CCW

    def copy(self) -> "OBBBox":
        return OBBBox(self.class_id, self.cx, self.cy,
                      self.w, self.h, self.angle)

    # ---- geometry --------------------------------------------------------

    def corners_image(self) -> List[Tuple[float, float]]:
        """Return the 4 corners in image pixel space."""
        hw = self.w / 2.0
        hh = self.h / 2.0
        rad = math.radians(self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        # local corners (unrotated): TL, TR, BR, BL
        local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        corners = []
        for lx, ly in local:
            rx = lx * cos_a - ly * sin_a + self.cx
            ry = lx * sin_a + ly * cos_a + self.cy
            corners.append((rx, ry))
        return corners

    def corners_widget(self, scale, offset_x, offset_y):
        return [
            (x * scale + offset_x, y * scale + offset_y)
            for x, y in self.corners_image()
        ]

    def rotate_handle_widget(self, scale, offset_x, offset_y):
        """Widget coords of the rotation handle (above the top-centre)."""
        rad = math.radians(self.angle)
        # The "up" direction perpendicular to the long axis, outward from top edge
        perp_x = -math.sin(rad)
        perp_y =  math.cos(rad)
        # top-centre of the box in image space
        hw = self.w / 2.0
        hh = self.h / 2.0
        tx = self.cx + (-hh) * math.sin(rad)  # top-centre image x
        ty = self.cy + (-hh) * math.cos(rad)  # top-centre image y  -- wait, let me redo

        # Better: centre of the top edge
        c = self.corners_image()
        top_cx = (c[0][0] + c[1][0]) / 2.0
        top_cy = (c[0][1] + c[1][1]) / 2.0

        # Move outward by ROTATE_HANDLE_DIST / scale (in image pixels)
        dist_img = ROTATE_HANDLE_DIST / scale
        hx = top_cx + perp_x * dist_img
        hy = top_cy + perp_y * dist_img

        return hx * scale + offset_x, hy * scale + offset_y

    def contains_point(self, x, y) -> bool:
        """Test if image-pixel point (x,y) is inside the rotated rect."""
        rad = math.radians(-self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        dx = x - self.cx
        dy = y - self.cy
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        return abs(lx) <= self.w / 2.0 and abs(ly) <= self.h / 2.0

    def is_too_small(self, min_size=MIN_BOX_SIZE) -> bool:
        return self.w < min_size or self.h < min_size

    def move(self, dx, dy) -> None:
        self.cx += dx
        self.cy += dy

    # ---- serialisation ---------------------------------------------------

    def to_yolo_line(self, img_w, img_h) -> str:
        corners = self.corners_image()
        parts = [str(self.class_id)]
        for cx_pt, cy_pt in corners:
            parts.append(f"{cx_pt / img_w:.6f}")
            parts.append(f"{cy_pt / img_h:.6f}")
        return " ".join(parts)

    @staticmethod
    def from_yolo_line(parts, img_w, img_h) -> "OBBBox":
        """Parse:  class_id x1 y1 x2 y2 x3 y3 x4 y4  (9 values)."""
        class_id = int(parts[0])
        coords = [float(v) for v in parts[1:9]]
        pts = [(coords[i] * img_w, coords[i + 1] * img_h)
               for i in range(0, 8, 2)]

        # Recover cx, cy, w, h, angle from the 4 corners
        cx = sum(p[0] for p in pts) / 4.0
        cy = sum(p[1] for p in pts) / 4.0

        # Width = distance TL→TR, height = distance TL→BL
        w = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
        h = math.hypot(pts[3][0] - pts[0][0], pts[3][1] - pts[0][1])

        # Angle from TL→TR edge
        angle = math.degrees(math.atan2(pts[1][1] - pts[0][1],
                                         pts[1][0] - pts[0][0]))
        return OBBBox(class_id, cx, cy, w, h, angle)


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------

class ImageCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.main_window = None
        self.base_pixmap = None
        self.display_pixmap = None

        self.fit_scale = 1.0
        self.zoom_level = 1.0
        self.scale = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.offset_x = 0
        self.offset_y = 0

        self.mode = "idle"
        self.crop_mode_enabled = False
        self.resize_handle = None

        self.start_widget_point = None
        self.current_widget_point = None
        self.last_img_point = None
        self.original_box_before_edit = None

        # OBB-specific interaction state
        self._obb_rotating = False          # True while dragging the rotate handle
        self._obb_rotate_start_angle = 0.0  # box.angle when drag started
        self._obb_rotate_mouse_angle = 0.0  # mouse angle from centre when drag started

        self._pan_start = None
        self._pan_start_offset = None

    def set_main_window(self, window):
        self.main_window = window

    def set_image(self, qpixmap: QPixmap):
        self.base_pixmap = qpixmap
        self.zoom_level = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.update_scaled_pixmap()

    def clear_image(self):
        self.base_pixmap = None
        self.display_pixmap = None
        self.clear()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_scaled_pixmap()

    # ------------------------------------------------------------------
    def update_scaled_pixmap(self):
        if self.base_pixmap is None:
            self.clear()
            return

        label_w = max(1, self.width())
        label_h = max(1, self.height())
        img_w = self.base_pixmap.width()
        img_h = self.base_pixmap.height()

        scale_x = label_w / img_w
        scale_y = label_h / img_h
        self.fit_scale = min(scale_x, scale_y)
        self.scale = self.fit_scale * self.zoom_level

        disp_w = int(img_w * self.scale)
        disp_h = int(img_h * self.scale)

        center_x = label_w / 2.0
        center_y = label_h / 2.0
        self.offset_x = int(center_x - disp_w / 2.0 + self.pan_offset_x)
        self.offset_y = int(center_y - disp_h / 2.0 + self.pan_offset_y)

        self.display_pixmap = self.base_pixmap.scaled(
            disp_w, disp_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        canvas = QPixmap(label_w, label_h)
        canvas.fill(Qt.black)

        painter = QPainter(canvas)
        painter.drawPixmap(self.offset_x, self.offset_y, self.display_pixmap)

        if self.main_window is not None:
            self.main_window.draw_boxes(painter, self.scale,
                                        self.offset_x, self.offset_y)

        # Draft rectangle while drawing
        if self.mode in ("drawing", "crop") and \
                self.start_widget_point and self.current_widget_point:
            if self.mode == "crop":
                draft_color = QColor(0, 255, 255)
            else:
                draft_color = CLASS_COLORS.get(
                    self.main_window.current_class_id, QColor(255, 255, 255))
            painter.setPen(QPen(draft_color, 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            rect = QRectF(self.start_widget_point,
                          self.current_widget_point).normalized()
            painter.drawRect(rect)

        painter.end()
        self.setPixmap(canvas)

    # ------------------------------------------------------------------
    def widget_to_image(self, point: QPointF):
        if self.base_pixmap is None or self.display_pixmap is None:
            return None
        x = point.x() - self.offset_x
        y = point.y() - self.offset_y
        if x < 0 or y < 0 or \
                x > self.display_pixmap.width() or \
                y > self.display_pixmap.height():
            return None
        return x / self.scale, y / self.scale

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------
    def apply_zoom(self, factor: float, anchor_widget_pos: QPointF = None):
        old_zoom = self.zoom_level
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom_level * factor))
        if abs(new_zoom - old_zoom) < 1e-9:
            return

        if anchor_widget_pos is not None:
            ax = anchor_widget_pos.x()
            ay = anchor_widget_pos.y()
            label_w = self.width()
            label_h = self.height()
            img_w = self.base_pixmap.width()
            img_h = self.base_pixmap.height()
            old_scale = self.fit_scale * old_zoom
            old_disp_w = img_w * old_scale
            old_disp_h = img_h * old_scale
            old_ox = (label_w - old_disp_w) / 2.0 + self.pan_offset_x
            old_oy = (label_h - old_disp_h) / 2.0 + self.pan_offset_y
            img_x = (ax - old_ox) / old_scale
            img_y = (ay - old_oy) / old_scale
            new_scale = self.fit_scale * new_zoom
            new_disp_w = img_w * new_scale
            new_disp_h = img_h * new_scale
            self.pan_offset_x = ax - (label_w - new_disp_w) / 2.0 - img_x * new_scale
            self.pan_offset_y = ay - (label_h - new_disp_h) / 2.0 - img_y * new_scale

        self.zoom_level = new_zoom
        self.update_scaled_pixmap()
        if self.main_window:
            self.main_window.update_zoom_indicator()

    def reset_zoom(self):
        self.zoom_level = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.update_scaled_pixmap()
        if self.main_window:
            self.main_window.update_zoom_indicator()

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent):
        if self.base_pixmap is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return

        mods = event.modifiers()
        # Shift+Scroll → rotate selected OBB box
        if mods & Qt.ShiftModifier:
            if self.main_window:
                step = 0.5 if delta > 0 else -0.5
                self.main_window.rotate_selected_obb(step)
            event.accept()
            return

        factor = (1.0 + ZOOM_STEP) if delta > 0 else (1.0 / (1.0 + ZOOM_STEP))
        self.apply_zoom(factor, QPointF(event.position()))
        event.accept()

    def mousePressEvent(self, event):
        if self.base_pixmap is None or self.main_window is None:
            return

        if event.button() == Qt.MiddleButton:
            self._pan_start = QPointF(event.position())
            self._pan_start_offset = (self.pan_offset_x, self.pan_offset_y)
            return

        if event.button() != Qt.LeftButton:
            return

        widget_pos = QPointF(event.position())
        img_pos = self.widget_to_image(widget_pos)

        # --- check OBB rotate handle first ---
        rot_idx = self.main_window.get_obb_rotate_handle_at(widget_pos)
        if rot_idx is not None:
            self.main_window.push_undo_state()
            self.main_window.selected_box_index = rot_idx
            self._obb_rotating = True
            box = self.main_window.obb_boxes[rot_idx]
            self._obb_rotate_start_angle = box.angle
            cx_w = box.cx * self.scale + self.offset_x
            cy_w = box.cy * self.scale + self.offset_y
            self._obb_rotate_mouse_angle = math.degrees(
                math.atan2(widget_pos.y() - cy_w, widget_pos.x() - cx_w))
            self.mode = "obb_rotating"
            return

        if img_pos is None:
            return

        self.start_widget_point = widget_pos
        self.current_widget_point = widget_pos
        self.last_img_point = img_pos

        if self.crop_mode_enabled:
            self.mode = "crop"
            self.update_scaled_pixmap()
            return

        # --- check regular box handles ---
        handle = self.main_window.get_handle_at(*img_pos)
        if handle is not None:
            self.main_window.push_undo_state()
            self.mode = "resizing"
            self.resize_handle = handle[1]
            self.main_window.selected_box_index = handle[0]
            self.original_box_before_edit = \
                self.main_window.boxes[handle[0]].copy()
            self.update_scaled_pixmap()
            return

        border_box_index = self.main_window.get_border_at(*img_pos)
        if border_box_index is not None:
            self.main_window.push_undo_state()
            self.mode = "moving"
            self.main_window.selected_box_index = border_box_index
            self.original_box_before_edit = \
                self.main_window.boxes[border_box_index].copy()
            self.update_scaled_pixmap()
            return

        # --- check OBB box interior for move ---
        obb_idx = self.main_window.get_obb_at(*img_pos)
        if obb_idx is not None:
            self.main_window.push_undo_state()
            self.mode = "obb_moving"
            self.main_window.selected_box_index = obb_idx
            self.original_box_before_edit = \
                self.main_window.obb_boxes[obb_idx].copy()
            self.update_scaled_pixmap()
            return

        # --- click in empty space: deselect or start drawing ---
        clicked_box_index = self.main_window.find_box_at(*img_pos)
        self.main_window.selected_box_index = clicked_box_index

        self.mode = "drawing"
        self.resize_handle = None
        self.original_box_before_edit = None
        self.update_scaled_pixmap()

    def mouseMoveEvent(self, event):
        if self.main_window is None or self.base_pixmap is None:
            return

        widget_pos = QPointF(event.position())

        if self._pan_start is not None and (event.buttons() & Qt.MiddleButton):
            dx = widget_pos.x() - self._pan_start.x()
            dy = widget_pos.y() - self._pan_start.y()
            self.pan_offset_x = self._pan_start_offset[0] + dx
            self.pan_offset_y = self._pan_start_offset[1] + dy
            self.update_scaled_pixmap()
            return

        # OBB rotate drag
        if self.mode == "obb_rotating" and self._obb_rotating:
            idx = self.main_window.selected_box_index
            if idx is not None and idx < len(self.main_window.obb_boxes):
                box = self.main_window.obb_boxes[idx]
                cx_w = box.cx * self.scale + self.offset_x
                cy_w = box.cy * self.scale + self.offset_y
                current_mouse_angle = math.degrees(
                    math.atan2(widget_pos.y() - cy_w, widget_pos.x() - cx_w))
                delta_angle = current_mouse_angle - self._obb_rotate_mouse_angle
                box.angle = self._obb_rotate_start_angle + delta_angle
                self.update_scaled_pixmap()
                self.main_window.update_status(
                    extra=f"Rotating window: {box.angle:.1f}°")
            return

        img_pos = self.widget_to_image(widget_pos)

        if self.mode in ("drawing", "crop"):
            self.current_widget_point = widget_pos
            self.update_scaled_pixmap()
            return

        if img_pos is None:
            return

        if self.mode == "moving":
            if self.main_window.selected_box_index is None or \
                    self.last_img_point is None:
                return
            dx = img_pos[0] - self.last_img_point[0]
            dy = img_pos[1] - self.last_img_point[1]
            box = self.main_window.boxes[self.main_window.selected_box_index]
            box.move(dx, dy, self.main_window.current_image_w,
                     self.main_window.current_image_h)
            self.last_img_point = img_pos
            self.update_scaled_pixmap()
            self.main_window.update_status(extra="Moving box...")
            return

        if self.mode == "obb_moving":
            if self.main_window.selected_box_index is None or \
                    self.last_img_point is None:
                return
            dx = img_pos[0] - self.last_img_point[0]
            dy = img_pos[1] - self.last_img_point[1]
            box = self.main_window.obb_boxes[self.main_window.selected_box_index]
            box.move(dx, dy)
            self.last_img_point = img_pos
            self.update_scaled_pixmap()
            self.main_window.update_status(extra="Moving OBB window box...")
            return

        if self.mode == "resizing":
            if self.main_window.selected_box_index is None:
                return
            box = self.main_window.boxes[self.main_window.selected_box_index]
            x, y = img_pos
            if self.resize_handle == "tl":
                box.x1, box.y1 = x, y
            elif self.resize_handle == "tr":
                box.x2, box.y1 = x, y
            elif self.resize_handle == "bl":
                box.x1, box.y2 = x, y
            elif self.resize_handle == "br":
                box.x2, box.y2 = x, y
            box.clamp(self.main_window.current_image_w,
                      self.main_window.current_image_h)
            self.update_scaled_pixmap()
            self.main_window.update_status(extra="Resizing box...")
            return

    def mouseReleaseEvent(self, event):
        if self.base_pixmap is None or self.main_window is None:
            return

        if event.button() == Qt.MiddleButton:
            self._pan_start = None
            self._pan_start_offset = None
            return

        if event.button() != Qt.LeftButton:
            return

        if self.mode == "obb_rotating":
            self._obb_rotating = False
            self.mode = "idle"
            self.update_scaled_pixmap()
            idx = self.main_window.selected_box_index
            if idx is not None and idx < len(self.main_window.obb_boxes):
                a = self.main_window.obb_boxes[idx].angle
                self.main_window.update_status(
                    extra=f"Window angle set to {a:.1f}°")
            return

        if self.mode == "crop":
            end_point = QPointF(event.position())
            start_img = self.widget_to_image(self.start_widget_point) \
                if self.start_widget_point else None
            end_img = self.widget_to_image(end_point)
            self.mode = "idle"
            self.crop_mode_enabled = False
            if start_img is not None and end_img is not None:
                self.main_window.crop_current_image(start_img, end_img)
            else:
                self.main_window.update_status(extra="Crop cancelled.")
            self.start_widget_point = None
            self.current_widget_point = None
            self.last_img_point = None
            self.update_scaled_pixmap()
            return

        if self.mode == "drawing":
            end_point = QPointF(event.position())
            start_img = self.widget_to_image(self.start_widget_point) \
                if self.start_widget_point else None
            end_img = self.widget_to_image(end_point)

            if start_img is not None and end_img is not None:
                x1, y1 = start_img
                x2, y2 = end_img

                if self.main_window.current_class_id == WINDOW_CLASS_ID:
                    # Create OBB box (initially at 0°)
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    w = abs(x2 - x1)
                    h = abs(y2 - y1)
                    new_obb = OBBBox(WINDOW_CLASS_ID, cx, cy, w, h, 0.0)
                    if not new_obb.is_too_small():
                        self.main_window.push_undo_state()
                        self.main_window.obb_boxes.append(new_obb)
                        self.main_window.selected_box_index = \
                            len(self.main_window.obb_boxes) - 1
                        self.main_window.update_status(
                            extra="Window OBB box added. "
                                  "Drag the ◉ handle or Shift+Scroll to rotate.")
                    else:
                        self.main_window.update_status(extra="Ignored tiny box.")
                else:
                    new_box = Box(self.main_window.current_class_id,
                                  x1, y1, x2, y2)
                    if not new_box.is_too_small():
                        self.main_window.push_undo_state()
                        self.main_window.boxes.append(new_box)
                        self.main_window.selected_box_index = \
                            len(self.main_window.boxes) - 1
                        self.main_window.update_status(extra="Box added.")
                    else:
                        self.main_window.update_status(extra="Ignored tiny box.")

        elif self.mode in ("moving", "resizing"):
            if self.main_window.selected_box_index is not None:
                box = self.main_window.boxes[self.main_window.selected_box_index]
                box.clamp(self.main_window.current_image_w,
                           self.main_window.current_image_h)
                if box.is_too_small():
                    if self.original_box_before_edit is not None:
                        self.main_window.boxes[
                            self.main_window.selected_box_index] = \
                            self.original_box_before_edit
                    self.main_window.update_status(
                        extra="Edit cancelled: box too small.")
                else:
                    self.main_window.update_status(extra="Box updated.")

        elif self.mode == "obb_moving":
            self.main_window.update_status(extra="Window box moved.")

        self.mode = "idle"
        self.resize_handle = None
        self.start_widget_point = None
        self.current_widget_point = None
        self.last_img_point = None
        self.original_box_before_edit = None
        self.update_scaled_pixmap()


# ---------------------------------------------------------------------------
# Rotation slider widget
# ---------------------------------------------------------------------------
class RotationSlider(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)
        self._SCALE = 10

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(-1800)
        self.slider.setMaximum(1800)
        self.slider.setValue(0)
        self.slider.setTickInterval(150)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(150)

        self.angle_label = QLabel("0.0°")
        self.angle_label.setFixedWidth(52)
        self.angle_label.setAlignment(Qt.AlignCenter)
        self.angle_label.setStyleSheet("font-weight: bold; color: #ddd;")

        reset_btn = QPushButton("↺ Reset")
        reset_btn.setFixedWidth(64)
        reset_btn.setToolTip("Reset rotation to 0°")
        reset_btn.clicked.connect(self.reset_angle)

        apply_btn = QPushButton("✔ Apply")
        apply_btn.setFixedWidth(70)
        apply_btn.setToolTip("Destructively apply rotation and save image")
        apply_btn.clicked.connect(self._on_apply)

        row = QHBoxLayout(self)
        row.setContentsMargins(6, 4, 6, 4)
        row.setSpacing(6)
        rot_label = QLabel("Rotate:")
        rot_label.setStyleSheet("color: #aaa;")
        row.addWidget(rot_label)
        row.addWidget(self.slider, stretch=1)
        row.addWidget(self.angle_label)
        row.addWidget(reset_btn)
        row.addWidget(apply_btn)

        self.slider.valueChanged.connect(self._on_value_changed)
        self.main_window = None

    def set_main_window(self, window):
        self.main_window = window

    def current_angle(self) -> float:
        return self.slider.value() / self._SCALE

    def reset_angle(self):
        self.slider.setValue(0)

    def _on_value_changed(self, value: int):
        angle = value / self._SCALE
        self.angle_label.setText(f"{angle:.1f}°")
        if self.main_window:
            self.main_window.on_rotation_slider_changed(angle)

    def _on_apply(self):
        angle = self.current_angle()
        if self.main_window:
            self.main_window.apply_rotation_from_slider(angle)
        self.slider.setValue(0)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Water Meter YOLO Labeler")
        self.resize(1280, 900)

        self.image_dir = None
        self.label_dir = None
        self.image_files = []
        self.current_index = 0

        self.current_image = None
        self.current_image_w = 0
        self.current_image_h = 0

        self.boxes: List[Box] = []           # regular axis-aligned boxes
        self.obb_boxes: List[OBBBox] = []    # OBB boxes for window class
        self.selected_box_index = None       # index into boxes OR obb_boxes
                                             # (depends on context / mode)
        self.current_class_id = 0
        self.undo_stack = []
        self._preview_angle = 0.0

        # --- Canvas ---
        self.canvas = ImageCanvas()
        self.canvas.set_main_window(self)

        # --- Info label ---
        self.info_label = QLabel("Open an image folder to begin.")
        self.info_label.setWordWrap(True)

        # --- Zoom label ---
        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_label.setFixedWidth(100)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setStyleSheet("color: #aaa; font-size: 12px;")

        # --- OBB angle indicator ---
        self.obb_angle_label = QLabel("")
        self.obb_angle_label.setFixedWidth(160)
        self.obb_angle_label.setAlignment(Qt.AlignCenter)
        self.obb_angle_label.setStyleSheet(
            "color: #ff6666; font-size: 12px; font-weight: bold;")

        # --- Buttons ---
        open_btn = QPushButton("Open Folder")
        open_btn.clicked.connect(self.open_folder)
        prev_btn = QPushButton("Previous (P)")
        prev_btn.clicked.connect(self.prev_image)
        next_btn = QPushButton("Next (N)")
        next_btn.clicked.connect(self.next_image)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_labels)
        remove_btn = QPushButton("Remove Image")
        remove_btn.clicked.connect(self.remove_current_image)
        undo_btn = QPushButton("Undo")
        undo_btn.clicked.connect(self.undo)
        crop_btn = QPushButton("Crop Image")
        crop_btn.clicked.connect(self.enable_crop_mode)
        rot_left_btn = QPushButton("Rotate -15°")
        rot_left_btn.clicked.connect(lambda: self.rotate_current_image(-15))
        rot_right_btn = QPushButton("Rotate +15°")
        rot_right_btn.clicked.connect(lambda: self.rotate_current_image(15))
        rot_90_btn = QPushButton("Rotate 90°")
        rot_90_btn.clicked.connect(lambda: self.rotate_current_image(90))
        rot_custom_btn = QPushButton("Rotate Custom")
        rot_custom_btn.clicked.connect(self.rotate_custom_angle)

        zoom_in_btn = QPushButton("Zoom In (+)")
        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_out_btn = QPushButton("Zoom Out (–)")
        zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_reset_btn = QPushButton("Zoom Reset")
        zoom_reset_btn.clicked.connect(self.zoom_reset)

        top_bar = QHBoxLayout()
        top_bar.addWidget(open_btn)
        top_bar.addWidget(prev_btn)
        top_bar.addWidget(next_btn)
        top_bar.addWidget(save_btn)
        top_bar.addWidget(remove_btn)
        top_bar.addWidget(undo_btn)
        top_bar.addWidget(crop_btn)
        top_bar.addWidget(rot_left_btn)
        top_bar.addWidget(rot_right_btn)
        top_bar.addWidget(rot_90_btn)
        top_bar.addWidget(rot_custom_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        top_bar.addWidget(sep)

        top_bar.addWidget(zoom_out_btn)
        top_bar.addWidget(self.zoom_label)
        top_bar.addWidget(zoom_in_btn)
        top_bar.addWidget(zoom_reset_btn)
        top_bar.addWidget(self.obb_angle_label)
        top_bar.addStretch()

        # OBB hint bar
        obb_hint = QLabel(
            "  🔴 Window class (W): draw box → drag ◉ handle or Shift+Scroll to rotate  |  "
            "Ctrl+[ / Ctrl+] = rotate selected window box ±1°"
        )
        obb_hint.setStyleSheet(
            "color: #ff8888; background: #1a0000; "
            "padding: 3px 8px; font-size: 11px;")

        # Rotation slider
        self.rotation_slider = RotationSlider()
        self.rotation_slider.set_main_window(self)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_bar)
        main_layout.addWidget(obb_hint)
        main_layout.addWidget(self.canvas, stretch=1)
        main_layout.addWidget(self.rotation_slider)
        main_layout.addWidget(self.info_label)

        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        self.create_shortcuts()
        self.create_menu()
        self.update_status()
        self.update_zoom_indicator()

    # ------------------------------------------------------------------
    # OBB helpers
    # ------------------------------------------------------------------
    def get_obb_at(self, x, y) -> Optional[int]:
        """Return index of OBBBox containing image point (x, y), or None."""
        for i in range(len(self.obb_boxes) - 1, -1, -1):
            if self.obb_boxes[i].contains_point(x, y):
                return i
        return None

    def get_obb_rotate_handle_at(self, widget_pos: QPointF) -> Optional[int]:
        """Return index of OBBBox whose rotate handle is near widget_pos."""
        for i in range(len(self.obb_boxes) - 1, -1, -1):
            hx, hy = self.obb_boxes[i].rotate_handle_widget(
                self.canvas.scale, self.canvas.offset_x, self.canvas.offset_y)
            dist = math.hypot(widget_pos.x() - hx, widget_pos.y() - hy)
            if dist <= 12:
                return i
        return None

    def rotate_selected_obb(self, delta_degrees: float):
        """Rotate currently-selected OBB box by delta_degrees."""
        if self.selected_box_index is None:
            return
        # Determine if selection refers to an OBB box
        # (selected_box_index is used for both lists; we check obb list first
        #  when current class is window)
        if self.current_class_id == WINDOW_CLASS_ID and \
                0 <= self.selected_box_index < len(self.obb_boxes):
            self.obb_boxes[self.selected_box_index].angle += delta_degrees
            self.canvas.update_scaled_pixmap()
            a = self.obb_boxes[self.selected_box_index].angle
            self.obb_angle_label.setText(f"Window: {a:.1f}°")
            self.update_status(extra=f"Window angle: {a:.1f}°")
        elif 0 <= self.selected_box_index < len(self.obb_boxes):
            # fallback: if there are obb boxes selected
            self.obb_boxes[self.selected_box_index].angle += delta_degrees
            self.canvas.update_scaled_pixmap()

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------
    def zoom_in(self):
        if self.canvas.base_pixmap is None:
            return
        self.canvas.apply_zoom(1.0 + ZOOM_STEP)

    def zoom_out(self):
        if self.canvas.base_pixmap is None:
            return
        self.canvas.apply_zoom(1.0 / (1.0 + ZOOM_STEP))

    def zoom_reset(self):
        self.canvas.reset_zoom()

    def update_zoom_indicator(self):
        pct = int(self.canvas.zoom_level * 100)
        self.zoom_label.setText(f"Zoom: {pct}%")

    # ------------------------------------------------------------------
    # Rotation slider callbacks
    # ------------------------------------------------------------------
    def on_rotation_slider_changed(self, angle: float):
        self._preview_angle = angle
        self._refresh_canvas_with_preview()

    def _refresh_canvas_with_preview(self):
        if self.current_image is None:
            return
        if abs(self._preview_angle) < 0.05:
            rotated = self.current_image
        else:
            rotated = rotate_image_keep_size_crop_edges(
                self.current_image, self._preview_angle)
        rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        h, w = rotated.shape[:2]
        qimage = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage.copy())
        self.canvas.base_pixmap = pixmap
        self.canvas.update_scaled_pixmap()

    def apply_rotation_from_slider(self, angle: float):
        if abs(angle) < 0.05:
            self.update_status(extra="No rotation to apply.")
            return
        self._preview_angle = 0.0
        self._restore_original_canvas()
        self.rotate_current_image(angle)

    def _restore_original_canvas(self):
        if self.current_image is None:
            return
        rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        h, w = self.current_image.shape[:2]
        qimage = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        self.canvas.base_pixmap = QPixmap.fromImage(qimage.copy())
        self.canvas.update_scaled_pixmap()

    # ------------------------------------------------------------------
    # Menu + shortcuts
    # ------------------------------------------------------------------
    def create_menu(self):
        open_action = QAction("Open Folder", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_folder)

        save_action = QAction("Save", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_labels)

        remove_action = QAction("Remove Image", self)
        remove_action.setShortcut(QKeySequence("Ctrl+D"))
        remove_action.triggered.connect(self.remove_current_image)

        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(self.undo)

        crop_action = QAction("Crop Image", self)
        crop_action.setShortcut(QKeySequence("C"))
        crop_action.triggered.connect(self.enable_crop_mode)

        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut(QKeySequence("="))
        zoom_in_action.triggered.connect(self.zoom_in)

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut(QKeySequence("-"))
        zoom_out_action.triggered.connect(self.zoom_out)

        zoom_reset_action = QAction("Zoom Reset", self)
        zoom_reset_action.setShortcut(QKeySequence("Z"))
        zoom_reset_action.triggered.connect(self.zoom_reset)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        file_menu.addAction(remove_action)

        edit_menu = self.menuBar().addMenu("Edit")
        edit_menu.addAction(undo_action)
        edit_menu.addAction(crop_action)

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(zoom_in_action)
        view_menu.addAction(zoom_out_action)
        view_menu.addAction(zoom_reset_action)

    def create_shortcuts(self):
        shortcuts = [
            ("OpenFolder",    "Ctrl+O",     self.open_folder),
            ("Save",          "Ctrl+S",     self.save_labels),
            ("Undo",          "Ctrl+Z",     self.undo),
            ("Next",          "N",          self.next_image),
            ("Prev",          "P",          self.prev_image),
            ("RemoveImage",   "Ctrl+D",     self.remove_current_image),
            ("CropImage",     "C",          self.enable_crop_mode),
            ("RotMinus15",    "Ctrl+Left",  lambda: self.rotate_current_image(-15)),
            ("RotPlus15",     "Ctrl+Right", lambda: self.rotate_current_image(15)),
            ("Rotate90",      "R",          lambda: self.rotate_current_image(90)),
            ("RotateCustom",  "Ctrl+R",     self.rotate_custom_angle),
            ("Meter",         "M",          lambda: self.set_class(0)),
            ("Window",        "W",          lambda: self.set_class(1)),
            ("ZeroDigit",     "0",          lambda: self.set_class(2)),
            ("OneDigit",      "1",          lambda: self.set_class(3)),
            ("TwoDigit",      "2",          lambda: self.set_class(4)),
            ("ThreeDigit",    "3",          lambda: self.set_class(5)),
            ("FourDigit",     "4",          lambda: self.set_class(6)),
            ("FiveDigit",     "5",          lambda: self.set_class(7)),
            ("SixDigit",      "6",          lambda: self.set_class(8)),
            ("SevenDigit",    "7",          lambda: self.set_class(9)),
            ("EightDigit",    "8",          lambda: self.set_class(10)),
            ("NineDigit",     "9",          lambda: self.set_class(11)),
            ("UnknownDigit",  "U",          lambda: self.set_class(12)),
            ("Delete",        "Delete",     self.delete_selected_box),
            ("DeleteBksp",    "Backspace",  self.delete_selected_box),
            ("ClearSel",      "Escape",     self.clear_selection),
            ("ZoomIn",        "=",          self.zoom_in),
            ("ZoomOut",       "-",          self.zoom_out),
            ("ZoomReset",     "Z",          self.zoom_reset),
            # Fine OBB rotation
            ("OBBRotCCW",     "Ctrl+[",     lambda: self.rotate_selected_obb(-1.0)),
            ("OBBRotCW",      "Ctrl+]",     lambda: self.rotate_selected_obb(1.0)),
        ]
        for name, shortcut, slot in shortcuts:
            action = QAction(name, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(slot)
            self.addAction(action)

    # ------------------------------------------------------------------
    # Image / folder management
    # ------------------------------------------------------------------
    def enable_crop_mode(self):
        if self.current_image is None:
            self.update_status(extra="Open an image first.")
            return
        self.canvas.crop_mode_enabled = True
        self.canvas.mode = "idle"
        self.selected_box_index = None
        self.canvas.update_scaled_pixmap()
        self.update_status(
            extra="Crop mode enabled. Draw a rectangle around the useful area.")

    def open_folder(self):
        start_folder = self.image_dir if self.image_dir else os.getcwd()
        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", start_folder,
            QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog)
        if not folder:
            return
        image_files = list_image_files(folder)
        if not image_files:
            safe_message(self, "No Images",
                         "This folder has no supported images.")
            return
        self.image_dir = folder
        parent_dir = os.path.dirname(folder)
        self.label_dir = os.path.join(parent_dir, "labels_obb")
        os.makedirs(self.label_dir, exist_ok=True)
        self.image_files = image_files
        self.current_index = 0
        self.load_current_image()
        self.update_status(
            extra=f"Images: {self.image_dir} | Labels: {self.label_dir}")

    def current_label_path(self):
        if not self.label_dir or not self.image_files:
            return None
        stem = os.path.splitext(self.image_files[self.current_index])[0]
        return os.path.join(self.label_dir, f"{stem}.txt")

    def push_undo_state(self):
        self.undo_stack.append(
            ([b.copy() for b in self.boxes],
             [b.copy() for b in self.obb_boxes]))
        if len(self.undo_stack) > 200:
            self.undo_stack.pop(0)

    def undo(self):
        if not self.undo_stack:
            self.update_status(extra="Nothing to undo.")
            return
        self.boxes, self.obb_boxes = self.undo_stack.pop()
        self.selected_box_index = None
        self.canvas.update_scaled_pixmap()
        self.update_status(extra="Undid last action.")

    def clear_selection(self):
        self.selected_box_index = None
        self.canvas.crop_mode_enabled = False
        self.obb_angle_label.setText("")
        self.canvas.update_scaled_pixmap()
        self.update_status(extra="Selection cleared.")

    def load_current_image(self):
        if not self.image_dir or not self.image_files:
            return
        image_path = os.path.join(self.image_dir,
                                   self.image_files[self.current_index])
        image = cv2.imread(image_path)
        if image is None:
            self.canvas.clear_image()
            self.current_image = None
            self.current_image_w = 0
            self.current_image_h = 0
            self.boxes = []
            self.obb_boxes = []
            self.selected_box_index = None
            self.update_status(
                extra=f"Failed to load: {self.image_files[self.current_index]}")
            return

        self.current_image = image
        self.current_image_h, self.current_image_w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb.data, self.current_image_w, self.current_image_h,
                        rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage.copy())

        self.undo_stack = []
        self.boxes = []
        self.obb_boxes = []
        self.selected_box_index = None
        self.canvas.crop_mode_enabled = False
        self._preview_angle = 0.0
        self.rotation_slider.slider.blockSignals(True)
        self.rotation_slider.slider.setValue(0)
        self.rotation_slider.slider.blockSignals(False)
        self.rotation_slider.angle_label.setText("0.0°")
        self.obb_angle_label.setText("")

        self.canvas.set_image(pixmap)
        self.load_labels()
        self.update_status()
        self.update_zoom_indicator()

    # ------------------------------------------------------------------
    # Label I/O
    # ------------------------------------------------------------------
    def load_labels(self):
        self.boxes = []
        self.obb_boxes = []
        self.selected_box_index = None

        label_path = self.current_label_path()
        if not label_path or not os.path.exists(label_path):
            self.canvas.update_scaled_pixmap()
            return

        try:
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    class_id = int(parts[0])

                    if class_id == WINDOW_CLASS_ID and len(parts) == 9:
                        # OBB format: class x1 y1 x2 y2 x3 y3 x4 y4
                        obb = OBBBox.from_yolo_line(
                            parts, self.current_image_w, self.current_image_h)
                        self.obb_boxes.append(obb)

                    elif len(parts) == 5:
                        # Normal YOLO
                        x_center, y_center, width, height = map(float, parts[1:])
                        if class_id < 0 or class_id >= len(CLASS_NAMES):
                            continue
                        box = Box.from_normalized(
                            class_id, x_center, y_center, width, height,
                            self.current_image_w, self.current_image_h)
                        self.boxes.append(box)

        except Exception as e:
            self.update_status(extra=f"Label load error: {e}")

        self.canvas.update_scaled_pixmap()

    def save_labels(self):
        label_path = self.current_label_path()
        if not label_path:
            return
        try:
            with open(label_path, "w", encoding="utf-8") as f:
                # Regular boxes
                for box in self.boxes:
                    xc, yc, w, h = box.normalized(
                        self.current_image_w, self.current_image_h)
                    f.write(f"{box.class_id} "
                            f"{xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                # OBB boxes
                for obb in self.obb_boxes:
                    f.write(obb.to_yolo_line(
                        self.current_image_w, self.current_image_h) + "\n")
            self.update_status(extra="Saved.")
        except Exception as e:
            self.update_status(extra=f"Save error: {e}")

    def clear_label_file(self):
        label_path = self.current_label_path()
        if label_path and os.path.exists(label_path):
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("")

    # ------------------------------------------------------------------
    # Image operations
    # ------------------------------------------------------------------
    def crop_current_image(self, start_img, end_img):
        if not self.image_dir or not self.image_files:
            return
        image_path = os.path.join(self.image_dir,
                                   self.image_files[self.current_index])
        if self.boxes or self.obb_boxes:
            confirmed = safe_confirm(
                self, "Crop Image",
                "This image already has labels.\n\n"
                "Cropping will clear them. Continue?",
                yes_text="Yes, Crop", no_text="Cancel")
            if not confirmed:
                self.update_status(extra="Cropping cancelled.")
                return
        image = cv2.imread(image_path)
        if image is None:
            return
        x1, y1 = start_img
        x2, y2 = end_img
        cropped = crop_image(image, x1, y1, x2, y2)
        if cropped is None:
            self.update_status(extra="Invalid crop area.")
            return
        cv2.imwrite(image_path, cropped)
        self.clear_label_file()
        self.load_current_image()
        self.update_status(extra="Image cropped and saved. Labels cleared.")

    def remove_current_image(self):
        if not self.image_dir or not self.image_files:
            return
        image_name = self.image_files[self.current_index]
        image_path = os.path.join(self.image_dir, image_name)
        label_path = self.current_label_path()

        confirmed = safe_confirm(
            self, "Remove Image",
            f"Remove {image_name} from the dataset?\n\n"
            "Image and label will be moved to removed_images/ and removed_labels/.",
            yes_text="Yes, Remove", no_text="Cancel")
        if not confirmed:
            return

        dataset_root = os.path.dirname(self.image_dir)
        removed_images_dir = os.path.join(dataset_root, "obb_removed_images")
        removed_labels_dir = os.path.join(dataset_root, "obb_removed_labels")
        os.makedirs(removed_images_dir, exist_ok=True)
        os.makedirs(removed_labels_dir, exist_ok=True)

        try:
            self.canvas.clear_image()
            self.current_image = None
            self.current_image_w = 0
            self.current_image_h = 0
            self.boxes = []
            self.obb_boxes = []
            self.selected_box_index = None
            self.undo_stack = []

            if os.path.exists(image_path):
                dst = unique_destination_path(
                    os.path.join(removed_images_dir, image_name))
                shutil.move(image_path, dst)
            if label_path and os.path.exists(label_path):
                dst = unique_destination_path(
                    os.path.join(removed_labels_dir,
                                 os.path.basename(label_path)))
                shutil.move(label_path, dst)

            del self.image_files[self.current_index]
            if not self.image_files:
                self.current_index = 0
                self.update_status(extra="Removed image. No images left.")
                return
            if self.current_index >= len(self.image_files):
                self.current_index = len(self.image_files) - 1
            self.load_current_image()
            self.update_status(extra=f"Removed image: {image_name}")

        except Exception as e:
            self.update_status(extra=f"Remove error: {e}")

    def rotate_current_image(self, angle_degrees: float):
        if not self.image_dir or not self.image_files:
            return
        image_path = os.path.join(self.image_dir,
                                   self.image_files[self.current_index])
        if self.boxes or self.obb_boxes:
            confirmed = safe_confirm(
                self, "Rotate Image",
                "This image already has labels.\n\n"
                "Rotating will clear them. Continue?",
                yes_text="Yes, Rotate", no_text="Cancel")
            if not confirmed:
                self.update_status(extra="Rotation cancelled.")
                return
        image = cv2.imread(image_path)
        if image is None:
            return
        rotated = rotate_image_keep_size_crop_edges(image, angle_degrees)
        cv2.imwrite(image_path, rotated)
        self.clear_label_file()
        self.load_current_image()
        self.update_status(
            extra=f"Image rotated by {angle_degrees}° and saved. Labels cleared.")

    def rotate_custom_angle(self):
        angle, ok = QInputDialog.getDouble(
            self, "Rotate Image", "Enter rotation angle in degrees:",
            0.0, -360.0, 360.0, 2)
        if ok:
            self.rotate_current_image(angle)

    def next_image(self):
        if not self.image_files:
            return
        self.save_labels()
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()

    def prev_image(self):
        if not self.image_files:
            return
        self.save_labels()
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()

    def set_class(self, class_id: int):
        self.current_class_id = class_id
        self.canvas.crop_mode_enabled = False
        if self.selected_box_index is not None and \
                0 <= self.selected_box_index < len(self.boxes):
            if self.boxes[self.selected_box_index].class_id != class_id and \
                    class_id != WINDOW_CLASS_ID:
                self.push_undo_state()
                self.boxes[self.selected_box_index].class_id = class_id
                self.canvas.update_scaled_pixmap()
                self.update_status(
                    extra=f"Selected box → {CLASS_NAMES[class_id]}")
                return
        self.update_status(extra=f"Current class: {CLASS_NAMES[class_id]}")

    def delete_selected_box(self):
        if self.selected_box_index is None:
            self.update_status(extra="No box selected.")
            return
        # Try OBB first when window class active
        if self.current_class_id == WINDOW_CLASS_ID and \
                0 <= self.selected_box_index < len(self.obb_boxes):
            self.push_undo_state()
            del self.obb_boxes[self.selected_box_index]
            self.selected_box_index = None
            self.obb_angle_label.setText("")
            self.canvas.update_scaled_pixmap()
            self.update_status(extra="Window OBB box deleted.")
            return
        if 0 <= self.selected_box_index < len(self.boxes):
            self.push_undo_state()
            del self.boxes[self.selected_box_index]
            self.selected_box_index = None
            self.canvas.update_scaled_pixmap()
            self.update_status(extra="Box deleted.")

    def find_box_at(self, x, y):
        for i in range(len(self.boxes) - 1, -1, -1):
            if self.boxes[i].contains(x, y):
                return i
        return None

    def get_handle_at(self, x, y):
        tolerance = max(6, HANDLE_SIZE / max(self.canvas.scale, 1e-6))
        for i in range(len(self.boxes) - 1, -1, -1):
            box = self.boxes[i]
            x_min, y_min, x_max, y_max = box.rect()
            handles = {
                "tl": (x_min, y_min), "tr": (x_max, y_min),
                "bl": (x_min, y_max), "br": (x_max, y_max),
            }
            for name, (hx, hy) in handles.items():
                if abs(x - hx) <= tolerance and abs(y - hy) <= tolerance:
                    return i, name
        return None

    def get_border_at(self, x, y):
        tolerance = max(6, HANDLE_SIZE / max(self.canvas.scale, 1e-6))
        for i in range(len(self.boxes) - 1, -1, -1):
            box = self.boxes[i]
            x_min, y_min, x_max, y_max = box.rect()
            near_left = abs(x - x_min) <= tolerance and y_min <= y <= y_max
            near_right = abs(x - x_max) <= tolerance and y_min <= y <= y_max
            near_top = abs(y - y_min) <= tolerance and x_min <= x <= x_max
            near_bottom = abs(y - y_max) <= tolerance and x_min <= x <= x_max
            if near_left or near_right or near_top or near_bottom:
                return i
        return None

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw_boxes(self, painter: QPainter, scale: float,
                   offset_x: int, offset_y: int):
        # Regular boxes
        for i, box in enumerate(self.boxes):
            x_min, y_min, x_max, y_max = box.rect()
            x1 = x_min * scale + offset_x
            y1 = y_min * scale + offset_y
            x2 = x_max * scale + offset_x
            y2 = y_max * scale + offset_y

            color = CLASS_COLORS.get(box.class_id, QColor(255, 255, 255))
            pen_width = 3 if i == self.selected_box_index else 2
            painter.setPen(QPen(color, pen_width))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(QPointF(x1, y1), QPointF(x2, y2)))

            label_text = CLASS_NAMES[box.class_id]
            text_width = 22 if label_text in list("0123456789") \
                else len(label_text) * 8 + 12
            text_height = 20
            label_top = max(0, int(y1) - text_height)
            painter.fillRect(int(x1), label_top, text_width, text_height, color)
            painter.setPen(Qt.black)
            painter.drawText(int(x1) + 5, label_top + 15, label_text)

            if i == self.selected_box_index:
                painter.setBrush(QBrush(QColor(255, 255, 255)))
                painter.setPen(QPen(Qt.black, 1))
                for hx, hy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                    painter.drawRect(QRectF(
                        hx - HANDLE_SIZE / 2, hy - HANDLE_SIZE / 2,
                        HANDLE_SIZE, HANDLE_SIZE))

        # OBB boxes
        for i, obb in enumerate(self.obb_boxes):
            color = CLASS_COLORS.get(obb.class_id, QColor(255, 0, 0))
            is_selected = (i == self.selected_box_index and
                           self.current_class_id == WINDOW_CLASS_ID)
            # also treat as selected if canvas mode is obb_rotating/obb_moving
            if self.canvas.mode in ("obb_rotating", "obb_moving") and \
                    i == self.selected_box_index:
                is_selected = True

            pen_width = 3 if is_selected else 2
            painter.setPen(QPen(color, pen_width))
            painter.setBrush(Qt.NoBrush)

            corners_w = obb.corners_widget(scale, offset_x, offset_y)
            poly = QPolygonF([QPointF(x, y) for x, y in corners_w])
            painter.drawPolygon(poly)

            # Label tag
            tx, ty = corners_w[0]
            tag_text = f"window {obb.angle:.1f}°"
            tag_w = len(tag_text) * 7 + 10
            tag_h = 18
            tag_top = max(0, int(ty) - tag_h)
            painter.fillRect(int(tx), tag_top, tag_w, tag_h, color)
            painter.setPen(Qt.black)
            painter.drawText(int(tx) + 4, tag_top + 13, tag_text)

            # Rotation handle (circle above top-centre)
            hx, hy = obb.rotate_handle_widget(scale, offset_x, offset_y)
            # Line from top-centre to handle
            top_cx = (corners_w[0][0] + corners_w[1][0]) / 2.0
            top_cy = (corners_w[0][1] + corners_w[1][1]) / 2.0
            painter.setPen(QPen(color, 1, Qt.DashLine))
            painter.drawLine(QPointF(top_cx, top_cy), QPointF(hx, hy))

            # Handle circle
            hr = 8
            if is_selected:
                painter.setBrush(QBrush(QColor(255, 255, 255)))
                painter.setPen(QPen(color, 2))
            else:
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(Qt.black, 1))
            painter.drawEllipse(QPointF(hx, hy), hr, hr)

            # Draw direction arrow on the box to show "long side"
            # Arrow along the long axis from left-centre to right-centre
            if is_selected:
                lc_x = (corners_w[0][0] + corners_w[3][0]) / 2.0
                lc_y = (corners_w[0][1] + corners_w[3][1]) / 2.0
                rc_x = (corners_w[1][0] + corners_w[2][0]) / 2.0
                rc_y = (corners_w[1][1] + corners_w[2][1]) / 2.0
                painter.setPen(QPen(QColor(255, 255, 100), 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawLine(QPointF(lc_x, lc_y), QPointF(rc_x, rc_y))
                # Arrowhead at right end
                dx = rc_x - lc_x
                dy = rc_y - lc_y
                length = math.hypot(dx, dy)
                if length > 1:
                    ux = dx / length
                    uy = dy / length
                    arr_len = 10
                    arr_w = 5
                    p1 = QPointF(rc_x - ux * arr_len - uy * arr_w,
                                 rc_y - uy * arr_len + ux * arr_w)
                    p2 = QPointF(rc_x - ux * arr_len + uy * arr_w,
                                 rc_y - uy * arr_len - ux * arr_w)
                    painter.drawLine(QPointF(rc_x, rc_y), p1)
                    painter.drawLine(QPointF(rc_x, rc_y), p2)

    # ------------------------------------------------------------------
    def update_status(self, extra: str = ""):
        shortcuts_text = (
            "M=meter | W=window(OBB) | 0..9=digit | U=unknown | "
            "C=crop | N=next | P=prev | R=rotate90° | "
            "Ctrl←=-15° | Ctrl→=+15° | Ctrl+R=custom | "
            "Scroll=zoom | =/- =zoom | Z=reset | Mid-drag=pan | "
            "OBB: drag◉=rotate | Shift+Scroll=fine rotate | Ctrl+[/]=±1° | "
            "Delete=del box | Ctrl+D=remove img | Ctrl+Z=undo | Esc=clear"
        )
        if not self.image_files:
            self.info_label.setText(
                f"Open an image folder to begin.\n{shortcuts_text}")
            return

        image_name = self.image_files[self.current_index]
        class_name = CLASS_NAMES[self.current_class_id]
        status = (
            f"Image {self.current_index + 1}/{len(self.image_files)}: "
            f"{image_name} | "
            f"Class: {self.current_class_id} ({class_name}) | "
            f"Boxes: {len(self.boxes)} regular + {len(self.obb_boxes)} OBB"
        )
        if extra:
            status += f" | {extra}"
        status += "\n" + shortcuts_text
        self.info_label.setText(status)

    def closeEvent(self, event):
        try:
            self.save_labels()
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()