#!/usr/bin/env python3
"""
01_label_meter.py  —  Water Meter Window Labeler with Image Rotation

Draw the digit-window OBB (Oriented Bounding Box):
  1. Press W, draw a rectangle over the digit window
  2. Drag any CORNER or EDGE to resize at any angle
  3. Drag the CENTER to move
  4. Drag the ◉ handle (or Shift+Scroll) to rotate
  5. Press S / Ctrl+S to save

Image rotation (same as label_images.py):
  - Rotate -15° / +15° / 90° / Custom buttons
  - Rotation slider for live preview → Apply to save destructively
  - Ctrl+Left / Ctrl+Right / R shortcut

Other classes (meter, digits, unknown) use regular axis-aligned boxes.

Keyboard:
  M=meter  W=window(OBB)  0-9=digits  U=unknown
  N=next   P=prev   S/Ctrl+S=save   Delete=delete selected   R=reject image
  Ctrl+Z=undo   Escape=deselect
  Scroll=zoom   Middle-drag=pan   Z=reset zoom
  Ctrl+[ / Ctrl+] = rotate OBB ±1°   Shift+Scroll = fine rotate OBB
  Ctrl+Left=-15°  Ctrl+Right=+15°  Ctrl+R=custom rotate
"""

import math, os, sys, shutil
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore  import QPointF, QRectF, Qt
from PySide6.QtGui   import (QAction, QColor, QImage, QKeySequence,
                              QPainter, QPen, QPixmap, QBrush,
                              QWheelEvent, QPolygonF)
from PySide6.QtWidgets import (QApplication, QFileDialog, QInputDialog,
                                QLabel, QMainWindow, QPushButton,
                                QHBoxLayout, QVBoxLayout, QWidget,
                                QSizePolicy, QDialog, QFrame, QSlider)

# ── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = ["meter","window","0","1","2","3","4","5","6","7","8","9","unknown"]
CLASS_COLORS = {
    0: QColor(0,200,0),    1: QColor(255,60,60),
    2: QColor(255,220,0),  3: QColor(255,160,0),  4: QColor(0,220,255),
    5: QColor(200,0,255),  6: QColor(100,255,0),  7: QColor(0,120,255),
    8: QColor(255,120,120),9: QColor(140,0,255), 10: QColor(200,200,255),
    11: QColor(255,200,200),12: QColor(160,160,160),
}
IMAGE_EXTENSIONS = {".jpg",".jpeg",".png",".bmp",".webp"}
WINDOW_CLS = 1
HANDLE_R   = 9
ROT_DIST   = 32
ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 0.1, 10.0, 0.15


# ── Helpers ──────────────────────────────────────────────────────────────────
def rot2d(x, y, a_rad):
    c, s = math.cos(a_rad), math.sin(a_rad)
    return x*c - y*s, x*s + y*c

def rotate_image(image, angle_deg: float):
    """Rotate image keeping same canvas size (crops edges)."""
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0,0,0))

def safe_confirm(parent, title, message, yes_text="Yes", no_text="Cancel") -> bool:
    d = QDialog(parent); d.setWindowTitle(title); d.setModal(True); d.resize(480,180)
    lbl = QLabel(message); lbl.setWordWrap(True)
    y = QPushButton(yes_text); n = QPushButton(no_text)
    y.clicked.connect(d.accept); n.clicked.connect(d.reject)
    btns = QHBoxLayout(); btns.addStretch(); btns.addWidget(y); btns.addWidget(n)
    lay = QVBoxLayout(); lay.addWidget(lbl); lay.addStretch(); lay.addLayout(btns)
    d.setLayout(lay)
    return d.exec() == QDialog.Accepted


# ── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class Box:
    cls: int; x1: float; y1: float; x2: float; y2: float
    def copy(self): return Box(self.cls,self.x1,self.y1,self.x2,self.y2)
    def rect(self): return min(self.x1,self.x2),min(self.y1,self.y2),max(self.x1,self.x2),max(self.y1,self.y2)
    def contains(self,x,y):
        a,b,c,d=self.rect(); return a<=x<=c and b<=y<=d
    def w(self): return abs(self.x2-self.x1)
    def h(self): return abs(self.y2-self.y1)
    def too_small(self): return self.w()<4 or self.h()<4
    def normalized(self,iw,ih):
        a,b,c,d=self.rect()
        return (a+c)/2/iw,(b+d)/2/ih,(c-a)/iw,(d-b)/ih
    def move(self,dx,dy,iw,ih):
        a,b,c,d=self.rect(); bw,bh=c-a,d-b
        nx=max(0,min(iw-bw,a+dx)); ny=max(0,min(ih-bh,b+dy))
        self.x1,self.y1,self.x2,self.y2=nx,ny,nx+bw,ny+bh
    def clamp(self,iw,ih):
        self.x1=max(0,min(iw,self.x1)); self.y1=max(0,min(ih,self.y1))
        self.x2=max(0,min(iw,self.x2)); self.y2=max(0,min(ih,self.y2))


@dataclass
class OBB:
    """Oriented bounding box stored as cx,cy,w,h (pixels) + angle (degrees CCW)."""
    cx: float; cy: float; w: float; h: float; angle: float = 0.0
    cls: int = WINDOW_CLS

    def copy(self): return OBB(self.cx,self.cy,self.w,self.h,self.angle,self.cls)

    def corners(self) -> List[Tuple[float,float]]:
        hw,hh=self.w/2,self.h/2; rad=math.radians(self.angle)
        return [(rot2d(lx,ly,rad)[0]+self.cx, rot2d(lx,ly,rad)[1]+self.cy)
                for lx,ly in [(-hw,-hh),(hw,-hh),(hw,hh),(-hw,hh)]]

    def corners_w(self,sc,ox,oy):
        return [(x*sc+ox,y*sc+oy) for x,y in self.corners()]

    def handle_points(self):
        c=self.corners()
        mids=[((c[i][0]+c[(i+1)%4][0])/2,(c[i][1]+c[(i+1)%4][1])/2) for i in range(4)]
        return c+mids

    def rot_handle(self,sc,ox,oy):
        c=self.corners()
        tx,ty=(c[0][0]+c[1][0])/2,(c[0][1]+c[1][1])/2
        rad=math.radians(self.angle); dist=ROT_DIST/sc
        return (tx-math.sin(rad)*dist)*sc+ox, (ty-math.cos(rad)*dist)*sc+oy

    def contains(self,x,y):
        rad=math.radians(-self.angle); dx,dy=x-self.cx,y-self.cy
        lx,ly=rot2d(dx,dy,rad)
        return abs(lx)<=self.w/2 and abs(ly)<=self.h/2

    def too_small(self): return self.w<4 or self.h<4

    def to_yolo(self,iw,ih):
        pts=self.corners()
        vals=[str(self.cls)]
        for px,py in pts: vals+=[f"{px/iw:.6f}",f"{py/ih:.6f}"]
        return " ".join(vals)

    @staticmethod
    def from_yolo(parts,iw,ih):
        cls=int(parts[0])
        coords=[float(v) for v in parts[1:9]]
        pts=[(coords[i]*iw,coords[i+1]*ih) for i in range(0,8,2)]
        cx=sum(p[0] for p in pts)/4; cy=sum(p[1] for p in pts)/4
        w=math.hypot(pts[1][0]-pts[0][0],pts[1][1]-pts[0][1])
        h=math.hypot(pts[3][0]-pts[0][0],pts[3][1]-pts[0][1])
        angle=math.degrees(math.atan2(pts[1][1]-pts[0][1],pts[1][0]-pts[0][0]))
        return OBB(cx,cy,w,h,angle,cls)


# ── Rotation Slider (identical logic to label_images.py) ─────────────────────
class RotationSlider(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(52)
        self._SCALE = 10
        self.app = None

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(-1800); self.slider.setMaximum(1800)
        self.slider.setValue(0)
        self.slider.setTickInterval(150); self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setSingleStep(1);    self.slider.setPageStep(150)

        self.angle_lbl = QLabel("0.0°")
        self.angle_lbl.setFixedWidth(52); self.angle_lbl.setAlignment(Qt.AlignCenter)
        self.angle_lbl.setStyleSheet("font-weight:bold;color:#ddd;")

        reset_btn = QPushButton("↺ Reset"); reset_btn.setFixedWidth(64)
        reset_btn.clicked.connect(self.reset)

        apply_btn = QPushButton("✔ Apply"); apply_btn.setFixedWidth(70)
        apply_btn.setToolTip("Write rotation to disk and clear labels")
        apply_btn.clicked.connect(self._apply)

        row = QHBoxLayout(self)
        row.setContentsMargins(6,4,6,4); row.setSpacing(6)
        lbl = QLabel("Rotate:"); lbl.setStyleSheet("color:#aaa;")
        row.addWidget(lbl); row.addWidget(self.slider, stretch=1)
        row.addWidget(self.angle_lbl); row.addWidget(reset_btn); row.addWidget(apply_btn)

        self.slider.valueChanged.connect(self._changed)

    def angle(self) -> float: return self.slider.value()/self._SCALE
    def reset(self): self.slider.setValue(0)

    def _changed(self, v):
        a = v/self._SCALE
        self.angle_lbl.setText(f"{a:.1f}°")
        if self.app: self.app.on_slider_changed(a)

    def _apply(self):
        a = self.angle()
        if self.app: self.app.apply_slider_rotation(a)
        self.slider.setValue(0)


# ── Canvas ───────────────────────────────────────────────────────────────────
class Canvas(QLabel):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.base_pix  = None
        self.scale     = 1.0
        self.fit_scale = 1.0
        self.zoom      = 1.0
        self.ox = self.oy = 0
        self.pan_x = self.pan_y = 0.0

        self.mode            = "idle"
        self.drag_start_w    = None
        self.drag_last_img   = None
        self._dlw            = None
        self.pan_start       = None
        self.pan_start_off   = None
        self.obb_rot_start   = 0.0
        self.obb_rot_mouse   = 0.0
        self.obb_handle_idx  = None
        self.box_backup      = None

    @property
    def drag_last_w(self): return self._dlw
    @drag_last_w.setter
    def drag_last_w(self, v): self._dlw = v

    # ── coords ────────────────────────────────────────────────────────
    def w2i(self, qp: QPointF):
        x=(qp.x()-self.ox)/self.scale; y=(qp.y()-self.oy)/self.scale
        if x<0 or y<0: return None
        if self.base_pix and (x>self.base_pix.width() or y>self.base_pix.height()): return None
        return x, y

    # ── render ────────────────────────────────────────────────────────
    def refresh(self):
        if self.base_pix is None: self.clear(); return
        lw,lh=max(1,self.width()),max(1,self.height())
        iw,ih=self.base_pix.width(),self.base_pix.height()
        self.fit_scale=min(lw/iw,lh/ih); self.scale=self.fit_scale*self.zoom
        dw,dh=int(iw*self.scale),int(ih*self.scale)
        self.ox=int(lw/2-dw/2+self.pan_x); self.oy=int(lh/2-dh/2+self.pan_y)
        canvas=QPixmap(lw,lh); canvas.fill(Qt.black)
        p=QPainter(canvas)
        p.drawPixmap(self.ox,self.oy,self.base_pix.scaled(dw,dh,Qt.KeepAspectRatio,Qt.SmoothTransformation))
        self.app.draw_all(p,self.scale,self.ox,self.oy)
        if self.mode=="drawing" and self.drag_start_w and self.drag_last_w:
            color=CLASS_COLORS.get(self.app.cur_cls,QColor(255,255,255))
            p.setPen(QPen(color,2,Qt.DashLine)); p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(self.drag_start_w,self.drag_last_w).normalized())
        p.end(); self.setPixmap(canvas)

    def set_image(self, pix: QPixmap):
        self.base_pix=pix; self.zoom=1.0; self.pan_x=self.pan_y=0.0; self.refresh()

    def resizeEvent(self,e): super().resizeEvent(e); self.refresh()

    # ── zoom ──────────────────────────────────────────────────────────
    def apply_zoom(self, factor, anchor=None):
        old=self.zoom; self.zoom=max(ZOOM_MIN,min(ZOOM_MAX,self.zoom*factor))
        if anchor and self.base_pix:
            ax,ay=anchor.x(),anchor.y()
            iw,ih=self.base_pix.width(),self.base_pix.height()
            old_sc=self.fit_scale*old
            old_ox=(self.width()-iw*old_sc)/2+self.pan_x
            old_oy=(self.height()-ih*old_sc)/2+self.pan_y
            imgx=(ax-old_ox)/old_sc; imgy=(ay-old_oy)/old_sc
            new_sc=self.fit_scale*self.zoom
            self.pan_x=ax-(self.width()-iw*new_sc)/2-imgx*new_sc
            self.pan_y=ay-(self.height()-ih*new_sc)/2-imgy*new_sc
        self.refresh(); self.app.update_zoom_label()

    def wheelEvent(self,e: QWheelEvent):
        if self.base_pix is None: return
        if e.modifiers() & Qt.ShiftModifier:
            self.app.rotate_selected_obb(0.5 if e.angleDelta().y()>0 else -0.5)
            e.accept(); return
        factor=1.15 if e.angleDelta().y()>0 else 1/1.15
        self.apply_zoom(factor,QPointF(e.position())); e.accept()

    # ── mouse ─────────────────────────────────────────────────────────
    def mousePressEvent(self,e):
        if self.base_pix is None or e.button() not in (Qt.LeftButton,Qt.MiddleButton): return
        wp=QPointF(e.position())
        if e.button()==Qt.MiddleButton:
            self.pan_start=wp; self.pan_start_off=(self.pan_x,self.pan_y); return
        ip=self.w2i(wp)

        # OBB rotation handle
        for i,obb in enumerate(self.app.obbs):
            hx,hy=obb.rot_handle(self.scale,self.ox,self.oy)
            if math.hypot(wp.x()-hx,wp.y()-hy)<=12:
                self.app.push_undo()
                self.app.sel=("obb",i)
                self.obb_rot_start=obb.angle
                cx_w=obb.cx*self.scale+self.ox; cy_w=obb.cy*self.scale+self.oy
                self.obb_rot_mouse=math.degrees(math.atan2(wp.y()-cy_w,wp.x()-cx_w))
                self.mode="obb_rotating"; return

        # OBB resize handles
        for i,obb in enumerate(self.app.obbs):
            for hi,(hxi,hyi) in enumerate(obb.handle_points()):
                if math.hypot(wp.x()-hxi*self.scale-self.ox, wp.y()-hyi*self.scale-self.oy)<=HANDLE_R+2:
                    self.app.push_undo(); self.app.sel=("obb",i)
                    self.obb_handle_idx=hi; self.box_backup=obb.copy()
                    self.drag_start_w=wp; self.drag_last_img=ip
                    self.mode="obb_resizing"; return

        # OBB interior (move)
        if ip:
            for i,obb in enumerate(reversed(self.app.obbs)):
                ri=len(self.app.obbs)-1-i
                if obb.contains(*ip):
                    self.app.push_undo(); self.app.sel=("obb",ri)
                    self.box_backup=obb.copy(); self.drag_last_img=ip
                    self.mode="obb_moving"; return

        # Regular box handles
        if ip:
            hit=self.app.box_handle_at(*ip)
            if hit:
                bi,hname=hit; self.app.push_undo(); self.app.sel=("box",bi)
                self.box_backup=self.app.boxes[bi].copy()
                self.drag_last_img=ip; self.obb_handle_idx=hname
                self.mode="resizing"; return
            for i in range(len(self.app.boxes)-1,-1,-1):
                if self.app.boxes[i].contains(*ip):
                    self.app.push_undo(); self.app.sel=("box",i)
                    self.box_backup=self.app.boxes[i].copy()
                    self.drag_last_img=ip; self.mode="moving"; return

        self.app.sel=None; self.drag_start_w=wp; self.drag_last_w=wp
        self.drag_last_img=ip; self.mode="drawing"; self.refresh()

    def mouseMoveEvent(self,e):
        if self.base_pix is None: return
        wp=QPointF(e.position()); ip=self.w2i(wp)
        if self.pan_start and (e.buttons()&Qt.MiddleButton):
            self.pan_x=self.pan_start_off[0]+(wp.x()-self.pan_start.x())
            self.pan_y=self.pan_start_off[1]+(wp.y()-self.pan_start.y())
            self.refresh(); return
        if self.mode=="drawing":
            self.drag_last_w=wp; self.refresh(); return
        if self.mode=="obb_rotating":
            sel=self.app.sel
            if sel and sel[0]=="obb":
                obb=self.app.obbs[sel[1]]
                cx_w=obb.cx*self.scale+self.ox; cy_w=obb.cy*self.scale+self.oy
                cur=math.degrees(math.atan2(wp.y()-cy_w,wp.x()-cx_w))
                obb.angle=self.obb_rot_start+(cur-self.obb_rot_mouse)
                self.refresh(); self.app.set_status(f"Window angle {obb.angle:.1f}°"); return
        if self.mode=="obb_resizing" and ip and self.drag_last_img:
            sel=self.app.sel
            if sel and sel[0]=="obb":
                self._obb_resize(self.app.obbs[sel[1]],ip); self.refresh(); return
        if self.mode=="obb_moving" and ip and self.drag_last_img:
            sel=self.app.sel
            if sel and sel[0]=="obb":
                obb=self.app.obbs[sel[1]]
                obb.cx+=ip[0]-self.drag_last_img[0]; obb.cy+=ip[1]-self.drag_last_img[1]
                self.drag_last_img=ip; self.refresh(); return
        if self.mode=="moving" and ip and self.drag_last_img:
            sel=self.app.sel
            if sel and sel[0]=="box":
                b=self.app.boxes[sel[1]]
                b.move(ip[0]-self.drag_last_img[0],ip[1]-self.drag_last_img[1],self.app.iw,self.app.ih)
                self.drag_last_img=ip; self.refresh(); return
        if self.mode=="resizing" and ip:
            sel=self.app.sel
            if sel and sel[0]=="box":
                b=self.app.boxes[sel[1]]; h=self.obb_handle_idx
                if h=="tl": b.x1,b.y1=ip
                elif h=="tr": b.x2,b.y1=ip
                elif h=="bl": b.x1,b.y2=ip
                elif h=="br": b.x2,b.y2=ip
                b.clamp(self.app.iw,self.app.ih); self.refresh(); return

    def _obb_resize(self,obb,ip):
        hi=self.obb_handle_idx
        if hi is None or self.box_backup is None: return
        bk=self.box_backup; rad=math.radians(bk.angle)
        dx_w=ip[0]-bk.cx; dy_w=ip[1]-bk.cy
        lx,ly=rot2d(dx_w,dy_w,-rad)
        if   hi==0: obb.w,obb.h=max(4,2*(-lx)),max(4,2*(-ly))
        elif hi==1: obb.w,obb.h=max(4,2*lx),   max(4,2*(-ly))
        elif hi==2: obb.w,obb.h=max(4,2*lx),   max(4,2*ly)
        elif hi==3: obb.w,obb.h=max(4,2*(-lx)),max(4,2*ly)
        elif hi==4: obb.h=max(4,2*(-ly))
        elif hi==5: obb.w=max(4,2*lx)
        elif hi==6: obb.h=max(4,2*ly)
        elif hi==7: obb.w=max(4,2*(-lx))
        self.drag_last_img=ip

    def mouseReleaseEvent(self,e):
        if e.button()==Qt.MiddleButton: self.pan_start=None; return
        if e.button()!=Qt.LeftButton: return
        if self.mode=="drawing":
            ip_s=self.w2i(self.drag_start_w) if self.drag_start_w else None
            ip_e=self.w2i(QPointF(e.position()))
            if ip_s and ip_e:
                x1,y1=ip_s; x2,y2=ip_e
                if self.app.cur_cls==WINDOW_CLS:
                    obb=OBB((x1+x2)/2,(y1+y2)/2,abs(x2-x1),abs(y2-y1),0.0,WINDOW_CLS)
                    if not obb.too_small():
                        self.app.push_undo(); self.app.obbs.append(obb)
                        self.app.sel=("obb",len(self.app.obbs)-1)
                        self.app.set_status("Window OBB added — drag ◉ or Shift+Scroll to rotate")
                else:
                    b=Box(self.app.cur_cls,x1,y1,x2,y2)
                    if not b.too_small():
                        self.app.push_undo(); self.app.boxes.append(b)
                        self.app.sel=("box",len(self.app.boxes)-1)
                        self.app.set_status(f"{CLASS_NAMES[self.app.cur_cls]} box added")
        self.mode="idle"
        self.drag_start_w=self.drag_last_w=self.drag_last_img=None
        self.obb_handle_idx=self.box_backup=None; self.refresh()

    def zoom_manual(self, factor):
        self.zoom=max(ZOOM_MIN,min(ZOOM_MAX,self.zoom*factor)); self.refresh()
        self.app.update_zoom_label()


# ── Main window ───────────────────────────────────────────────────────────────
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Meter Window Labeler (OBB)")
        self.resize(1350,920)

        self.img_dir=self.lbl_dir=None
        self.img_files: List[str]=[]
        self.idx=0; self.iw=self.ih=0; self.cur_img=None
        self.boxes: List[Box]=[]; self.obbs: List[OBB]=[]
        self.sel=None; self.cur_cls=1; self._undo=[]
        self._preview_angle=0.0

        # ── Canvas ────────────────────────────────────────────────────
        self.canvas=Canvas(self)

        # ── Status ────────────────────────────────────────────────────
        self.status_lbl=QLabel("Open a folder to start.")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color:#ccc;font-size:11px;")

        # ── Zoom label ────────────────────────────────────────────────
        self.zoom_lbl=QLabel("100%")
        self.zoom_lbl.setFixedWidth(55); self.zoom_lbl.setStyleSheet("color:#aaa;")

        # ── Class buttons ─────────────────────────────────────────────
        cls_row=QHBoxLayout(); cls_row.addWidget(QLabel("Class:"))
        hex_cols={0:"#00c800",1:"#ff3c3c",2:"#ffdc00",3:"#ffa000",4:"#00dcff",
                  5:"#c800ff",6:"#64ff00",7:"#0078ff",8:"#ff7878",9:"#8c00ff",
                  10:"#c8c8ff",11:"#ffc8c8",12:"#a0a0a0"}
        self._cls_btns=[]
        for cid,name in enumerate(CLASS_NAMES):
            b=QPushButton(name); b.setCheckable(True)
            col=hex_cols.get(cid,"#888")
            b.setStyleSheet(
                f"QPushButton{{background:{col};color:#000;font-weight:bold;"
                f"border:2px solid transparent;border-radius:3px;padding:2px 5px;}}"
                f"QPushButton:checked{{border:2px solid #fff;}}")
            b.clicked.connect(lambda _,c=cid: self.set_cls(c))
            cls_row.addWidget(b); self._cls_btns.append(b)
        cls_row.addStretch()

        # ── Toolbar ───────────────────────────────────────────────────
        def btn(label,slot,tip=""):
            b=QPushButton(label); b.setToolTip(tip); b.clicked.connect(slot); return b

        tb=QHBoxLayout()
        tb.addWidget(btn("📂 Open",       self.open_folder))
        tb.addWidget(btn("◀ Prev",        self.prev_img,   "P"))
        tb.addWidget(btn("▶ Next",        self.next_img,   "N"))
        tb.addWidget(btn("💾 Save",       self.save,       "Ctrl+S"))
        tb.addWidget(btn("⛔ Reject",     self.reject_img, "R"))
        sep1=QFrame(); sep1.setFrameShape(QFrame.VLine); tb.addWidget(sep1)
        tb.addWidget(btn("↶ -15°",        lambda: self.rotate_image(-15)))
        tb.addWidget(btn("↷ +15°",        lambda: self.rotate_image(15)))
        tb.addWidget(btn("↻ 90°",         lambda: self.rotate_image(90)))
        tb.addWidget(btn("✎ Custom°",     self.rotate_custom))
        sep2=QFrame(); sep2.setFrameShape(QFrame.VLine); tb.addWidget(sep2)
        tb.addWidget(btn("🔍+", lambda: self.canvas.zoom_manual(1.15)))
        tb.addWidget(self.zoom_lbl)
        tb.addWidget(btn("🔍−", lambda: self.canvas.zoom_manual(1/1.15)))
        tb.addWidget(btn("⊡",  self.zoom_reset, "Z"))
        tb.addStretch()

        hint=QLabel(
            "W=window(OBB) | M=meter | 0-9=digit | U=unknown | "
            "Drag corner/edge=resize | Drag ◉=rotate | Shift+Scroll=fine rotate | "
            "Ctrl+[/]=±1° | R=reject | ↶↷ or slider=rotate image")
        hint.setStyleSheet("background:#1a0000;color:#ff9999;padding:3px 8px;font-size:11px;")

        # ── Rotation slider ───────────────────────────────────────────
        self.rot_slider=RotationSlider(); self.rot_slider.app=self

        # ── Layout ────────────────────────────────────────────────────
        lay=QVBoxLayout()
        lay.addLayout(tb); lay.addLayout(cls_row); lay.addWidget(hint)
        lay.addWidget(self.canvas,stretch=1)
        lay.addWidget(self.rot_slider)
        lay.addWidget(self.status_lbl)
        w=QWidget(); w.setLayout(lay); self.setCentralWidget(w)

        self._setup_shortcuts(); self.set_cls(1)

    # ── Shortcuts ─────────────────────────────────────────────────────
    def _setup_shortcuts(self):
        pairs=[
            ("Ctrl+O",self.open_folder),("Ctrl+S",self.save),
            ("Ctrl+Z",self.undo),       ("N",self.next_img),
            ("P",self.prev_img),        ("Delete",self.delete_sel),
            ("Backspace",self.delete_sel),("Escape",self.deselect),
            ("Z",self.zoom_reset),      ("R",self.reject_img),
            ("Ctrl+Left",  lambda: self.rotate_image(-15)),
            ("Ctrl+Right", lambda: self.rotate_image(15)),
            ("Ctrl+R",     self.rotate_custom),
            ("M",lambda: self.set_cls(0)), ("W",lambda: self.set_cls(1)),
            ("0",lambda: self.set_cls(2)), ("1",lambda: self.set_cls(3)),
            ("2",lambda: self.set_cls(4)), ("3",lambda: self.set_cls(5)),
            ("4",lambda: self.set_cls(6)), ("5",lambda: self.set_cls(7)),
            ("6",lambda: self.set_cls(8)), ("7",lambda: self.set_cls(9)),
            ("8",lambda: self.set_cls(10)),("9",lambda: self.set_cls(11)),
            ("U",lambda: self.set_cls(12)),
            ("Ctrl+[",lambda: self.rotate_selected_obb(-1.0)),
            ("Ctrl+]",lambda: self.rotate_selected_obb(1.0)),
        ]
        for key,slot in pairs:
            a=QAction(key,self); a.setShortcut(QKeySequence(key))
            a.triggered.connect(slot); self.addAction(a)

    # ── Image rotation (same pattern as label_images.py) ──────────────
    def on_slider_changed(self, angle: float):
        """Live preview: rotate the displayed image without writing to disk."""
        self._preview_angle=angle
        if self.cur_img is None: return
        preview=(self.cur_img if abs(angle)<0.05
                 else rotate_image(self.cur_img, angle))
        rgb=cv2.cvtColor(preview,cv2.COLOR_BGR2RGB)
        h,w=preview.shape[:2]
        qi=QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888)
        self.canvas.base_pix=QPixmap.fromImage(qi.copy())
        self.canvas.refresh()

    def apply_slider_rotation(self, angle: float):
        """Write the slider rotation to disk, same as the button rotation."""
        if abs(angle)<0.05: self.set_status("No rotation to apply."); return
        self._preview_angle=0.0
        self._restore_canvas()        # put real image back first
        self.rotate_image(angle)      # then do destructive rotate

    def _restore_canvas(self):
        if self.cur_img is None: return
        rgb=cv2.cvtColor(self.cur_img,cv2.COLOR_BGR2RGB)
        h,w=self.cur_img.shape[:2]
        qi=QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888)
        self.canvas.base_pix=QPixmap.fromImage(qi.copy())
        self.canvas.refresh()

    def rotate_image(self, angle_deg: float):
        """Destructively rotate the current image file and clear its labels."""
        if not self.img_files: return
        path=os.path.join(self.img_dir,self.img_files[self.idx])
        if self.boxes or self.obbs:
            if not safe_confirm(self,"Rotate Image",
                "This image already has labels.\n\n"
                "Rotating will clear them because positions no longer match.\n\nContinue?",
                "Yes, Rotate","Cancel"):
                self.set_status("Rotation cancelled."); return
        img=cv2.imread(path)
        if img is None: self.set_status(f"Cannot read {path}"); return
        cv2.imwrite(path, rotate_image(img, angle_deg))
        self._clear_label_file(); self.load()
        self.set_status(f"Image rotated {angle_deg:+.1f}° and saved. Labels cleared.")

    def rotate_custom(self):
        angle,ok=QInputDialog.getDouble(
            self,"Rotate Image","Enter angle in degrees:",0.0,-360.0,360.0,2)
        if ok: self.rotate_image(angle)

    # ── OBB rotation ──────────────────────────────────────────────────
    def rotate_selected_obb(self, delta):
        if self.sel and self.sel[0]=="obb" and self.sel[1]<len(self.obbs):
            self.obbs[self.sel[1]].angle+=delta; self.canvas.refresh()
            self.set_status(f"Angle {self.obbs[self.sel[1]].angle:.1f}°")

    # ── Undo ──────────────────────────────────────────────────────────
    def push_undo(self):
        self._undo.append(([b.copy() for b in self.boxes],[o.copy() for o in self.obbs]))
        if len(self._undo)>200: self._undo.pop(0)

    def undo(self):
        if not self._undo: self.set_status("Nothing to undo"); return
        self.boxes,self.obbs=self._undo.pop(); self.sel=None
        self.canvas.refresh(); self.set_status("Undone")

    # ── Navigation ────────────────────────────────────────────────────
    def open_folder(self):
        folder=QFileDialog.getExistingDirectory(
            self,"Select Image Folder",os.getcwd(),
            QFileDialog.ShowDirsOnly|QFileDialog.DontUseNativeDialog)
        if not folder: return
        files=sorted(f for f in os.listdir(folder)
                     if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS)
        if not files: self.set_status("No images found."); return
        self.img_dir=folder
        parent=os.path.dirname(folder)
        self.lbl_dir=os.path.join(parent,"labels_obb")
        self.removed_img_dir=os.path.join(parent,"obb_removed_images")
        self.removed_lbl_dir=os.path.join(parent,"obb_removed_labels")
        for d in [self.lbl_dir,self.removed_img_dir,self.removed_lbl_dir]:
            os.makedirs(d,exist_ok=True)
        self.img_files=files; self.idx=0; self.load()
        self.set_status(f"Loaded {len(files)} images")

    def next_img(self): self.save(); \
        (self.idx.__setitem__(0,self.idx+1) if False else None); \
        self._nav(1)
    def prev_img(self): self._nav(-1)

    def _nav(self, d):
        self.save()
        self.idx=max(0,min(len(self.img_files)-1,self.idx+d)); self.load()

    # ── Load / Save ───────────────────────────────────────────────────
    def _lbl_path(self):
        if not self.lbl_dir or not self.img_files: return None
        stem=os.path.splitext(self.img_files[self.idx])[0]
        return os.path.join(self.lbl_dir,f"{stem}.txt")

    def _img_path(self):
        if not self.img_dir or not self.img_files: return None
        return os.path.join(self.img_dir,self.img_files[self.idx])

    def load(self):
        path=self._img_path()
        if not path: return
        img=cv2.imread(path)
        if img is None: self.set_status(f"Cannot read {path}"); return
        self.cur_img=img; self.ih,self.iw=img.shape[:2]
        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        qi=QImage(rgb.data,self.iw,self.ih,rgb.strides[0],QImage.Format_RGB888)
        self.canvas.set_image(QPixmap.fromImage(qi.copy()))
        self.boxes=[]; self.obbs=[]; self.sel=None; self._undo=[]
        self._preview_angle=0.0
        self.rot_slider.slider.blockSignals(True)
        self.rot_slider.slider.setValue(0)
        self.rot_slider.slider.blockSignals(False)
        self.rot_slider.angle_lbl.setText("0.0°")
        lp=self._lbl_path()
        if lp and os.path.exists(lp):
            with open(lp) as f:
                for line in f:
                    parts=line.strip().split()
                    if not parts: continue
                    cid=int(parts[0])
                    if cid==WINDOW_CLS and len(parts)==9:
                        self.obbs.append(OBB.from_yolo(parts,self.iw,self.ih))
                    elif len(parts)==5:
                        xc,yc,w,h=map(float,parts[1:])
                        bw,bh=w*self.iw,h*self.ih; cx,cy=xc*self.iw,yc*self.ih
                        self.boxes.append(Box(cid,cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2))
        self.canvas.refresh()
        self.set_status(f"{self.idx+1}/{len(self.img_files)}  {self.img_files[self.idx]}")

    def save(self):
        lp=self._lbl_path()
        if not lp: return
        with open(lp,"w") as f:
            for b in self.boxes:
                xc,yc,w,h=b.normalized(self.iw,self.ih)
                f.write(f"{b.cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            for o in self.obbs:
                f.write(o.to_yolo(self.iw,self.ih)+"\n")
        self.set_status("Saved ✓")

    def _clear_label_file(self):
        lp=self._lbl_path()
        if lp and os.path.exists(lp):
            open(lp,"w").close()

    def reject_img(self):
        if not self.img_files: return
        ip=self._img_path(); lp=self._lbl_path()
        try:
            if ip and os.path.exists(ip):
                shutil.move(ip,os.path.join(self.removed_img_dir,self.img_files[self.idx]))
            if lp and os.path.exists(lp):
                shutil.move(lp,os.path.join(self.removed_lbl_dir,os.path.basename(lp)))
            removed=self.img_files.pop(self.idx)
            if not self.img_files: self.set_status("All images done!"); return
            if self.idx>=len(self.img_files): self.idx=len(self.img_files)-1
            self.load(); self.set_status(f"Rejected '{removed}'")
        except Exception as ex: self.set_status(f"Reject error: {ex}")

    # ── Hit-test helpers ──────────────────────────────────────────────
    def box_handle_at(self,x,y):
        tol=max(6,9/max(self.canvas.scale,1e-6))
        for i in range(len(self.boxes)-1,-1,-1):
            b=self.boxes[i]; a,c2,d,e=b.rect()
            for name,(hx,hy) in [("tl",(a,c2)),("tr",(d,c2)),("bl",(a,e)),("br",(d,e))]:
                if abs(x-hx)<=tol and abs(y-hy)<=tol: return i,name
        return None

    # ── Drawing ───────────────────────────────────────────────────────
    def draw_all(self,p:QPainter,sc,ox,oy):
        for i,b in enumerate(self.boxes):
            a,top,c2,bot=b.rect(); color=CLASS_COLORS.get(b.cls,QColor(255,255,255))
            p.setPen(QPen(color,3 if self.sel==("box",i) else 2)); p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(a*sc+ox,top*sc+oy,(c2-a)*sc,(bot-top)*sc))
            tx,ty=int(a*sc+ox),int(top*sc+oy); tag=CLASS_NAMES[b.cls]
            p.fillRect(tx,max(0,ty-18),len(tag)*8+8,18,color)
            p.setPen(Qt.black); p.drawText(tx+4,max(13,ty-4),tag)
            if self.sel==("box",i):
                p.setBrush(QBrush(Qt.white)); p.setPen(QPen(Qt.black,1))
                for hx,hy in [(a,top),(c2,top),(a,bot),(c2,bot)]:
                    p.drawRect(QRectF(hx*sc+ox-5,hy*sc+oy-5,10,10))

        for i,obb in enumerate(self.obbs):
            color=CLASS_COLORS.get(obb.cls,QColor(255,60,60))
            is_sel=(self.sel and self.sel[0]=="obb" and self.sel[1]==i)
            p.setPen(QPen(color,3 if is_sel else 2)); p.setBrush(Qt.NoBrush)
            cw=obb.corners_w(sc,ox,oy)
            p.drawPolygon(QPolygonF([QPointF(x,y) for x,y in cw]))
            tx,ty=int(cw[0][0]),int(cw[0][1]); tag=f"window {obb.angle:.1f}°"
            p.fillRect(tx,max(0,ty-18),len(tag)*7+8,18,color)
            p.setPen(Qt.black); p.drawText(tx+4,max(13,ty-4),tag)
            p.setBrush(QBrush(Qt.white if is_sel else color)); p.setPen(QPen(Qt.black,1))
            for hxi,hyi in obb.handle_points():
                p.drawEllipse(QPointF(hxi*sc+ox,hyi*sc+oy),5,5)
            hx_w,hy_w=obb.rot_handle(sc,ox,oy)
            top_cx=(cw[0][0]+cw[1][0])/2; top_cy=(cw[0][1]+cw[1][1])/2
            p.setPen(QPen(color,1,Qt.DashLine)); p.setBrush(Qt.NoBrush)
            p.drawLine(QPointF(top_cx,top_cy),QPointF(hx_w,hy_w))
            p.setBrush(QBrush(Qt.white if is_sel else color))
            p.setPen(QPen(color,2)); p.drawEllipse(QPointF(hx_w,hy_w),9,9)
            p.setPen(QPen(Qt.black,1)); p.setBrush(Qt.NoBrush)
            p.drawText(int(hx_w)-3,int(hy_w)+5,"◉")
            if is_sel:
                lc=((cw[0][0]+cw[3][0])/2,(cw[0][1]+cw[3][1])/2)
                rc=((cw[1][0]+cw[2][0])/2,(cw[1][1]+cw[2][1])/2)
                p.setPen(QPen(QColor(255,255,100),2)); p.setBrush(Qt.NoBrush)
                p.drawLine(QPointF(*lc),QPointF(*rc))

    # ── Misc ──────────────────────────────────────────────────────────
    def delete_sel(self):
        if not self.sel: return
        self.push_undo(); kind,idx=self.sel
        if kind=="obb" and idx<len(self.obbs): del self.obbs[idx]
        elif kind=="box" and idx<len(self.boxes): del self.boxes[idx]
        self.sel=None; self.canvas.refresh(); self.set_status("Deleted")

    def deselect(self):
        self.sel=None; self.canvas.refresh()

    def set_cls(self,cid):
        self.cur_cls=cid
        for i,b in enumerate(self._cls_btns): b.setChecked(i==cid)
        self.set_status(f"Class → {CLASS_NAMES[cid]}")

    def zoom_reset(self):
        self.canvas.zoom=1.0; self.canvas.pan_x=self.canvas.pan_y=0.0
        self.canvas.refresh(); self.update_zoom_label()

    def update_zoom_label(self):
        self.zoom_lbl.setText(f"{int(self.canvas.zoom*100)}%")

    def set_status(self,msg):
        n=len(self.img_files)
        base=f"[{self.idx+1}/{n}] {self.img_files[self.idx]}  " if n else ""
        self.status_lbl.setText(
            base+msg+"  |  "
            "W=window M=meter 0-9=digit U=unknown  N/P=next/prev  "
            "S=save  R=reject  Del=delete  Ctrl+Z=undo  "
            "Scroll=zoom  Mid=pan  Z=reset  "
            "Drag◉/Shift+Scroll/Ctrl+[]=rotate OBB  "
            "↶↷=rotate image  Ctrl+R=custom  Slider=live preview")
        self.update_zoom_label()

    def next_img(self):
        self.save()
        if self.idx<len(self.img_files)-1: self.idx+=1; self.load()

    def prev_img(self):
        self.save()
        if self.idx>0: self.idx-=1; self.load()

    def closeEvent(self,e):
        try: self.save()
        except: pass
        e.accept()


def main():
    app=QApplication(sys.argv); app.setStyle("Fusion")
    win=App(); win.show(); sys.exit(app.exec())

if __name__=="__main__":
    main()