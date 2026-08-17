from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QWidget


class CompareCanvas(QWidget):
    zoomChanged = Signal(float, bool)  # zoom, fit_mode
    modeChanged = Signal(str)

    MODES = ("single", "split", "side")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(320, 240)
        self._original = QImage()
        self._edited = QImage()
        self._mode = "single"
        self._zoom = 1.0
        self._fit = True
        self._pan = QPointF(0, 0)
        self._drag_last: QPointF | None = None
        self._drag_split = False
        self._split = 0.5
        self._filename = ""

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def zoom(self) -> float:
        return self._zoom

    @property
    def fit_mode(self) -> bool:
        return self._fit

    def clear_images(self) -> None:
        self._original = QImage()
        self._edited = QImage()
        self._filename = ""
        self.update()

    def set_images(self, original_path: str | Path | None, edited_path: str | Path | None = None) -> None:
        self._original = QImage(str(original_path)) if original_path else QImage()
        self._edited = QImage(str(edited_path)) if edited_path else QImage()
        self._filename = Path(original_path).name if original_path else ""
        self._pan = QPointF(0, 0)
        self.update()

    def set_edited(self, edited_path: str | Path | None) -> None:
        self._edited = QImage(str(edited_path)) if edited_path else QImage()
        self.update()

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            return
        if self._mode != mode:
            self._mode = mode
            self.modeChanged.emit(mode)
            self.update()

    def zoom_fit(self) -> None:
        self._fit = True
        self._pan = QPointF(0, 0)
        self.zoomChanged.emit(self._zoom, True)
        self.update()

    def zoom_100(self) -> None:
        self._fit = False
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self.zoomChanged.emit(self._zoom, False)
        self.update()

    def set_zoom(self, value: float) -> None:
        self._fit = False
        self._zoom = max(0.05, min(8.0, float(value)))
        self.zoomChanged.emit(self._zoom, False)
        self.update()

    def zoom_by(self, factor: float) -> None:
        if self._fit:
            current = self._fit_scale(self._display_image(), self.rect())
            self._zoom = max(0.05, min(8.0, current * factor))
            self._fit = False
        else:
            self._zoom = max(0.05, min(8.0, self._zoom * factor))
        self.zoomChanged.emit(self._zoom, False)
        self.update()

    def _display_image(self) -> QImage:
        if not self._edited.isNull():
            return self._edited
        return self._original

    @staticmethod
    def _fit_scale(image: QImage, rect) -> float:
        if image.isNull() or image.width() <= 0 or image.height() <= 0:
            return 1.0
        return min(rect.width() / image.width(), rect.height() / image.height())

    def _image_rect(self, image: QImage, viewport: QRectF) -> QRectF:
        if image.isNull():
            return QRectF()
        scale = self._fit_scale(image, viewport) if self._fit else self._zoom
        w = image.width() * scale
        h = image.height() * scale
        cx = viewport.center().x() + self._pan.x()
        cy = viewport.center().y() + self._pan.y()
        return QRectF(cx - w / 2, cy - h / 2, w, h)

    def _draw_image(self, p: QPainter, image: QImage, viewport: QRectF) -> QRectF:
        target = self._image_rect(image, viewport)
        if not image.isNull() and not target.isNull():
            p.drawImage(target, image)
        return target

    def _draw_tag(self, p: QPainter, text: str, x: float, y: float) -> None:
        p.save()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(20, 21, 23, 210))
        box = QRectF(x, y, 70, 24)
        p.drawRoundedRect(box, 4, 4)
        p.setPen(QColor(230, 232, 235))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
        p.restore()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#151618"))
        if self._original.isNull():
            p.setPen(QColor("#8d929a"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open one or more images to begin")
            return

        bounds = QRectF(self.rect()).adjusted(10, 10, -10, -10)
        edited = self._display_image()

        if self._mode == "single" or self._edited.isNull():
            self._draw_image(p, edited, bounds)
            self._draw_tag(p, "Edited" if not self._edited.isNull() else "Original", 18, 18)
            return

        if self._mode == "side":
            gap = 6.0
            half = (bounds.width() - gap) / 2.0
            left = QRectF(bounds.left(), bounds.top(), half, bounds.height())
            right = QRectF(bounds.left() + half + gap, bounds.top(), half, bounds.height())
            self._draw_image(p, self._original, left)
            self._draw_image(p, self._edited, right)
            self._draw_tag(p, "Original", left.left() + 8, left.top() + 8)
            self._draw_tag(p, "Edited", right.left() + 8, right.top() + 8)
            p.setPen(QPen(QColor("#393b40"), 1))
            p.drawLine(QPointF(left.right() + gap / 2, bounds.top()), QPointF(left.right() + gap / 2, bounds.bottom()))
            return

        # Split comparison: original on the left, edited on the right.
        target = self._image_rect(self._edited, bounds)
        original_target = self._image_rect(self._original, bounds)
        split_x = bounds.left() + bounds.width() * self._split
        p.save()
        p.setClipRect(QRectF(bounds.left(), bounds.top(), split_x - bounds.left(), bounds.height()))
        p.drawImage(original_target, self._original)
        p.restore()
        p.save()
        p.setClipRect(QRectF(split_x, bounds.top(), bounds.right() - split_x, bounds.height()))
        p.drawImage(target, self._edited)
        p.restore()
        p.setPen(QPen(QColor("#f0f2f5"), 2))
        p.drawLine(QPointF(split_x, bounds.top()), QPointF(split_x, bounds.bottom()))
        p.setBrush(QColor("#f0f2f5"))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(split_x, bounds.center().y()), 7, 18)
        self._draw_tag(p, "Original", bounds.left() + 8, bounds.top() + 8)
        self._draw_tag(p, "Edited", bounds.right() - 78, bounds.top() + 8)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            if self._mode == "split" and not self._edited.isNull():
                split_x = self.width() * self._split
                if abs(pos.x() - split_x) <= 14:
                    self._drag_split = True
                    return
            self._drag_last = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        if self._drag_split:
            self._split = max(0.02, min(0.98, pos.x() / max(1, self.width())))
            self.update()
            return
        if self._drag_last is not None:
            delta = pos - self._drag_last
            self._pan += delta
            self._drag_last = pos
            self.update()
            return
        if self._mode == "split" and not self._edited.isNull():
            split_x = self.width() * self._split
            self.setCursor(Qt.CursorShape.SplitHCursor if abs(pos.x() - split_x) <= 14 else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_last = None
        self._drag_split = False
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta:
            self.zoom_by(1.15 if delta > 0 else 1 / 1.15)
            event.accept()
