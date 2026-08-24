from __future__ import annotations

from pathlib import Path
import math
import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor, QCursor, QDragEnterEvent, QDragMoveEvent, QDropEvent, QIcon, QImage,
    QMouseEvent, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient, QWheelEvent,
)
from PySide6.QtWidgets import (
    QAbstractScrollArea, QAbstractSpinBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
    QSizePolicy, QSlider, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

from .app_paths import resource_path


def eye_icon(size=24, crossed=False) -> QIcon:
    """Return a reliable painted eye / eye-off icon.

    Keep this intentionally simple. Earlier Bezier-path versions rendered
    inconsistently in the frozen Windows build (sometimes leaving only the
    pupil visible). An ellipse outline, pupil and optional diagonal slash use
    only basic QPainter primitives and survive Qt/Nuitka rendering reliably.
    """
    px = max(16, int(size))
    pm = QPixmap(px, px)
    pm.fill(Qt.GlobalColor.transparent)

    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    color = QColor('#d7dbe2')

    outline = QPen(color, max(1.6, px * 0.085))
    outline.setCapStyle(Qt.PenCapStyle.RoundCap)
    outline.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(outline)
    p.setBrush(Qt.BrushStyle.NoBrush)

    # A simple oval is visually unambiguous at toolbar size and avoids the
    # path-rendering problem seen in previous Windows builds.
    eye_rect = QRectF(px * 0.12, px * 0.28, px * 0.76, px * 0.44)
    p.drawEllipse(eye_rect)

    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    pupil = px * 0.20
    p.drawEllipse(QRectF(px/2-pupil/2, px/2-pupil/2, pupil, pupil))

    if crossed:
        # Dark under-stroke separates the slash from the eye outline/pupil;
        # foreground stroke then gives the usual "eye off" appearance.
        p.setBrush(Qt.BrushStyle.NoBrush)
        under = QPen(QColor('#222326'), max(3.5, px * 0.18))
        under.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(under)
        p.drawLine(QPointF(px*0.16, px*0.16), QPointF(px*0.84, px*0.84))
        slash = QPen(color, max(1.8, px * 0.09))
        slash.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(slash)
        p.drawLine(QPointF(px*0.16, px*0.16), QPointF(px*0.84, px*0.84))

    p.end()
    return QIcon(pm)




class _NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):
        # Ignore the wheel so the surrounding settings panel can scroll without
        # accidentally changing a numeric value merely because the cursor is
        # hovering over the field.
        event.ignore()


class _NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class ResettableSlider(QSlider):
    resetRequested = Signal()

    def wheelEvent(self, event):
        # Do not let the mouse wheel change a setting merely because the cursor
        # happens to be over a slider. Ignoring the event lets the surrounding
        # settings QScrollArea consume it instead, matching the numeric fields.
        event.ignore()

    def mouseDoubleClickEvent(self, event):
        if self.isEnabled() and event.button() == Qt.MouseButton.LeftButton:
            self.resetRequested.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class ResetButton(QToolButton):
    """Small reset-to-default button backed by fixed SVG assets.

    Vector assets keep the circular arrow crisp at arbitrary Windows/Qt DPI
    scaling while avoiding runtime QPainter geometry differences.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('SettingResetButton')
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(30, 30)

        icon = QIcon()
        normal = resource_path('assets', 'reset.svg')
        disabled = resource_path('assets', 'reset_disabled.svg')
        if normal.exists():
            icon.addFile(str(normal), QSize(), QIcon.Mode.Normal, QIcon.State.Off)
            icon.addFile(str(normal), QSize(), QIcon.Mode.Active, QIcon.State.Off)
            icon.addFile(str(normal), QSize(), QIcon.Mode.Selected, QIcon.State.Off)
        if disabled.exists():
            icon.addFile(str(disabled), QSize(), QIcon.Mode.Disabled, QIcon.State.Off)
        self.setIcon(icon)
        self.setIconSize(QSize(20, 20))


class NumericStepButton(QToolButton):
    """Small SVG-chevron button used by the modern numeric fields.

    Fixed vector assets keep the step arrows crisp and consistent at arbitrary
    DPI in both the development run and frozen Windows application.
    """
    def __init__(self, direction: int, parent=None):
        super().__init__(parent)
        self.direction = 1 if direction >= 0 else -1
        self.setObjectName('NumericStepUp' if self.direction > 0 else 'NumericStepDown')
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAutoRepeat(True)
        self.setAutoRepeatDelay(350)
        self.setAutoRepeatInterval(70)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        asset = 'spin_up.svg' if self.direction > 0 else 'spin_down.svg'
        icon_path = resource_path('assets', asset)
        if icon_path.exists():
            self.setIcon(QIcon(str(icon_path)))
        self.setIconSize(QSize(14, 10))


class _ModernRightControlFrame(QFrame):
    """Base for input widgets with a custom control anchored at far right.

    Layout managers are intentionally not used here. On Windows the previous
    layout-based implementation could leave spare horizontal space to the
    right of the stepper. Explicit resize geometry guarantees that the control
    column is always flush against the field's inside-right edge.
    """
    CONTROL_WIDTH = 35

    def _position_right_control(self, editor: QWidget, control: QWidget):
        border = 1
        w = max(0, self.width())
        h = max(0, self.height())
        cw = min(self.CONTROL_WIDTH, max(0, w - border * 2))
        inner_h = max(0, h - border * 2)
        control_x = max(border, w - border - cw)
        editor_w = max(0, control_x - border)
        editor.setGeometry(border, border, editor_w, inner_h)
        control.setGeometry(control_x, border, cw, inner_h)


class ModernSpinBox(_ModernRightControlFrame):
    """Integer spin box with a custom, platform-independent stepper column."""
    valueChanged = Signal(int)
    editingFinished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('ModernSpinField')
        self.setMinimumWidth(0)
        self.setFixedHeight(36)

        self._editor = _NoWheelSpinBox(self)
        self._editor.setObjectName('ModernSpinEditor')
        self._editor.setFrame(False)
        self._editor.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._editor.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._editor.setKeyboardTracking(False)
        self._editor.setCorrectionMode(QAbstractSpinBox.CorrectionMode.CorrectToNearestValue)
        if self._editor.lineEdit() is not None:
            self._editor.lineEdit().setTextMargins(0, 0, 0, 0)
        self._editor.valueChanged.connect(self.valueChanged.emit)
        self._editor.editingFinished.connect(self.editingFinished.emit)

        self._stepper = QFrame(self)
        self._stepper.setObjectName('NumericStepper')
        self._up = NumericStepButton(1, self._stepper)
        self._down = NumericStepButton(-1, self._stepper)
        self._up.clicked.connect(lambda: self._step(1))
        self._down.clicked.connect(lambda: self._step(-1))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_right_control(self._editor, self._stepper)
        h = self._stepper.height()
        top_h = h // 2
        self._up.setGeometry(0, 0, self._stepper.width(), top_h)
        self._down.setGeometry(0, top_h, self._stepper.width(), h - top_h)

    def _step(self, direction: int):
        self._editor.setValue(self._editor.value() + direction * self._editor.singleStep())

    def setRange(self, minimum: int, maximum: int): self._editor.setRange(minimum, maximum)
    def setValue(self, value: int): self._editor.setValue(int(value))
    def value(self) -> int: return int(self._editor.value())
    def setSingleStep(self, step: int): self._editor.setSingleStep(int(step))
    def singleStep(self) -> int: return int(self._editor.singleStep())
    def setSuffix(self, suffix: str): self._editor.setSuffix(str(suffix))


class ModernDoubleSpinBox(_ModernRightControlFrame):
    """Floating-point counterpart to :class:`ModernSpinBox`."""
    valueChanged = Signal(float)
    editingFinished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('ModernSpinField')
        self.setMinimumWidth(0)
        self.setFixedHeight(36)

        self._editor = _NoWheelDoubleSpinBox(self)
        self._editor.setObjectName('ModernSpinEditor')
        self._editor.setFrame(False)
        self._editor.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._editor.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._editor.setKeyboardTracking(False)
        self._editor.setCorrectionMode(QAbstractSpinBox.CorrectionMode.CorrectToNearestValue)
        if self._editor.lineEdit() is not None:
            self._editor.lineEdit().setTextMargins(0, 0, 0, 0)
        self._editor.valueChanged.connect(self.valueChanged.emit)
        self._editor.editingFinished.connect(self.editingFinished.emit)

        self._stepper = QFrame(self)
        self._stepper.setObjectName('NumericStepper')
        self._up = NumericStepButton(1, self._stepper)
        self._down = NumericStepButton(-1, self._stepper)
        self._up.clicked.connect(lambda: self._step(1))
        self._down.clicked.connect(lambda: self._step(-1))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_right_control(self._editor, self._stepper)
        h = self._stepper.height()
        top_h = h // 2
        self._up.setGeometry(0, 0, self._stepper.width(), top_h)
        self._down.setGeometry(0, top_h, self._stepper.width(), h - top_h)

    def _step(self, direction: int):
        self._editor.setValue(self._editor.value() + direction * self._editor.singleStep())

    def setRange(self, minimum: float, maximum: float): self._editor.setRange(float(minimum), float(maximum))
    def setValue(self, value: float): self._editor.setValue(float(value))
    def value(self) -> float: return float(self._editor.value())
    def setSingleStep(self, step: float): self._editor.setSingleStep(float(step))
    def singleStep(self) -> float: return float(self._editor.singleStep())
    def setDecimals(self, decimals: int): self._editor.setDecimals(int(decimals))


class DownChevronButton(QToolButton):
    """Tool button that paints the same thin chevron as settings combos.

    The button chrome remains entirely controlled by its object-name stylesheet;
    only the arrow glyph is custom-painted.  Export keeps the normal downward
    chevron, while :class:`ComboDropButton` flips it upward while its popup is
    expanded.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._chevron_up = False

    def setChevronUp(self, up: bool):
        up = bool(up)
        if self._chevron_up == up:
            return
        self._chevron_up = up
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = QColor('#eef0f3') if self.isEnabled() else QColor('#777b82')
        pen = QPen(color, 1.8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        c = self.rect().center()
        half_w = 5.0
        half_h = 3.0
        direction = -1.0 if self._chevron_up else 1.0
        path = QPainterPath()
        path.moveTo(c.x()-half_w, c.y()-direction*half_h/2)
        path.lineTo(c.x(), c.y()+direction*half_h)
        path.lineTo(c.x()+half_w, c.y()-direction*half_h/2)
        p.drawPath(path)
        p.end()


class ComboDropButton(DownChevronButton):
    """Custom dropdown chevron used by :class:`ModernComboBox`."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('ModernComboDrop')
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def setExpanded(self, expanded: bool):
        self.setChevronUp(expanded)


class _NoWheelComboBox(QComboBox):
    """Combo box that ignores hover-wheel changes and reports popup state."""
    popupShown = Signal()
    popupHidden = Signal()

    def wheelEvent(self, event):
        # Let the settings scroll area consume the wheel instead.
        event.ignore()

    def showPopup(self):
        super().showPopup()
        self.popupShown.emit()

    def hidePopup(self):
        super().hidePopup()
        self.popupHidden.emit()


class ModernComboBox(_ModernRightControlFrame):
    """Combo box with a custom right-edge drop button and intact outer border.

    The custom button behaves like a real combo arrow: opening the popup flips
    the chevron upward, and clicking the same button while the popup is open
    closes it.  Qt closes a popup before redispatching an outside click to the
    underlying button, so a small suppression flag prevents that redispatched
    click from immediately reopening the popup.
    """
    currentIndexChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('ModernComboField')
        self.setMinimumWidth(0)
        self.setMinimumHeight(34)
        self.setMaximumHeight(36)
        self._popup_open = False
        self._suppress_next_drop_click = False

        self._editor = _NoWheelComboBox(self)
        self._editor.setObjectName('ModernComboEditor')
        self._editor.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._editor.currentIndexChanged.connect(self.currentIndexChanged.emit)
        self._editor.popupShown.connect(self._popup_shown)
        self._editor.popupHidden.connect(self._popup_hidden)

        self._drop = ComboDropButton(self)
        self._drop.clicked.connect(self._toggle_popup)

    def _popup_shown(self):
        self._popup_open = True
        self._suppress_next_drop_click = False
        self._drop.setExpanded(True)

    def _popup_hidden(self):
        was_open = self._popup_open
        self._popup_open = False
        self._drop.setExpanded(False)

        # A combo popup owns the mouse while it is visible. Clicking our custom
        # arrow therefore first closes the popup, then Qt redispatches the same
        # click to the arrow button. Remember that close so the redispatched
        # click is consumed instead of reopening the popup.
        if was_open and self._drop.rect().contains(self._drop.mapFromGlobal(QCursor.pos())):
            self._suppress_next_drop_click = True
            # Keep the guard long enough to consume Qt's replayed click, but do
            # not let a popup closed by Escape leave the next deliberate click
            # suppressed just because the pointer happened to rest on the arrow.
            QTimer.singleShot(200, self._clear_drop_click_suppression)

    def _clear_drop_click_suppression(self):
        self._suppress_next_drop_click = False

    def _toggle_popup(self):
        if self._suppress_next_drop_click:
            self._suppress_next_drop_click = False
            return
        if self._popup_open:
            self._editor.hidePopup()
        else:
            self._editor.showPopup()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_right_control(self._editor, self._drop)

    def addItem(self, text, userData=None): self._editor.addItem(text, userData)
    def currentData(self): return self._editor.currentData()
    def currentText(self): return self._editor.currentText()
    def findData(self, data): return self._editor.findData(data)
    def setCurrentIndex(self, index: int): self._editor.setCurrentIndex(index)
    def currentIndex(self) -> int: return self._editor.currentIndex()
    def count(self) -> int: return self._editor.count()
    def clear(self): self._editor.clear()
    def itemData(self, index: int): return self._editor.itemData(index)
    def itemText(self, index: int) -> str: return self._editor.itemText(index)


class EyeToggleButton(QToolButton):
    """Eye button whose original-view slash is painted over the final widget.

    Frozen Qt builds can cache/transform QIcon state pixmaps in ways that make
    a crossed variant unreliable.  Keeping one normal eye icon and painting
    the slash *after* the button/icon has been rendered makes the state
    deterministic: checked = Edited/open eye; unchecked = Original/slashed eye.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setIcon(eye_icon(20, False))
        self.setIconSize(QSize(20, 20))

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.isChecked():
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = self.rect()
        cx = r.center().x()
        cy = r.center().y()
        half = min(10.0, max(7.0, min(r.width(), r.height()) * 0.31))
        # Dark separator plus a bright foreground stroke keeps the slash visible
        # over both the eye outline and the pupil in enabled/disabled states.
        under = QPen(QColor('#1f2023'), 4.2)
        under.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(under)
        p.drawLine(QPointF(cx-half, cy-half), QPointF(cx+half, cy+half))
        fg = QColor('#d7dbe2') if self.isEnabled() else QColor('#7b7f86')
        slash = QPen(fg, 2.2)
        slash.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(slash)
        p.drawLine(QPointF(cx-half, cy-half), QPointF(cx+half, cy+half))
        p.end()




class ClipStripItem:
    """Small data object used by ImageStrip.

    This deliberately mirrors the tiny subset of QListWidgetItem used by the
    main window, without involving QListView's internal item-grid geometry.
    """

    def __init__(self, text=''):
        self._text = str(text)
        self._data = {Qt.ItemDataRole.DisplayRole: self._text}
        self._tooltip = ''
        self._selected = False
        self._processing = False
        self._thumbnail = QPixmap()
        self._strip = None
        self._tile = None

    def text(self):
        return self._text

    def setData(self, role, value):
        self._data[role] = value
        if role == Qt.ItemDataRole.DisplayRole:
            self._text = str(value)
        if self._tile is not None:
            self._tile.update()

    def data(self, role):
        if role == Qt.ItemDataRole.DisplayRole:
            return self._text
        return self._data.get(role)

    def setToolTip(self, text):
        self._tooltip = str(text)
        if self._tile is not None:
            self._tile.setToolTip(self._tooltip)

    def toolTip(self):
        return self._tooltip

    def setThumbnail(self, pixmap):
        self._thumbnail = QPixmap(pixmap)
        if self._tile is not None:
            self._tile.update()

    def thumbnail(self):
        return self._thumbnail

    def setProcessing(self, processing):
        processing = bool(processing)
        if self._processing == processing:
            return
        self._processing = processing
        if self._tile is not None:
            self._tile.setProcessing(processing)

    def isProcessing(self):
        return self._processing

    def isSelected(self):
        return self._selected

    def setSelected(self, selected):
        selected = bool(selected)
        if self._strip is not None:
            self._strip._set_item_selected(self, selected)
        else:
            self._selected = selected


class _ClipTile(QWidget):
    """One fixed 132x92 thumbnail + filename tile."""

    def __init__(self, strip, item):
        super().__init__(strip.viewport())
        self.strip = strip
        self.item = item
        self.setFixedSize(strip.TILE_WIDTH, strip.TILE_HEIGHT)
        self.setMouseTracking(True)
        self._processing_angle = 0
        self._processing_timer = QTimer(self)
        self._processing_timer.setInterval(70)
        self._processing_timer.timeout.connect(self._advance_processing_spinner)
        self.setProcessing(item.isProcessing())
        if item.toolTip():
            self.setToolTip(item.toolTip())

    def setProcessing(self, processing):
        if processing:
            if not self._processing_timer.isActive():
                self._processing_angle = 0
                self._processing_timer.start()
        else:
            self._processing_timer.stop()
        self.update()

    def _advance_processing_spinner(self):
        self._processing_angle = (self._processing_angle + 30) % 360
        self.update()

    def mousePressEvent(self, event):
        self.strip._tile_mouse_press(self.item, event)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        r = self.rect()
        p.fillRect(r, QColor('#252629'))

        if self.item.isSelected():
            # Keep the selection frame one physical pixel inside the tile.
            # A QPen stroke centered on x=0/y=0 loses half of its top/left
            # stroke to QWidget clipping on Windows. Filled strips avoid that
            # half-pixel alignment problem completely and stay crisp at DPI
            # scaling.
            sel = r.adjusted(1, 1, -2, -2)
            p.fillRect(sel, QColor('#2d3744'))
            c = QColor('#4d96e8')
            t = 1
            p.fillRect(sel.x(), sel.y(), sel.width(), t, c)
            p.fillRect(sel.x(), sel.y() + sel.height() - t, sel.width(), t, c)
            p.fillRect(sel.x(), sel.y(), t, sel.height(), c)
            p.fillRect(sel.x() + sel.width() - t, sel.y(), t, sel.height(), c)

        fm = p.fontMetrics()
        text_h = fm.height()
        content_h = self.strip.THUMB_HEIGHT + self.strip.TEXT_GAP + text_h
        content_top = max(0, (self.height() - content_h) // 2)
        icon_x = (self.width() - self.strip.THUMB_WIDTH) // 2
        icon_y = content_top

        thumb = self.item.thumbnail()
        if not thumb.isNull():
            p.drawPixmap(icon_x, icon_y, thumb)

        if self.item.isProcessing():
            # Match the canvas ProcessingOverlay styling, scaled down to the
            # thumbnail: the image stays identifiable underneath a neutral
            # gray veil while the same 12-spoke spinner animation runs above.
            p.fillRect(
                icon_x, icon_y, self.strip.THUMB_WIDTH, self.strip.THUMB_HEIGHT,
                QColor(42, 43, 47, 210),
            )
            p.save()
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.translate(
                icon_x + self.strip.THUMB_WIDTH / 2,
                icon_y + self.strip.THUMB_HEIGHT / 2,
            )
            p.rotate(self._processing_angle)
            for i in range(12):
                color = QColor('#e5e8ed')
                color.setAlpha(45 + int(210 * (i + 1) / 12))
                pen = QPen(color, 2.1)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                p.drawLine(QPointF(0, -8.5), QPointF(0, -12.5))
                p.rotate(30)
            p.restore()

        # The current/canvas marker is four filled strips in unscaled widget
        # coordinates. There is no QIcon rescaling, pen alignment, or delegate
        # clipping involved, so all four red sides remain intact at any DPI.
        if self.strip.currentItem() is self.item:
            c = QColor('#ff3b30')
            t = 2
            w = self.strip.THUMB_WIDTH
            h = self.strip.THUMB_HEIGHT
            p.fillRect(icon_x, icon_y, w, t, c)
            p.fillRect(icon_x, icon_y + h - t, w, t, c)
            p.fillRect(icon_x, icon_y, t, h, c)
            p.fillRect(icon_x + w - t, icon_y, t, h, c)

        text_y = icon_y + self.strip.THUMB_HEIGHT + self.strip.TEXT_GAP
        text_w = self.width() - 12
        text = fm.elidedText(self.item.text(), Qt.TextElideMode.ElideRight, text_w)
        p.setPen(QColor('#f1f2f4') if self.isEnabled() else QColor('#777b82'))
        p.drawText(6, text_y, text_w, text_h,
                   Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, text)
        p.end()


class ImageStrip(QAbstractScrollArea):
    """Deterministic horizontal clip strip.

    QListWidget/QListView was a poor fit for this UI because its internal icon
    grid, viewport margins, selection delegate and scrollbar layout all affect
    item geometry independently. This strip positions fixed-size child widgets
    directly in the viewport, so vertical centering is exactly
    ``(viewport_height - tile_height) / 2`` and does not depend on Qt's item
    layout heuristics.
    """

    currentItemChanged = Signal(object, object)
    itemSelectionChanged = Signal()
    emptyContextRequested = Signal(QPoint)
    itemContextRequested = Signal(QPoint)
    droppedPaths = Signal(list)

    TILE_WIDTH = 132
    TILE_HEIGHT = 92
    THUMB_WIDTH = 72
    THUMB_HEIGHT = 54
    TEXT_GAP = 4
    SIDE_INSET = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items = []
        self._current = None
        self._anchor_index = None
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.viewport().setStyleSheet('background:#252629; border:none;')
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.horizontalScrollBar().valueChanged.connect(self._layout_tiles)

        self._empty_label = QLabel('Open images to begin', self.viewport())
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet('color:#777b82; background:transparent; border:none;')
        self._empty_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._update_layout()

    def count(self):
        return len(self._items)

    def item(self, row):
        return self._items[row] if 0 <= row < len(self._items) else None

    def row(self, item):
        try:
            return self._items.index(item)
        except ValueError:
            return -1

    def addItem(self, item):
        if item in self._items:
            return
        item._strip = self
        tile = _ClipTile(self, item)
        item._tile = tile
        self._items.append(item)
        tile.show()
        self._update_layout()

    def takeItem(self, row):
        if not (0 <= row < len(self._items)):
            return None
        item = self._items[row]
        old_current = self._current
        was_selected = item._selected
        if item._tile is not None:
            item._tile.hide()
            item._tile.deleteLater()
        item._tile = None
        item._strip = None
        item._selected = False
        del self._items[row]

        if old_current is item:
            self._current = self._items[min(row, len(self._items)-1)] if self._items else None
            self.currentItemChanged.emit(self._current, old_current)
        if was_selected:
            self.itemSelectionChanged.emit()
        self._anchor_index = None if not self._items else min(row, len(self._items)-1)
        self._update_layout()
        return item

    def clear(self):
        if not self._items:
            return
        old_current = self._current
        had_selection = any(i._selected for i in self._items)
        for item in self._items:
            if item._tile is not None:
                item._tile.hide()
                item._tile.deleteLater()
            item._tile = None
            item._strip = None
            item._selected = False
        self._items.clear()
        self._current = None
        self._anchor_index = None
        self.horizontalScrollBar().setValue(0)
        self._update_layout()
        if old_current is not None:
            self.currentItemChanged.emit(None, old_current)
        if had_selection:
            self.itemSelectionChanged.emit()

    def selectedItems(self):
        return [i for i in self._items if i._selected]

    def currentItem(self):
        return self._current

    def setCurrentItem(self, item, _flags=None):
        if item is not None and item not in self._items:
            return
        if item is self._current:
            self._ensure_item_visible(item)
            return
        prev = self._current
        self._current = item
        if prev is not None and prev._tile is not None:
            prev._tile.update()
        if item is not None and item._tile is not None:
            item._tile.update()
        self._ensure_item_visible(item)
        self.currentItemChanged.emit(item, prev)

    def _set_item_selected(self, item, selected, emit=True):
        if item not in self._items or item._selected == bool(selected):
            return
        item._selected = bool(selected)
        if item._tile is not None:
            item._tile.update()
        if emit:
            self.itemSelectionChanged.emit()

    def clearSelection(self):
        changed = False
        for item in self._items:
            if item._selected:
                item._selected = False
                changed = True
                if item._tile is not None:
                    item._tile.update()
        if changed:
            self.itemSelectionChanged.emit()

    def selectAll(self):
        changed = False
        for item in self._items:
            if not item._selected:
                item._selected = True
                changed = True
                if item._tile is not None:
                    item._tile.update()
        if changed:
            self.itemSelectionChanged.emit()

    def _tile_mouse_press(self, item, event):
        idx = self.row(item)
        if idx < 0:
            return

        if event.button() == Qt.MouseButton.RightButton:
            if not item._selected:
                self.clearSelection()
                self._set_item_selected(item, True)
            self.setCurrentItem(item)
            self._anchor_index = idx
            self.itemContextRequested.emit(event.globalPosition().toPoint())
            event.accept()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return

        mods = event.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)

        if shift and self._items:
            anchor = self._anchor_index
            if anchor is None:
                anchor = self.row(self._current) if self._current in self._items else idx
            lo, hi = sorted((max(0, anchor), idx))
            if not ctrl:
                for it in self._items:
                    it._selected = False
            for n in range(lo, hi + 1):
                self._items[n]._selected = True
            for it in self._items:
                if it._tile is not None:
                    it._tile.update()
            self.itemSelectionChanged.emit()
        elif ctrl:
            self._set_item_selected(item, not item._selected)
            self._anchor_index = idx
        else:
            changed = any(it._selected != (it is item) for it in self._items)
            for it in self._items:
                it._selected = (it is item)
                if it._tile is not None:
                    it._tile.update()
            if changed:
                self.itemSelectionChanged.emit()
            self._anchor_index = idx

        self.setCurrentItem(item)
        event.accept()

    def mousePressEvent(self, event):
        # Events reaching the scroll-area viewport are clicks on empty strip
        # space; tile clicks are handled by _ClipTile above.
        if event.button() == Qt.MouseButton.RightButton:
            self.emptyContextRequested.emit(event.globalPosition().toPoint())
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and not (
            event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
        ):
            self.clearSelection()
            event.accept()
            return
        super().mousePressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_layout()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._update_layout)

    def _update_scroll_range(self):
        content_w = 2 * self.SIDE_INSET + len(self._items) * self.TILE_WIDTH
        viewport_w = max(1, self.viewport().width())
        bar = self.horizontalScrollBar()
        bar.setPageStep(viewport_w)
        bar.setSingleStep(self.TILE_WIDTH)
        bar.setRange(0, max(0, content_w - viewport_w))

    def _update_layout(self):
        self._update_scroll_range()
        self._layout_tiles()
        self._empty_label.setGeometry(self.viewport().rect())
        self._empty_label.setVisible(not self._items)
        self._empty_label.raise_()

    def _layout_tiles(self):
        vp = self.viewport()
        y = max(0, (vp.height() - self.TILE_HEIGHT) // 2)
        scroll_x = self.horizontalScrollBar().value()
        for idx, item in enumerate(self._items):
            if item._tile is None:
                continue
            x = self.SIDE_INSET + idx * self.TILE_WIDTH - scroll_x
            item._tile.setGeometry(x, y, self.TILE_WIDTH, self.TILE_HEIGHT)
        if hasattr(self, '_empty_label'):
            self._empty_label.setGeometry(vp.rect())

    def _ensure_item_visible(self, item):
        if item is None or item not in self._items:
            return
        idx = self.row(item)
        left = self.SIDE_INSET + idx * self.TILE_WIDTH
        right = left + self.TILE_WIDTH
        bar = self.horizontalScrollBar()
        value = bar.value()
        width = self.viewport().width()
        if left < value:
            bar.setValue(left)
        elif right > value + width:
            bar.setValue(max(0, right - width))

    @staticmethod
    def _local_paths_from_mime(mime):
        if not mime or not mime.hasUrls():
            return []
        paths = []
        for url in mime.urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if path:
                    paths.append(path)
        return paths

    def dragEnterEvent(self, event: QDragEnterEvent):
        if self._local_paths_from_mime(event.mimeData()):
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent):
        if self._local_paths_from_mime(event.mimeData()):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        paths = self._local_paths_from_mime(event.mimeData())
        if paths:
            self.droppedPaths.emit(paths)
            event.acceptProposedAction()
            return
        event.ignore()


class ComparisonCanvas(QWidget):
    zoomChanged = Signal(float)
    splitChanged = Signal(float)
    droppedPaths = Signal(list)
    maskEdited = Signal(str, object, bool)  # stage, QImage, authored

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(500,350)
        self.setAcceptDrops(True)
        self.original=QImage(); self.edited=QImage(); self.mode='single'; self.show_edited=True
        self.zoom=0.0; self.pan=QPointF(); self.split=0.5; self._panning=False; self._split_drag=False; self._last=QPointF()
        self._mask_stage: str|None=None
        self._mask=QImage()
        self._mask_authored=False
        self._mask_tool='eraser'
        self._mask_size=120
        self._mask_feather=24
        self._mask_opacity=100
        self._mask_painting=False
        self._mask_last_image: QPointF|None=None
        self._mask_hover: QPointF|None=None
        # Eraser opacity is evaluated from a snapshot taken at mouse-down so
        # overlapping stamps within one drag do not repeatedly erase the same
        # pixel. Brush painting uses a per-pixel maximum and therefore also
        # never builds opacity merely because a stroke overlaps itself.
        self._mask_stroke_base=QImage()
        self._mask_eraser_coverage=None
        self.setCursor(Qt.CursorShape.ArrowCursor)

    @property
    def mask_stage(self):
        return self._mask_stage

    def set_images(self, original: QImage|None, edited: QImage|None):
        self.original=original or QImage(); self.edited=edited or QImage(); self.zoom=0.0; self.pan=QPointF()
        if self._mask_stage is not None:
            self._ensure_mask_size()
        self.update()

    def set_mode(self, mode):
        self.mode=mode; self.pan=self._constrained_pan(self.pan); self.update()

    def set_single_edited(self, value): self.show_edited=bool(value); self.update()
    def fit(self): self.zoom=0.0; self.pan=QPointF(); self.zoomChanged.emit(0.0); self.update()
    def zoom_100(self): self.zoom=1.0; self.pan=self._constrained_pan(QPointF()); self.zoomChanged.emit(1.0); self.update()
    def set_zoom(self,z):
        self.zoom=max(.05,min(8.0,float(z))); self.pan=self._constrained_pan(self.pan); self.zoomChanged.emit(self.zoom); self.update()

    def set_mask_mode(self, stage: str|None, mask: QImage|None=None, authored: bool=False):
        self._mask_painting=False
        self._mask_last_image=None
        self._mask_hover=None
        self._mask_stroke_base=QImage()
        self._mask_eraser_coverage=None
        self._mask_stage=stage
        self._mask_authored=bool(authored)
        if stage is None:
            self._mask=QImage()
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
            return
        if isinstance(mask,QImage) and not mask.isNull():
            # RGBA8888 has a stable byte layout, which lets the brush update
            # alpha directly without SourceOver opacity accumulation.
            self._mask=mask.convertToFormat(QImage.Format.Format_RGBA8888).copy()
        else:
            self._mask=QImage()
        self._ensure_mask_size()
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def set_mask_brush(self, tool: str, size_px: int, feather_px: int, opacity: int):
        self._mask_tool='eraser' if str(tool).lower()=='eraser' else 'brush'
        self._mask_size=max(1,int(size_px))
        self._mask_feather=max(0,int(feather_px))
        self._mask_opacity=max(1,min(100,int(opacity)))
        self.update()

    def current_mask(self):
        """Return a copy of the currently edited mask and its authored state."""
        if self._mask_stage is None:
            return QImage(),False
        self._ensure_mask_size()
        return self._mask.copy(),bool(self._mask_authored)

    def replace_current_mask(self, mask: QImage|None, authored: bool=True):
        """Replace the active stage mask without changing brush/eraser settings."""
        if self._mask_stage is None:
            return
        self._mask_painting=False
        self._mask_last_image=None
        self._end_mask_stroke()
        if isinstance(mask,QImage) and not mask.isNull():
            self._mask=mask.convertToFormat(QImage.Format.Format_RGBA8888).copy()
        else:
            self._mask=QImage()
        self._ensure_mask_size()
        self._mask_authored=bool(authored)
        self.update()
        self.maskEdited.emit(self._mask_stage,self._mask.copy(),self._mask_authored)

    def invert_current_mask(self):
        """Invert active mask alpha (alpha -> 255-alpha) and author the result."""
        if self._mask_stage is None:
            return
        self._mask_painting=False
        self._mask_last_image=None
        self._end_mask_stroke()
        self._ensure_mask_size()
        rgba=self._mask_rgba_view(self._mask)
        if rgba is None:
            return
        rgba[...,3]=255-rgba[...,3]
        visible=rgba[...,3]>0
        rgba[...,0][visible]=255
        rgba[...,1][visible]=32
        rgba[...,2][visible]=32
        self._mask_authored=True
        self.update()
        self.maskEdited.emit(self._mask_stage,self._mask.copy(),True)

    def _ensure_mask_size(self):
        if self.original.isNull():
            return
        size=self.original.size()
        if not self._mask.isNull() and self._mask.size()==size:
            return
        replacement=QImage(size,QImage.Format.Format_RGBA8888)
        replacement.fill(Qt.GlobalColor.transparent)
        if not self._mask.isNull():
            p=QPainter(replacement)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform,True)
            p.drawImage(QRectF(0,0,size.width(),size.height()),self._mask)
            p.end()
        self._mask=replacement

    def _content_area(self): return QRectF(self.rect()).adjusted(8,8,-8,-8)
    def _side_areas(self):
        area=self._content_area(); half=area.width()/2.0
        return (QRectF(area.left(),area.top(),half,area.height()), QRectF(area.left()+half,area.top(),half,area.height()))
    def _active(self):
        if self.mode=='single' and self.show_edited and not self.edited.isNull(): return self.edited
        return self.original
    def _reference_area(self):
        if self.mode=='side' and not self.edited.isNull(): return self._side_areas()[0]
        return self._content_area()
    def _base_scale(self,img,area):
        if img.isNull() or area.width()<=0 or area.height()<=0: return 1.0
        return min(area.width()/img.width(), area.height()/img.height())
    def _effective_scale(self,img,area): return self._base_scale(img,area) if self.zoom<=0 else self.zoom
    def _pan_limits(self,img,area):
        if img.isNull(): return (0.0,0.0)
        scale=self._effective_scale(img,area); w=img.width()*scale; h=img.height()*scale
        return (max(0.0,(w-area.width())/2.0), max(0.0,(h-area.height())/2.0))
    def _constrained_pan(self,requested):
        img=self._active(); area=self._reference_area(); mx,my=self._pan_limits(img,area)
        return QPointF(max(-mx,min(mx,requested.x())), max(-my,min(my,requested.y())))
    def _rect(self,img,area):
        scale=self._effective_scale(img,area); w=img.width()*scale; h=img.height()*scale
        c=area.center()+self.pan
        return QRectF(c.x()-w/2,c.y()-h/2,w,h)
    def _paint_image(self,p,img,area,clip=True):
        if img.isNull(): return
        r=self._rect(img,area)
        if clip: p.save(); p.setClipRect(area)
        p.drawImage(r,img)
        if clip: p.restore()

    def _image_point_from_widget(self, pos: QPointF) -> QPointF|None:
        if self._mask_stage is None or self.original.isNull() or self.mode!='single':
            return None
        img=self._active()
        if img.isNull():
            img=self.original
        area=self._content_area()
        target=self._rect(img,area)
        if not target.contains(pos) or target.width()<=0 or target.height()<=0:
            return None
        self._ensure_mask_size()
        mw=max(1,self._mask.width()); mh=max(1,self._mask.height())
        x=(pos.x()-target.left())/target.width()*mw
        y=(pos.y()-target.top())/target.height()*mh
        return QPointF(max(0.0,min(mw-1.0,x)),max(0.0,min(mh-1.0,y)))

    @staticmethod
    def _mask_rgba_view(image: QImage):
        """Return a writable HxWx4 uint8 view for an RGBA8888 QImage."""
        if image.isNull():
            return None
        if image.format()!=QImage.Format.Format_RGBA8888:
            raise ValueError('Mask image must use RGBA8888 format')
        h=image.height(); w=image.width(); stride=image.bytesPerLine()
        raw=np.frombuffer(image.bits(),dtype=np.uint8,count=image.sizeInBytes())
        rows=raw.reshape((h,stride))
        return rows[:,:w*4].reshape((h,w,4))

    def _begin_mask_stroke(self):
        self._mask_stroke_base=QImage()
        self._mask_eraser_coverage=None
        if self._mask_tool!='eraser' or self._mask.isNull():
            return
        # Eraser strength is relative to the mask as it existed at mouse-down.
        # A 50% eraser therefore removes 50% once over the whole drag, even if
        # the cursor crosses the same pixel many times during that drag.
        self._mask_stroke_base=self._mask.copy()
        self._mask_eraser_coverage=np.zeros((self._mask.height(),self._mask.width()),dtype=np.uint8)

    def _end_mask_stroke(self):
        self._mask_stroke_base=QImage()
        self._mask_eraser_coverage=None

    def _mask_brush_radii(self):
        # Brush Size is the diameter of the fully opaque/selected core.
        # Feather is an absolute source-image-pixel falloff *outside* that
        # core, so a 100 px feather is always 100 px regardless of Size.
        core_radius=max(0.5,float(self._mask_size)*0.5)
        feather=max(0.0,float(self._mask_feather))
        return core_radius,feather,core_radius+feather

    def _apply_mask_capsule(self, start: QPointF, end: QPointF):
        self._ensure_mask_size()
        if self._mask.isNull():
            return

        core_radius,feather,outer_radius=self._mask_brush_radii()
        strength=max(1,min(255,int(round(255.0*self._mask_opacity/100.0))))

        # Rasterize the complete mouse segment as one capsule instead of a
        # chain of overlapping circular stamps. Besides being faster for very
        # large absolute feathers (up to 1000 px), this gives a continuous
        # hard core even when the feather is much larger than the brush Size.
        ax=float(start.x()); ay=float(start.y())
        bx=float(end.x()); by=float(end.y())
        x0=max(0,int(math.floor(min(ax,bx)-outer_radius-1.0)))
        y0=max(0,int(math.floor(min(ay,by)-outer_radius-1.0)))
        x1=min(self._mask.width(),int(math.ceil(max(ax,bx)+outer_radius+1.0))+1)
        y1=min(self._mask.height(),int(math.ceil(max(ay,by)+outer_radius+1.0))+1)
        if x1<=x0 or y1<=y0:
            return

        yy,xx=np.ogrid[y0:y1,x0:x1]
        dx=bx-ax; dy=by-ay; seg_len2=dx*dx+dy*dy
        if seg_len2<=1e-12:
            dist=np.sqrt((xx-ax)**2+(yy-ay)**2)
        else:
            t=np.clip(((xx-ax)*dx+(yy-ay)*dy)/seg_len2,0.0,1.0)
            nearest_x=ax+t*dx
            nearest_y=ay+t*dy
            dist=np.sqrt((xx-nearest_x)**2+(yy-nearest_y)**2)

        if feather<=0.01:
            weight=(dist<=core_radius).astype(np.float32)
        else:
            weight=np.clip((outer_radius-dist)/feather,0.0,1.0).astype(np.float32)
            weight[dist<=core_radius]=1.0
        target=np.rint(weight*strength).astype(np.uint8)

        rgba=self._mask_rgba_view(self._mask)
        if rgba is None:
            return
        dst=rgba[y0:y1,x0:x1]

        if self._mask_tool=='eraser':
            if self._mask_stroke_base.isNull() or self._mask_eraser_coverage is None:
                self._begin_mask_stroke()
            base=self._mask_rgba_view(self._mask_stroke_base)
            if base is None or self._mask_eraser_coverage is None:
                return
            coverage=self._mask_eraser_coverage[y0:y1,x0:x1]
            np.maximum(coverage,target,out=coverage)
            base_alpha=base[y0:y1,x0:x1,3].astype(np.uint16)
            remain=(255-coverage.astype(np.uint16))
            dst[...,3]=((base_alpha*remain+127)//255).astype(np.uint8)
        else:
            # Brush opacity is a target mask opacity, not paint volume:
            # 50% over 20% becomes 50%; 20% over 50% stays 50%. Repeated
            # overlap cannot build past the slider's requested opacity unless
            # a stronger brush is used.
            np.maximum(dst[...,3],target,out=dst[...,3])
            touched=target>0
            dst[...,0][touched]=255
            dst[...,1][touched]=32
            dst[...,2][touched]=32

    def _mask_stamp(self, center: QPointF):
        self._apply_mask_capsule(center,center)

    def _paint_mask_segment(self, start: QPointF, end: QPointF):
        self._apply_mask_capsule(start,end)
        self._mask_authored=True
        self.update()

    def _paint_mask_overlay(self,p:QPainter,img:QImage,area:QRectF):
        if self._mask_stage is None or self._mask.isNull() or img.isNull():
            return
        # Keep the mask readable without hiding the image underneath. The
        # stored alpha remains the real processing mask; only its on-screen red
        # visualization is scaled, so 100% mask alpha displays as 70% red,
        # 50% as 40%, and 0% remains invisible.
        p.save(); p.setClipRect(area); p.setOpacity(0.70); p.drawImage(self._rect(img,area),self._mask); p.restore()
        if self._mask_hover is not None:
            image_pt=self._image_point_from_widget(self._mask_hover)
            if image_pt is not None:
                target=self._rect(img,area)
                scale=target.width()/max(1.0,float(self._mask.width()))
                core_radius,feather,outer_radius=self._mask_brush_radii()
                core=max(1.0,core_radius*scale)
                outer=max(core,outer_radius*scale)
                pen=QPen(QColor(255,245,245,220),1.25)
                pen.setCosmetic(True)
                p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(self._mask_hover,core,core)
                if feather>0.01 and outer>core+0.5:
                    outer_pen=QPen(QColor(255,245,245,145),1.0,Qt.PenStyle.DashLine)
                    outer_pen.setCosmetic(True)
                    p.setPen(outer_pen)
                    p.drawEllipse(self._mask_hover,outer,outer)

    def paintEvent(self,_):
        p=QPainter(self); p.fillRect(self.rect(),QColor('#151618')); area=self._content_area()
        if self.original.isNull():
            p.setPen(QColor('#777b82')); p.drawText(area,Qt.AlignmentFlag.AlignCenter,'Open images to begin'); return
        if self.mode=='side' and not self.edited.isNull():
            a,b=self._side_areas(); self._paint_image(p,self.original,a,True); self._paint_image(p,self.edited,b,True)
            divider=a.right(); p.setPen(QPen(QColor('#3b3d42'),1)); p.drawLine(int(divider),int(area.top()),int(divider),int(area.bottom()))
            p.setPen(QColor('#a7abb2')); p.drawText(a.adjusted(10,10,-10,-10),Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop,'Original'); p.drawText(b.adjusted(10,10,-10,-10),Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop,'Edited'); return
        if self.mode=='split' and not self.edited.isNull():
            self._paint_image(p,self.original,area,True)
            x=area.left()+area.width()*self.split; clip=QRectF(x,area.top(),area.right()-x,area.height())
            p.save(); p.setClipRect(clip); p.drawImage(self._rect(self.edited,area),self.edited); p.restore()
            p.setPen(QPen(QColor('#f0f2f5'),2)); p.drawLine(int(x),int(area.top()),int(x),int(area.bottom())); p.setBrush(QColor('#f0f2f5')); p.drawEllipse(QPointF(x,area.center().y()),7,7); return
        img=self.edited if self.show_edited and not self.edited.isNull() else self.original
        self._paint_image(p,img,area,True)
        self._paint_mask_overlay(p,img,area)
        p.setPen(QColor('#a7abb2')); p.drawText(area.adjusted(10,10,-10,-10),Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop,'Edited' if img is self.edited else 'Original')

    def mousePressEvent(self,e):
        pos=e.position(); area=self._content_area()
        if e.button()==Qt.MouseButton.MiddleButton:
            self._panning=True; self._last=pos; self.setCursor(Qt.CursorShape.ClosedHandCursor); e.accept(); return
        if e.button()==Qt.MouseButton.LeftButton and self._mask_stage is not None:
            image_pt=self._image_point_from_widget(pos)
            if image_pt is not None:
                self._mask_painting=True; self._mask_last_image=image_pt; self._mask_hover=pos
                self._begin_mask_stroke()
                self._paint_mask_segment(image_pt,image_pt); e.accept(); return
        if e.button()==Qt.MouseButton.LeftButton and self.mode=='split' and not self.edited.isNull():
            sx=area.left()+area.width()*self.split
            if abs(pos.x()-sx)<12: self._split_drag=True; e.accept(); return
        super().mousePressEvent(e)

    def mouseMoveEvent(self,e):
        if self._panning:
            d=e.position()-self._last; self.pan=self._constrained_pan(self.pan+d); self._last=e.position(); self.update(); e.accept(); return
        if self._mask_stage is not None:
            self._mask_hover=e.position()
            if self._mask_painting:
                image_pt=self._image_point_from_widget(e.position())
                if image_pt is not None and self._mask_last_image is not None:
                    self._paint_mask_segment(self._mask_last_image,image_pt); self._mask_last_image=image_pt
                self.update(); e.accept(); return
            self.setCursor(Qt.CursorShape.CrossCursor); self.update()
        if self._split_drag:
            area=self._content_area(); self.split=max(0.02,min(.98,(e.position().x()-area.left())/max(1.0,area.width()))); self.splitChanged.emit(self.split); self.update(); e.accept(); return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self,e):
        if e.button()==Qt.MouseButton.MiddleButton and self._panning:
            self._panning=False; self.setCursor(Qt.CursorShape.CrossCursor if self._mask_stage is not None else Qt.CursorShape.ArrowCursor); e.accept(); return
        if e.button()==Qt.MouseButton.LeftButton and self._mask_painting:
            self._mask_painting=False; self._mask_last_image=None
            if self._mask_stage is not None:
                self.maskEdited.emit(self._mask_stage,self._mask.copy(),self._mask_authored)
            self._end_mask_stroke()
            e.accept(); return
        if e.button()==Qt.MouseButton.LeftButton: self._split_drag=False
        super().mouseReleaseEvent(e)

    def leaveEvent(self,e):
        self._mask_hover=None
        self.update()
        super().leaveEvent(e)

    def resizeEvent(self,e):
        self.pan=self._constrained_pan(self.pan); super().resizeEvent(e)

    @staticmethod
    def _local_paths_from_mime(mime):
        if not mime or not mime.hasUrls():
            return []
        paths=[]
        for url in mime.urls():
            if url.isLocalFile():
                path=url.toLocalFile()
                if path:
                    paths.append(path)
        return paths

    def dragEnterEvent(self, event: QDragEnterEvent):
        if self._local_paths_from_mime(event.mimeData()):
            event.acceptProposedAction(); return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent):
        if self._local_paths_from_mime(event.mimeData()):
            event.acceptProposedAction(); return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        paths=self._local_paths_from_mime(event.mimeData())
        if paths:
            self.droppedPaths.emit(paths); event.acceptProposedAction(); return
        event.ignore()

    def wheelEvent(self,e:QWheelEvent):
        step=1.12 if e.angleDelta().y()>0 else 1/1.12
        current=self.zoom if self.zoom>0 else self._base_scale(self._active(),self._reference_area())
        self.set_zoom(current*step); e.accept()

