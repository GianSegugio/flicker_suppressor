from __future__ import annotations

import json, tempfile, uuid
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QItemSelectionModel, QPointF, QSize, Qt, QThreadPool, QSignalBlocker, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QClipboard, QColor, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QFrame, QGroupBox,
    QButtonGroup, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QPushButton,
    QScrollArea, QSlider, QSpinBox, QDoubleSpinBox, QStackedWidget, QToolButton, QVBoxLayout, QWidget, QSizePolicy
)

from .app_paths import load_state, resource_path, save_state
from .engine import FlickerEngine
from .device_options import device_options
from .settings_schema import GROUP_ORDER, canonical_cutoff, cutoff_luma, default_settings, namespace, specs, validate_imported_settings
from .widgets import (
    ClipStripItem, ComparisonCanvas, DownChevronButton, EyeToggleButton, ImageStrip, ModernComboBox,
    ModernDoubleSpinBox, ModernSpinBox, ResetButton, ResettableSlider,
)
from .workers import AutoSettingsWorker, CopyWorker, InferenceWorker

IMAGE_FILTER='Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)'
SUPPORTED_IMAGE_EXTS={'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}
MIME_SETTINGS='application/x-flicker-suppressor-settings'
THUMBNAIL_ROLE=int(Qt.ItemDataRole.UserRole)+1
MASK_STAGE_SETTING={'flat_filter':'flat','flat_profile':'profile','flat_surface_equalizer':'broad'}

@dataclass
class ImageDocument:
    id: str
    path: Path
    settings: dict=field(default_factory=lambda:dict(default_settings()))
    preview_path: Path|None=None
    activated: bool=False
    dirty: bool=False
    processing: bool=False
    masks: dict[str,QImage]=field(default_factory=dict)
    mask_authored: set[str]=field(default_factory=set)
    # Auto-settings estimate for this image, or None if not run / still running.
    # Holds confidence, reason, period_px, and the ranked candidates list the
    # period dropdown will read.
    auto: dict|None=None
    # The "could not be determined" popup is shown once per image, on first
    # selection, not on every reselect.
    auto_notified: bool=False


@dataclass
class ExportJob:
    doc_id: str
    target: Path


class SpinnerWidget(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setFixedSize(48,48)
        self._angle=0
        self._timer=QTimer(self)
        self._timer.setInterval(70)
        self._timer.timeout.connect(self._tick)

    def start(self):
        self._angle=0
        self._timer.start()
        self.update()

    def stop(self):
        self._timer.stop()

    def _tick(self):
        self._angle=(self._angle+30)%360
        self.update()

    def paintEvent(self,event):
        p=QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing,True)
        p.translate(self.width()/2,self.height()/2)
        p.rotate(self._angle)
        for i in range(12):
            color=QColor('#e5e8ed')
            color.setAlpha(45+int(210*(i+1)/12))
            pen=QPen(color,3.0)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawLine(QPointF(0,-13),QPointF(0,-19))
            p.rotate(30)
        p.end()


class ProcessingOverlay(QWidget):
    '''Canvas-only busy overlay used by explicit Preview / Apply jobs.'''

    def __init__(self,parent=None):
        super().__init__(parent)
        self.setObjectName('ProcessingOverlay')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground,True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet('''
            QWidget#ProcessingOverlay { background:rgba(42,43,47,210); }
            QFrame#ProcessingOverlayCard { background:#292b2f; border:1px solid #4a4d53; border-radius:8px; }
            QLabel#ProcessingOverlayLabel { background:transparent; font-size:12pt; font-weight:600; }
        ''')
        outer=QVBoxLayout(self)
        outer.setContentsMargins(20,20,20,20)
        outer.addStretch(1)
        card=QFrame()
        card.setObjectName('ProcessingOverlayCard')
        card.setFixedWidth(290)
        lay=QVBoxLayout(card)
        lay.setContentsMargins(28,24,28,24)
        lay.setSpacing(14)
        self.spinner=SpinnerWidget()
        lay.addWidget(self.spinner,0,Qt.AlignmentFlag.AlignHCenter)
        label=QLabel('Processing...')
        label.setObjectName('ProcessingOverlayLabel')
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(label)
        outer.addWidget(card,0,Qt.AlignmentFlag.AlignCenter)
        outer.addStretch(1)
        self.hide()

    def showEvent(self,event):
        self.spinner.start()
        self.setFocus(Qt.FocusReason.OtherFocusReason)
        super().showEvent(event)

    def hideEvent(self,event):
        self.spinner.stop()
        super().hideEvent(event)

    def keyPressEvent(self,event):
        # Canvas input remains blocked while Preview / Apply is rendering.
        event.accept()


class ExportOverlay(QWidget):
    cancelRequested=Signal()

    def __init__(self,parent=None):
        super().__init__(parent)
        self.setObjectName('ExportOverlay')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground,True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet('''
            QWidget#ExportOverlay { background:rgba(18,19,22,205); }
            QFrame#ExportOverlayCard { background:#292b2f; border:1px solid #4a4d53; border-radius:8px; }
            QLabel#ExportOverlayLabel { background:transparent; font-size:12pt; font-weight:600; }
            QPushButton#ExportCancelButton { min-width:96px; min-height:28px; }
        ''')
        outer=QVBoxLayout(self)
        outer.setContentsMargins(20,20,20,20)
        outer.addStretch(1)
        card=QFrame()
        card.setObjectName('ExportOverlayCard')
        card.setFixedWidth(290)
        lay=QVBoxLayout(card)
        lay.setContentsMargins(28,24,28,24)
        lay.setSpacing(14)
        self.spinner=SpinnerWidget()
        lay.addWidget(self.spinner,0,Qt.AlignmentFlag.AlignHCenter)
        label=QLabel('Export in progress...')
        label.setObjectName('ExportOverlayLabel')
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(label)
        self.cancel_button=QPushButton('Cancel')
        self.cancel_button.setObjectName('ExportCancelButton')
        self.cancel_button.clicked.connect(self._cancel)
        lay.addWidget(self.cancel_button,0,Qt.AlignmentFlag.AlignHCenter)
        outer.addWidget(card,0,Qt.AlignmentFlag.AlignCenter)
        outer.addStretch(1)
        self.hide()

    def _cancel(self):
        self.set_cancelling(True)
        self.cancelRequested.emit()

    def set_cancelling(self,cancelling: bool):
        self.cancel_button.setEnabled(not cancelling)
        self.cancel_button.setText('Cancelling...' if cancelling else 'Cancel')

    def showEvent(self,event):
        self.set_cancelling(False)
        self.spinner.start()
        self.setFocus(Qt.FocusReason.OtherFocusReason)
        super().showEvent(event)

    def hideEvent(self,event):
        self.spinner.stop()
        super().hideEvent(event)

    def keyPressEvent(self,event):
        # Consume keyboard input while the rest of the window is locked.
        event.accept()


class AutoSettingsOverlay(QWidget):
    """Busy overlay shown while band settings are being estimated.

    Visually identical to ExportOverlay, with two differences: no Cancel
    button, and it lets file drops through. Importing more images is the one
    thing a user is likely to want while a batch is being analysed, so the
    overlay accepts drops itself and re-emits them on the same signal the
    filmstrip and canvas use.
    """
    droppedPaths=Signal(list)

    def __init__(self,parent=None):
        super().__init__(parent)
        self.setObjectName('AutoSettingsOverlay')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground,True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)
        self.setStyleSheet("""
            QWidget#AutoSettingsOverlay { background:rgba(18,19,22,205); }
            QFrame#AutoSettingsOverlayCard { background:#292b2f; border:1px solid #4a4d53; border-radius:8px; }
            QLabel#AutoSettingsOverlayLabel { background:transparent; font-size:12pt; font-weight:600; }
        """)
        outer=QVBoxLayout(self)
        outer.setContentsMargins(20,20,20,20)
        outer.addStretch(1)
        card=QFrame()
        card.setObjectName('AutoSettingsOverlayCard')
        # Wider than ExportOverlay's 290 so the longer message fits on one line.
        # Wrapping was tried twice and clipped both times: QVBoxLayout ignores a
        # QLabel's heightForWidth unless its size policy opts in, so the card
        # kept sizing itself to a single line regardless.
        card.setFixedWidth(420)
        lay=QVBoxLayout(card)
        lay.setContentsMargins(28,24,28,24)
        lay.setSpacing(14)
        self.spinner=SpinnerWidget()
        lay.addWidget(self.spinner,0,Qt.AlignmentFlag.AlignHCenter)
        label=QLabel('Determining essential editing settings...')
        label.setObjectName('AutoSettingsOverlayLabel')
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(label)
        outer.addWidget(card,0,Qt.AlignmentFlag.AlignCenter)
        outer.addStretch(1)
        self.hide()

    @staticmethod
    def _local_paths_from_mime(mime):
        if not mime or not mime.hasUrls():
            return []
        out=[]
        for url in mime.urls():
            if url.isLocalFile():
                p=url.toLocalFile()
                if p:
                    out.append(p)
        return out

    def dragEnterEvent(self,event):
        if self._local_paths_from_mime(event.mimeData()):
            event.acceptProposedAction(); return
        event.ignore()

    def dragMoveEvent(self,event):
        if self._local_paths_from_mime(event.mimeData()):
            event.acceptProposedAction(); return
        event.ignore()

    def dropEvent(self,event):
        paths=self._local_paths_from_mime(event.mimeData())
        if paths:
            self.droppedPaths.emit(paths)
            event.acceptProposedAction(); return
        event.ignore()

    def showEvent(self,event):
        self.spinner.start()
        self.setFocus(Qt.FocusReason.OtherFocusReason)
        super().showEvent(event)

    def hideEvent(self,event):
        self.spinner.stop()
        super().hideEvent(event)

    def keyPressEvent(self,event):
        # Consume keyboard input while the window is locked.
        event.accept()


class BatchExportDialog(QDialog):
    _INVALID_FILENAME_CHARS=set('<>:"/\\|?*')

    def __init__(self,title: str,parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(520)
        outer=QVBoxLayout(self)
        form=QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)

        folder_row=QWidget()
        folder_lay=QHBoxLayout(folder_row)
        folder_lay.setContentsMargins(0,0,0,0)
        folder_lay.setSpacing(6)
        self.folder_edit=QLineEdit()
        browse=QPushButton('Browse...')
        browse.clicked.connect(self._browse)
        folder_lay.addWidget(self.folder_edit,1)
        folder_lay.addWidget(browse)
        form.addRow('Output folder',folder_row)

        self.prefix_edit=QLineEdit()
        self.suffix_edit=QLineEdit()
        self.prefix_edit.setPlaceholderText('Optional')
        self.suffix_edit.setPlaceholderText('Optional')
        form.addRow('Filename prefix',self.prefix_edit)
        form.addRow('Filename suffix',self.suffix_edit)
        outer.addLayout(form)

        note=QLabel('Files are exported as PNG. Prefix and suffix are inserted around the original filename.')
        note.setWordWrap(True)
        note.setObjectName('Muted')
        outer.addWidget(note)

        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Save|QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _browse(self):
        start=self.folder_edit.text().strip()
        folder=QFileDialog.getExistingDirectory(self,'Select export folder',start)
        if folder:
            self.folder_edit.setText(folder)

    def accept(self):
        folder=Path(self.folder_edit.text().strip()).expanduser() if self.folder_edit.text().strip() else None
        if folder is None or not folder.is_dir():
            QMessageBox.warning(self,'Export images','Choose an existing output folder.')
            return
        for label,text in [('prefix',self.prefix_edit.text()),('suffix',self.suffix_edit.text())]:
            if any(ch in self._INVALID_FILENAME_CHARS for ch in text):
                QMessageBox.warning(self,'Export images',f'The filename {label} contains a character that is not valid on Windows.')
                return
        super().accept()

    def values(self):
        return Path(self.folder_edit.text().strip()),self.prefix_edit.text(),self.suffix_edit.text()



class CurrentPageStack(QStackedWidget):
    """Stack whose layout height follows only the visible page.

    QStackedWidget normally reports the maximum size hint of every page.  The
    Residual Profile section is much taller than its mask controls, so using
    the default hint leaves a large empty hole when Mask mode is active.
    """
    def sizeHint(self):
        page=self.currentWidget()
        return page.sizeHint() if page is not None else super().sizeHint()

    def minimumSizeHint(self):
        page=self.currentWidget()
        return page.minimumSizeHint() if page is not None else super().minimumSizeHint()

    def setCurrentIndex(self,index):
        super().setCurrentIndex(index)
        self.updateGeometry()


class MaskToolPanel(QWidget):
    changed = Signal(object)
    copyRequested = Signal()
    pasteRequested = Signal()
    invertRequested = Signal()

    DEFAULTS = {
        'brush': {'size': 160, 'feather': 32, 'opacity': 100},
        'eraser': {'size': 160, 'feather': 32, 'opacity': 100},
    }

    def __init__(self,parent=None):
        super().__init__(parent)
        self._loading=False
        self._values={k:dict(v) for k,v in self.DEFAULTS.items()}
        # Masks begin fully selected, so Eraser is the natural first tool.
        # Brush remains independent and keeps its own Size/Feather/Opacity.
        self._tool='eraser'
        outer=QVBoxLayout(self); outer.setContentsMargins(0,0,0,0); outer.setSpacing(9)

        tool_row=QHBoxLayout(); tool_row.setContentsMargins(0,0,0,0); tool_row.setSpacing(6)
        self.brush_btn=QToolButton(); self.brush_btn.setObjectName('MaskToolButton'); self.brush_btn.setText('Brush'); self.brush_btn.setCheckable(True)
        self.eraser_btn=QToolButton(); self.eraser_btn.setObjectName('MaskToolButton'); self.eraser_btn.setText('Eraser'); self.eraser_btn.setCheckable(True)
        # SVG assets stay sharp at arbitrary Windows/Qt DPI scaling.  The
        # right-to-left button layout places the icon after the text label.
        for button,asset in ((self.brush_btn,'mask_brush.svg'),(self.eraser_btn,'mask_eraser.svg')):
            icon_path=resource_path('assets',asset)
            if icon_path.exists(): button.setIcon(QIcon(str(icon_path)))
            button.setIconSize(QSize(16,16))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.invert_btn=QToolButton(); self.invert_btn.setObjectName('MaskToolButton'); self.invert_btn.setText('Invert')
        self.invert_btn.setToolTip('Invert the current mask opacity: painted areas become unpainted and unpainted areas become painted.')
        self.tool_group=QButtonGroup(self); self.tool_group.setExclusive(True); self.tool_group.addButton(self.brush_btn); self.tool_group.addButton(self.eraser_btn)
        self.eraser_btn.setChecked(True)
        tool_row.addWidget(self.brush_btn,1); tool_row.addWidget(self.eraser_btn,1); tool_row.addWidget(self.invert_btn,1); outer.addLayout(tool_row)

        self.size_slider,self.size_spin,self.size_row=self._slider_row(1,1000,160,'px')
        self.feather_slider,self.feather_spin,self.feather_row=self._slider_row(0,1000,32,'px')
        self.opacity_slider,self.opacity_spin,self.opacity_row=self._slider_row(1,100,100,'%')
        outer.addWidget(self._labeled('Size',self.size_row))
        outer.addWidget(self._labeled('Feather',self.feather_row))
        outer.addWidget(self._labeled('Opacity',self.opacity_row))

        transfer_row=QHBoxLayout(); transfer_row.setContentsMargins(0,0,0,0); transfer_row.setSpacing(6)
        self.copy_mask_btn=QPushButton('Copy mask')
        self.copy_mask_btn.setToolTip('Copy this section mask so it can be pasted into another cleanup section.')
        self.paste_mask_btn=QPushButton('Paste mask')
        self.paste_mask_btn.setToolTip('Replace this section mask with the copied cleanup mask.')
        self.paste_mask_btn.setEnabled(False)
        transfer_row.addWidget(self.copy_mask_btn,1); transfer_row.addWidget(self.paste_mask_btn,1)
        outer.addLayout(transfer_row)

        note=QLabel('Paint on the image. Red indicates where this cleanup stage is allowed to apply.')
        note.setObjectName('Muted'); note.setWordWrap(True); outer.addWidget(note)

        self.brush_btn.clicked.connect(lambda:self._switch_tool('brush'))
        self.eraser_btn.clicked.connect(lambda:self._switch_tool('eraser'))
        self.invert_btn.clicked.connect(self.invertRequested.emit)
        self.copy_mask_btn.clicked.connect(self.copyRequested.emit)
        self.paste_mask_btn.clicked.connect(self.pasteRequested.emit)
        for w in (self.size_spin,self.feather_spin,self.opacity_spin): w.valueChanged.connect(self._parameter_changed)
        self._load_tool()

    @staticmethod
    def _labeled(text,control):
        w=QWidget(); v=QVBoxLayout(w); v.setContentsMargins(0,0,0,0); v.setSpacing(4); v.addWidget(QLabel(text)); v.addWidget(control); return w

    @staticmethod
    def _slider_row(lo,hi,value,suffix):
        row=QWidget(); h=QHBoxLayout(row); h.setContentsMargins(0,0,0,0); h.setSpacing(8)
        slider=QSlider(Qt.Orientation.Horizontal); slider.setRange(lo,hi); slider.setValue(value); slider.setMinimumWidth(90)
        spin=ModernSpinBox(); spin.setRange(lo,hi); spin.setValue(value); spin.setFixedWidth(86); spin.setSuffix(f' {suffix}')
        slider.valueChanged.connect(spin.setValue); spin.valueChanged.connect(slider.setValue)
        h.addWidget(slider,1); h.addWidget(spin,0)
        return slider,spin,row

    def _store_tool(self):
        self._values[self._tool]={
            'size':int(self.size_spin.value()),
            'feather':int(self.feather_spin.value()),
            'opacity':int(self.opacity_spin.value()),
        }

    def _load_tool(self):
        vals=self._values[self._tool]; self._loading=True
        try:
            self.size_spin.setValue(vals['size']); self.feather_spin.setValue(vals['feather']); self.opacity_spin.setValue(vals['opacity'])
        finally: self._loading=False

    def _switch_tool(self,tool):
        tool='eraser' if tool=='eraser' else 'brush'
        if tool==self._tool: return
        self._store_tool(); self._tool=tool; self._load_tool(); self.changed.emit(self.current())

    def _parameter_changed(self,*_):
        if self._loading: return
        self._store_tool(); self.changed.emit(self.current())

    def current(self):
        self._store_tool(); out=dict(self._values[self._tool]); out['tool']=self._tool; return out

    def set_paste_available(self,available):
        self.paste_mask_btn.setEnabled(bool(available))


class SettingsPanel(QFrame):
    changed = Signal(str,object)
    activationChanged = Signal(bool)
    exportSettingsRequested = Signal()
    importSettingsRequested = Signal()
    maskModeChanged = Signal(str,bool)
    maskBrushChanged = Signal(str,object)
    maskCopyRequested = Signal(str)
    maskPasteRequested = Signal(str)
    maskInvertRequested = Signal(str)
    def __init__(self,parent=None):
        super().__init__(parent); self.setObjectName('SettingsPanel'); self.widgets={}; self.reset_buttons={}; self._defaults=dict(default_settings()); self._loading=False; self.mask_buttons={}; self.mask_stacks={}; self.mask_panels={}; self._active_mask_stage=None
        outer=QVBoxLayout(self); outer.setContentsMargins(12,10,12,10); outer.setSpacing(8)
        title=QLabel('Editing settings'); title.setStyleSheet('font-size:12pt;font-weight:600;'); outer.addWidget(title)
        self.activation_checkbox=QCheckBox('Activate Flicker Suppressor for this image')
        self.activation_checkbox.setToolTip('Enable Flicker Suppressor editing and export for the current image.')
        self.activation_checkbox.toggled.connect(lambda checked:self.activationChanged.emit(bool(checked)))
        outer.addWidget(self.activation_checkbox)
        self.scroll=QScrollArea(); self.scroll.setObjectName('SettingsScrollArea'); self.scroll.setWidgetResizable(True); self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); outer.addWidget(self.scroll,1)
        content=QWidget(); content.setMinimumWidth(0); content.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred); self.scroll.setWidget(content); lay=QVBoxLayout(content); lay.setContentsMargins(0,0,0,0); lay.setSpacing(9)
        grouped={g:[] for g in GROUP_ORDER}
        for s in specs(): grouped.setdefault(s.group,[]).append(s)
        mask_groups={
            'Flat-region cleanup':('flat','flat_filter'),
            'Residual profile':('profile','flat_profile'),
            'Broad residual cleanup':('broad','flat_surface_equalizer'),
        }
        for group in GROUP_ORDER:
            if not grouped.get(group):
                continue
            box=QGroupBox(group)
            box.setMinimumWidth(0)
            box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            group_lay=QVBoxLayout(box)
            group_lay.setContentsMargins(10,12,10,9)
            group_lay.setSpacing(8)

            normal=QWidget(); normal_lay=QVBoxLayout(normal); normal_lay.setContentsMargins(0,0,0,0); normal_lay.setSpacing(8)
            target_lay=normal_lay
            stage=None; stage_toggle_dest=None; mask_btn=None; stack=None
            if group in mask_groups:
                stage,stage_toggle_dest=mask_groups[group]
                mask_btn=QPushButton('Mask')
                mask_btn.setCheckable(True)
                mask_btn.setToolTip('Paint a per-image mask that limits this cleanup stage to the red-painted area.')
                mask_btn.toggled.connect(lambda checked,st=stage:self._mask_toggled(st,checked))
                self.mask_buttons[stage]=mask_btn
                stack=CurrentPageStack(); self.mask_stacks[stage]=stack
                mask_panel=MaskToolPanel(); self.mask_panels[stage]=mask_panel
                mask_panel.changed.connect(lambda params,st=stage:self.maskBrushChanged.emit(st,params))
                mask_panel.copyRequested.connect(lambda st=stage:self.maskCopyRequested.emit(st))
                mask_panel.pasteRequested.connect(lambda st=stage:self.maskPasteRequested.emit(st))
                mask_panel.invertRequested.connect(lambda st=stage:self.maskInvertRequested.emit(st))
                stack.addWidget(normal); stack.addWidget(mask_panel)
            else:
                group_lay.addWidget(normal)

            mask_header_added=False
            for sp in grouped[group]:
                w=self._make_widget(sp)
                self.widgets[sp.dest]=w
                w.setToolTip(sp.help)

                if isinstance(w,QCheckBox):
                    if sp.dest == 'orthogonal_profile':
                        target_lay.addSpacing(10)
                    w.setText(sp.label)
                    w.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
                    if stage is not None and sp.dest==stage_toggle_dest:
                        # Keep the stage On/Off switch visible while painting,
                        # with Mask immediately underneath it. Only the stage's
                        # remaining controls are swapped for the brush panel.
                        group_lay.addWidget(w)
                        group_lay.addWidget(mask_btn)
                        group_lay.addWidget(stack)
                        mask_header_added=True
                    else:
                        target_lay.addWidget(w)
                    continue

                item=QWidget()
                item.setMinimumWidth(0)
                item.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
                item_lay=QVBoxLayout(item)
                item_lay.setContentsMargins(0,0,0,0)
                item_lay.setSpacing(4)
                label=QLabel(sp.label)
                label.setWordWrap(False)
                label.setMinimumWidth(0)
                label.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
                label.setToolTip(sp.help)
                item_lay.addWidget(label)
                item_lay.addWidget(self._control_with_reset(sp,w))
                target_lay.addWidget(item)
            if stage is not None and not mask_header_added:
                # Defensive fallback if a future schema reorders/removes the
                # expected enable checkbox.
                group_lay.addWidget(mask_btn)
                group_lay.addWidget(stack)
            lay.addWidget(box)

        # Developer utility: export/import the exact complete inference settings
        # used for the current image. This intentionally lives after Broad
        # residual cleanup and is not part of the CLI-backed spec list.
        export_box=QGroupBox('Export/import settings (dev)')
        export_box.setMinimumWidth(0)
        export_box.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Preferred)
        export_lay=QVBoxLayout(export_box)
        export_lay.setContentsMargins(10,12,10,9)
        export_lay.setSpacing(8)
        button_row=QWidget()
        button_lay=QVBoxLayout(button_row)
        button_lay.setContentsMargins(0,0,0,0)
        button_lay.setSpacing(6)
        self.export_settings_button=QPushButton('Export json')
        self.export_settings_button.setToolTip(
            'Export all processing settings for the current image as JSON, '
            'including settings that are hidden from the GUI.'
        )
        self.export_settings_button.clicked.connect(self.exportSettingsRequested.emit)
        button_lay.addWidget(self.export_settings_button,1)
        self.import_settings_button=QPushButton('Import json')
        self.import_settings_button.setToolTip(
            'Import processing settings from a Flicker Suppressor JSON export. '
            'The file is validated before any settings are applied. Masks are not imported.'
        )
        self.import_settings_button.clicked.connect(self.importSettingsRequested.emit)
        button_lay.addWidget(self.import_settings_button,1)
        export_lay.addWidget(button_row)
        lay.addWidget(export_box)

        lay.addStretch(1)
        self.activation_checkbox.setChecked(False)
        self.activation_checkbox.setEnabled(False)
        self.scroll.setEnabled(False)

    @property
    def active_mask_stage(self):
        return self._active_mask_stage

    def current_mask_parameters(self,stage):
        panel=self.mask_panels.get(stage)
        return panel.current() if panel is not None else {'tool':'eraser','size':160,'feather':32,'opacity':100}

    def set_mask_paste_available(self,available):
        for panel in self.mask_panels.values():
            panel.set_paste_available(available)

    def _mask_toggled(self,stage,checked):
        if checked:
            previous=self._active_mask_stage
            if previous and previous!=stage:
                btn=self.mask_buttons.get(previous)
                if btn is not None:
                    with QSignalBlocker(btn): btn.setChecked(False)
                if previous in self.mask_stacks: self.mask_stacks[previous].setCurrentIndex(0)
                if previous in self.mask_buttons: self.mask_buttons[previous].setText('Mask')
            self._active_mask_stage=stage
            self.mask_buttons[stage].setText('Settings')
            self.mask_stacks[stage].setCurrentIndex(1)
            self.maskBrushChanged.emit(stage,self.current_mask_parameters(stage))
            self.maskModeChanged.emit(stage,True)
        else:
            if stage in self.mask_stacks: self.mask_stacks[stage].setCurrentIndex(0)
            if stage in self.mask_buttons: self.mask_buttons[stage].setText('Mask')
            if self._active_mask_stage==stage:
                self._active_mask_stage=None
                self.maskModeChanged.emit(stage,False)

    def exit_mask_mode(self):
        stage=self._active_mask_stage
        if not stage: return
        btn=self.mask_buttons.get(stage)
        if btn is not None:
            with QSignalBlocker(btn): btn.setChecked(False)
        if stage in self.mask_stacks: self.mask_stacks[stage].setCurrentIndex(0)
        if stage in self.mask_buttons: self.mask_buttons[stage].setText('Mask')
        self._active_mask_stage=None
        self.maskModeChanged.emit(stage,False)

    @staticmethod
    def _gray_hex_from_slider(value: int) -> str:
        v=max(0,min(255,int(value)))
        return f"#{v:02X}{v:02X}{v:02X}"

    @staticmethod
    def _slider_from_gray_hex(text: str, fallback: int=128) -> int:
        t=str(text).strip().lower().lstrip('#')
        if len(t)==6 and all(c in '0123456789abcdef' for c in t):
            r,g,b=(int(t[i:i+2],16) for i in (0,2,4))
            return max(0,min(255,int(round(0.299*r+0.587*g+0.114*b))))
        return max(0,min(255,int(fallback)))

    def _control_with_reset(self, sp, control):
        resettable = bool(
            sp.slider
            or sp.dest in {'flat_highpass','flat_lowpass','flat_profile_band_period'}
            or (sp.value_type in {int,float} and not sp.choices)
        )
        if not resettable:
            return control
        row=QWidget()
        row.setObjectName('ResetControlRow')
        row.setMinimumWidth(0)
        row.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
        h=QHBoxLayout(row)
        h.setContentsMargins(0,0,0,0)
        h.setSpacing(7)
        h.addWidget(control,1)
        btn=ResetButton()
        btn.setToolTip(f'Reset {sp.label} to its default value')
        btn.clicked.connect(lambda _=False,d=sp.dest:self._reset_setting(d))
        align = Qt.AlignmentFlag.AlignBottom if sp.dest == 'flat_profile_band_period' else Qt.AlignmentFlag.AlignVCenter
        h.addWidget(btn,0,align)
        self.reset_buttons[sp.dest]=btn
        return row

    def _set_widget_value(self, d, w, v):
        if hasattr(w,'_period_auto'):
            period=float(v or 0.0)
            if period > 0:
                w._last_manual_period=max(1.0,min(7680.0,period))
                with QSignalBlocker(w._period_spin):
                    w._period_spin.setValue(w._last_manual_period)
            elif w._last_manual_period is None:
                w._last_manual_period=1.0
                with QSignalBlocker(w._period_spin):
                    w._period_spin.setValue(1.0)
            with QSignalBlocker(w._period_auto):
                idx=0 if period <= 0.0 else -1
                if period > 0:
                    # Match a listed candidate within 0.5%; otherwise Custom.
                    for i in range(w._period_auto.count()):
                        data=w._period_auto.itemData(i)
                        if data is not None and float(data) > 0 and abs(float(data)/period-1.0) < 0.005:
                            idx=i
                            break
                    if idx < 0:
                        idx=w._period_auto.count()-1      # Custom is always last
                w._period_auto.setCurrentIndex(max(0,idx))
            w._period_spin.setEnabled(w.isEnabled() and w._period_auto.currentData() is None)
            return
        if hasattr(w,'_cutoff'):
            fallback='#000000' if d=='flat_highpass' else '#FFFFFF'
            text=canonical_cutoff(v,fallback)
            w._line.setText(text)
            with QSignalBlocker(w._slider):
                w._slider.setValue(self._slider_from_gray_hex(text,w._slider.value()))
            return
        if hasattr(w,'_spin'):
            w._spin.setValue(float(v))
            return
        if isinstance(w,QCheckBox):
            w.setChecked(bool(v)); return
        if isinstance(w,(ModernComboBox,QComboBox)):
            i=w.findData(v)
            # Old GUI versions stored the first CUDA device as plain "cuda".
            if i < 0 and str(v).lower() == 'cuda':
                i=w.findData('cuda:0')
            w.setCurrentIndex(i if i>=0 else 0); return
        if isinstance(w,(ModernSpinBox,QSpinBox)):
            iv=int(v)
            if d=='processing_size':
                iv=max(256,min(2048,((iv + 4)//8)*8))
            elif d=='flat_local_edge_distance':
                iv=max(0,min(100,iv))
            w.setValue(iv); return
        if isinstance(w,(ModernDoubleSpinBox,QDoubleSpinBox)):
            w.setValue(float(v)); return
        w.setText(str(v))

    def _reset_setting(self, dest):
        if dest not in self.widgets:
            return
        self._loading=True
        try:
            if dest=='flat_profile_band_period':
                w=self.widgets[dest]
                w._last_manual_period=1.0
                with QSignalBlocker(w._period_spin):
                    w._period_spin.setValue(1.0)
            self._set_widget_value(dest,self.widgets[dest],self._defaults.get(dest))
            if dest in {'flat_highpass','flat_lowpass'}:
                self._normalize_cutoff_pair(dest,emit=False)
        finally:
            self._loading=False
        self._emit(dest)

    def _sync_cutoff_from_slider(self, wrap, value: int, dest: str):
        with QSignalBlocker(wrap._line):
            wrap._line.setText(self._gray_hex_from_slider(value))
        self._normalize_cutoff_pair(dest,emit=True)

    def _normalize_cutoff_pair(self, changed_dest: str, emit=True):
        shadow=self.widgets.get('flat_highpass')
        highlight=self.widgets.get('flat_lowpass')
        if not shadow or not highlight:
            return
        shadow_text=canonical_cutoff(shadow._line.text(),'#000000')
        highlight_text=canonical_cutoff(highlight._line.text(),'#FFFFFF')
        if cutoff_luma(shadow_text) > cutoff_luma(highlight_text):
            if changed_dest=='flat_highpass':
                shadow_text='#000000'
            elif changed_dest=='flat_lowpass':
                highlight_text='#FFFFFF'
            else:
                shadow_text='#000000'
                highlight_text='#FFFFFF'
        for d,w,text in (('flat_highpass',shadow,shadow_text),('flat_lowpass',highlight,highlight_text)):
            with QSignalBlocker(w._line):
                w._line.setText(text)
            with QSignalBlocker(w._slider):
                w._slider.setValue(self._slider_from_gray_hex(text,w._slider.value()))
        if emit:
            self._emit(changed_dest)

    def _sync_cutoff_from_text(self, wrap, dest: str):
        fallback='#000000' if dest=='flat_highpass' else '#FFFFFF'
        text=canonical_cutoff(wrap._line.text(),fallback)
        with QSignalBlocker(wrap._line):
            wrap._line.setText(text)
        self._normalize_cutoff_pair(dest,emit=True)

    def _normalize_processing_size(self, widget):
        value=max(256,min(2048,int(widget.value())))
        value=max(256,min(2048,((value + 4)//8)*8))
        if value != widget.value():
            with QSignalBlocker(widget):
                widget.setValue(value)
        self._emit('processing_size')

    def _emit(self,dest):
        if not self._loading: self.changed.emit(dest,self.value(dest)); self.update_dependencies()

    def _make_widget(self,sp):
        if sp.dest == 'flat_profile_band_period':
            wrap=QWidget()
            wrap.setObjectName('PeriodAutoControl')
            wrap.setMinimumWidth(0)
            wrap.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
            v=QVBoxLayout(wrap)
            v.setContentsMargins(0,0,0,0)
            v.setSpacing(5)
            auto=ModernComboBox()
            auto.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
            auto.setMinimumWidth(0)
            spin=ModernDoubleSpinBox()
            spin.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
            spin.setRange(1,7680)
            spin.setDecimals(2)
            spin.setSingleStep(1.0)
            spin.setValue(1.0)
            v.addWidget(auto)
            v.addWidget(spin)
            wrap._period_auto=auto
            wrap._period_spin=spin
            wrap._last_manual_period=1.0
            wrap._period_candidates=[]

            def mode_changed(_idx, w=wrap, d=sp.dest):
                data=w._period_auto.currentData()
                if data is None:                       # Custom: user drives the spin
                    with QSignalBlocker(w._period_spin):
                        w._period_spin.setValue(max(1.0,min(7680.0,float(w._last_manual_period or 1.0))))
                elif float(data) > 0:                  # a candidate
                    with QSignalBlocker(w._period_spin):
                        w._period_spin.setValue(max(1.0,min(7680.0,float(data))))
                w._period_spin.setEnabled(w.isEnabled() and data is None)
                self._emit(d)

            def period_changed(value, w=wrap, d=sp.dest):
                w._last_manual_period=max(1.0,min(7680.0,float(value)))
                self._emit(d)

            auto.currentIndexChanged.connect(mode_changed)
            spin.valueChanged.connect(period_changed)
            self._rebuild_period_items(wrap,[])
            return wrap
        if sp.dest in {'flat_highpass','flat_lowpass'}:
            wrap=QWidget()
            wrap.setObjectName('SliderRow')
            wrap.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            wrap.setMinimumWidth(0)
            wrap.setMinimumHeight(38)
            wrap.setMaximumHeight(38)
            wrap.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
            h=QHBoxLayout(wrap)
            h.setContentsMargins(0,0,0,0)
            h.setSpacing(8)
            sl=ResettableSlider(Qt.Orientation.Horizontal)
            sl.setObjectName('FlatSlider')
            sl.setRange(0,255)
            sl.setMinimumWidth(80)
            sl.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
            line=QLineEdit()
            line.setMaxLength(7)
            line.setFixedWidth(82)
            line.setAlignment(Qt.AlignmentFlag.AlignCenter)
            line.setToolTip('RGB hex cutoff, for example #232323 or 232323')
            h.addWidget(sl,1)
            h.addWidget(line,0,Qt.AlignmentFlag.AlignVCenter)
            wrap._slider=sl
            wrap._line=line
            wrap._cutoff=True
            sl.valueChanged.connect(lambda v,w=wrap,d=sp.dest:self._sync_cutoff_from_slider(w,v,d))
            sl.resetRequested.connect(lambda d=sp.dest:self._reset_setting(d))
            line.editingFinished.connect(lambda w=wrap,d=sp.dest:self._sync_cutoff_from_text(w,d))
            return wrap
        if sp.value_type is bool:
            w=QCheckBox(); w.toggled.connect(lambda _=False,d=sp.dest:self._emit(d)); return w
        if sp.dest == 'device':
            w=ModernComboBox(); w.setMinimumWidth(0); w.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
            for option in device_options():
                w.addItem(option.label,option.value)
            w.currentIndexChanged.connect(lambda _=0,d=sp.dest:self._emit(d)); return w
        if sp.dest == 'flat_profile_mode':
            w=ModernComboBox(); w.setMinimumWidth(0); w.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
            w.addItem('Smooth periodic','smooth')
            w.addItem('PWM / Step','pwm')
            w.currentIndexChanged.connect(lambda _=0,d=sp.dest:self._emit(d)); return w
        if sp.dest == 'flat_surface_equalizer_mode':
            w=ModernComboBox(); w.setMinimumWidth(0); w.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
            w.addItem('Dominant surface','dominant')
            w.addItem('Multi-surface consensus','consensus')
            w.currentIndexChanged.connect(lambda _=0,d=sp.dest:self._emit(d)); return w
        if sp.choices:
            w=ModernComboBox(); w.setMinimumWidth(0); w.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed); [w.addItem(str(x),x) for x in sp.choices]; w.currentIndexChanged.connect(lambda _=0,d=sp.dest:self._emit(d)); return w
        if sp.value_type is int:
            w=ModernSpinBox(); w.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
            if sp.dest == 'processing_size':
                w.setRange(256,2048); w.setSingleStep(8); w.editingFinished.connect(lambda box=w:self._normalize_processing_size(box))
            elif sp.dest == 'flat_local_edge_distance':
                w.setRange(0,100); w.setSingleStep(1)
            elif sp.dest == 'flat_profile_pwm_polish_passes':
                w.setRange(1,3); w.setSingleStep(1)
            else:
                w.setRange(-1000000,1000000)
            w.valueChanged.connect(lambda _=0,d=sp.dest:self._emit(d)); return w
        if sp.value_type is float:
            if sp.slider:
                wrap=QWidget()
                wrap.setObjectName('SliderRow')
                wrap.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
                wrap.setMinimumWidth(0)
                wrap.setMinimumHeight(38)
                wrap.setMaximumHeight(38)
                wrap.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
                h=QHBoxLayout(wrap)
                h.setContentsMargins(0,0,0,0)
                h.setSpacing(8)
                sl=ResettableSlider(Qt.Orientation.Horizontal)
                sl.setObjectName('FlatSlider')
                slider_ranges={
                    'first_pass_luma_strength': (0.0,2.0),
                    'first_pass_chroma_strength': (0.0,2.0),
                    'second_pass_strength': (0.0,2.0),
                    'flat_luma_strength': (0.0,2.0),
                    'flat_chroma_strength': (0.0,2.0),
                    'flat_profile_luma_strength': (0.0,4.0),
                    'flat_profile_chroma_strength': (0.0,4.0),
                    'flat_profile_pwm_polish_strength': (0.0,1.25),
                    'orthogonal_profile_luma_strength': (-1.0,4.0),
                    'orthogonal_profile_chroma_strength': (-1.0,4.0),
                    'flat_surface_equalizer_luma_strength': (0.0,2.0),
                    'flat_surface_equalizer_chroma_strength': (0.0,2.0),
                }
                lo,hi=slider_ranges.get(sp.dest,(-2.0,4.0))
                sl.setRange(int(round(lo*100)),int(round(hi*100)))
                sl.setMinimumWidth(65)
                sl.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
                sb=ModernDoubleSpinBox()
                sb.setRange(lo,hi)
                sb.setDecimals(2)
                sb.setSingleStep(.05)
                sb.setFixedWidth(96)
                h.addWidget(sl,1)
                h.addWidget(sb,0,Qt.AlignmentFlag.AlignVCenter)
                sl.valueChanged.connect(lambda v,s=sb:s.setValue(v/100))
                sl.resetRequested.connect(lambda d=sp.dest:self._reset_setting(d))
                sb.valueChanged.connect(lambda v,s=sl:s.setValue(int(round(v*100))))
                sb.valueChanged.connect(lambda _=0,d=sp.dest:self._emit(d))
                wrap._spin=sb
                wrap._slider=sl
                return wrap
            w=ModernDoubleSpinBox(); w.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed); w.setRange(-1000000,1000000); w.setDecimals(5); w.setSingleStep(.05); w.valueChanged.connect(lambda _=0,d=sp.dest:self._emit(d)); return w
        w=QLineEdit(); w.setMinimumWidth(0); w.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed); w.editingFinished.connect(lambda d=sp.dest:self._emit(d)); return w
    def value(self,d):
        w=self.widgets[d]
        if hasattr(w,'_period_auto'):
            data=w._period_auto.currentData()
            if data is None:
                return float(w._period_spin.value())    # Custom
            return float(data)                          # 0.0 for Auto, else the candidate
        if hasattr(w,'_cutoff'): return w._line.text().strip()
        if hasattr(w,'_spin'): return float(w._spin.value())
        if isinstance(w,QCheckBox): return w.isChecked()
        if isinstance(w,(ModernComboBox,QComboBox)): return w.currentData()
        if isinstance(w,(ModernSpinBox,QSpinBox)): return int(w.value())
        if isinstance(w,(ModernDoubleSpinBox,QDoubleSpinBox)): return float(w.value())
        return w.text()
    def _rebuild_period_items(self,w,candidates):
        """Repopulate Auto / candidates / Custom, preserving the current value."""
        current=None
        if w._period_auto.count():
            current=w._period_auto.currentData()
        with QSignalBlocker(w._period_auto):
            w._period_auto.clear()
            w._period_auto.addItem('Auto',0.0)
            for c in (candidates or []):
                try:
                    per=float(c.get('period_px'))
                except (TypeError,ValueError):
                    continue
                if per <= 0:
                    continue
                cyc=c.get('cycles')
                rel=str(c.get('relation') or '')
                bits=[]
                if cyc:
                    bits.append(f'{float(cyc):g} cycles')
                if rel=='harmonic':
                    bits.append('harmonic')
                elif rel=='independent':
                    bits.append('alt')
                suffix=f"  ({', '.join(bits)})" if bits else ''
                w._period_auto.addItem(f'{per:g} px{suffix}',per)
            w._period_auto.addItem('Custom...',None)
            idx=0
            if current is None and w._period_auto.count():
                idx=w._period_auto.count()-1
            elif current is not None:
                found=w._period_auto.findData(current)
                idx=found if found >= 0 else 0
            w._period_auto.setCurrentIndex(max(0,idx))
        w._period_candidates=list(candidates or [])
        w._period_spin.setEnabled(w.isEnabled() and w._period_auto.currentData() is None)

    def set_period_candidates(self,candidates):
        """Feed the estimator's ranked periods into the dropdown.

        Called when a document becomes current and again when its estimate
        arrives. Passing None or [] leaves just Auto and Custom.
        """
        w=self.widgets.get('flat_profile_band_period')
        if w is None or not hasattr(w,'_period_auto'):
            return
        value=self.value('flat_profile_band_period')
        self._rebuild_period_items(w,candidates)
        self._set_widget_value('flat_profile_band_period',w,value)

    def load(self,values,activated=False):
        with QSignalBlocker(self.activation_checkbox):
            self.activation_checkbox.setChecked(bool(activated))
        self._loading=True
        try:
            for d,w in self.widgets.items():
                v=values.get(d,self._defaults.get(d))
                self._set_widget_value(d,w,v)
            self._normalize_cutoff_pair('load',emit=False)
        finally: self._loading=False
        self.update_dependencies()
    def set_interaction_state(self,can_toggle_activation: bool,can_edit: bool):
        self.activation_checkbox.setEnabled(bool(can_toggle_activation))
        self.scroll.setEnabled(bool(can_edit))

    def update_dependencies(self):
        restormer=bool(self.value('restormer')) if 'restormer' in self.widgets else True
        ff=bool(self.value('flat_filter')) if 'flat_filter' in self.widgets else False
        fp=bool(self.value('flat_profile')) if 'flat_profile' in self.widgets else False
        profile_mode=str(self.value('flat_profile_mode')).lower() if 'flat_profile_mode' in self.widgets else 'smooth'
        pwm_profile = fp and profile_mode == 'pwm'
        feq=bool(self.value('flat_surface_equalizer')) if 'flat_surface_equalizer' in self.widgets else False
        orth=bool(self.value('orthogonal_profile')) if 'orthogonal_profile' in self.widgets else False
        device=str(self.value('device')).lower() if 'device' in self.widgets else 'auto'
        passes=int(self.value('passes')) if 'passes' in self.widgets else 1
        any_cleanup = ff or fp or feq or orth
        stage_enabled={'flat':ff,'profile':(fp or orth),'broad':feq}
        for stage,btn in self.mask_buttons.items():
            # If the user switches a stage off while its mask editor is open,
            # keep the checked Settings button usable so mask mode is never a trap.
            btn.setEnabled(bool(stage_enabled.get(stage,False) or self._active_mask_stage==stage))
        for d,w in self.widgets.items():
            enabled=True
            # Restormer-only controls follow the master Restormer switch, just
            # like the controls in the optional deterministic cleanup sections.
            # Band direction intentionally remains editable because the residual
            # profile / broad cleanup paths use the same axis even when the
            # neural Restormer stage is disabled.
            if d in {
                'device','amp','processing_size','passes',
                'first_pass_luma_strength','first_pass_chroma_strength',
                'second_pass_strength','luma_mode',
            }:
                enabled=restormer
            # AMP is meaningful only when Restormer is enabled and CUDA is
            # selected explicitly or may be selected by Auto. Preserve its
            # checked state while disabled.
            if d=='amp': enabled=restormer and (device=='auto' or device.startswith('cuda'))
            # Local surface safety only controls the local flat-region blend.
            if d in {'flat_luma_strength','flat_chroma_strength','flat_local_edge_distance'}: enabled=ff
            # Tone cutoffs are shared support gates for all deterministic cleanup
            # paths, so they remain editable when any such cleanup is active.
            if d in {'flat_highpass','flat_lowpass'}: enabled=any_cleanup
            # Main residual-profile controls depend only on their own enable flag.
            # Profile mode is shared with the optional orthogonal pass, so keep
            # that one selector editable when either residual-profile path runs.
            if d.startswith('flat_profile_'): enabled=fp
            # Final PWM polish is meaningful only for PWM / Step residual mode.
            # Keep the user's checked state while disabled, matching the other
            # optional sections, but never allow the polish controls to become
            # interactive when Smooth periodic is selected.
            if d == 'flat_profile_pwm_polish':
                enabled = pwm_profile
            if d in {'flat_profile_pwm_polish_strength','flat_profile_pwm_polish_passes'} and 'flat_profile_pwm_polish' in self.widgets:
                enabled = pwm_profile and bool(self.value('flat_profile_pwm_polish'))
            if d=='flat_profile_mode': enabled=(fp or orth)
            # Orthogonal cleanup shares the residual-profile algorithm but is
            # independently opt-in. Its strength controls follow its checkbox.
            if d.startswith('orthogonal_') and d!='orthogonal_profile': enabled=orth
            if d in {'flat_surface_equalizer_mode','flat_surface_equalizer_luma_strength','flat_surface_equalizer_chroma_strength'}: enabled=feq
            if d=='second_pass_strength': enabled=restormer and (passes==2)
            w.setEnabled(enabled)
            if d in self.reset_buttons:
                self.reset_buttons[d].setEnabled(enabled)
            if hasattr(w,'_period_auto'):
                w._period_spin.setEnabled(enabled and w._period_auto.currentData() is None)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle('Flicker Suppressor'); self.resize(1500,900)
        icon=resource_path('assets','logo_256.png');
        if icon.exists(): self.setWindowIcon(QIcon(str(icon)))
        self.docs={}; self.current_id=None; self.copied_settings=None; self.tasks={}; self.thread_pool=QThreadPool.globalInstance(); self.engine=FlickerEngine(resource_path('models'))
        self.state=load_state(); self.recent=list(self.state.get('recent',[]))[:12]; self.cache=Path(tempfile.gettempdir())/'FlickerSuppressorPreview'; self.cache.mkdir(exist_ok=True)
        self._export_jobs=[]; self._export_current=None; self._export_worker=None; self._export_temp=None; self._export_cancel_requested=False; self._mask_previous_view=None; self._mask_clipboard=None
        self._preview_processing_ids=set()
        self._build_ui(); self._build_menus()
        self.export_overlay=ExportOverlay(self); self.export_overlay.cancelRequested.connect(self.cancel_export)
        self.auto_overlay=AutoSettingsOverlay(self); self.auto_overlay.droppedPaths.connect(self.add_paths)
        self._auto_pending=0
        self.processing_overlay=ProcessingOverlay(self.canvas); self.processing_overlay.setGeometry(self.canvas.rect())
        self._update_recent(); self.update_ui_state()
    def _build_ui(self):
        root=QWidget(); self.setCentralWidget(root); main=QVBoxLayout(root); main.setContentsMargins(0,0,0,0); main.setSpacing(0)
        middle=QHBoxLayout(); middle.setContentsMargins(0,0,0,0); middle.setSpacing(0)
        canvas_col=QVBoxLayout(); canvas_col.setContentsMargins(0,0,0,0); canvas_col.setSpacing(0)
        self.canvas=ComparisonCanvas(); self.canvas.droppedPaths.connect(self.add_paths); canvas_col.addWidget(self.canvas,1)
        toolbar=QFrame(); toolbar.setObjectName('CanvasToolbar'); th=QHBoxLayout(toolbar); th.setContentsMargins(8,6,8,6); th.setSpacing(4)
        self.btn_single=QToolButton(); self.btn_single.setText('Single'); self.btn_single.setCheckable(True); self.btn_single.setChecked(True)
        self.btn_eye=EyeToggleButton(); self.btn_eye.setChecked(True); self.btn_eye.setToolTip('Showing edited image; click to show original'); self.btn_eye.setFixedWidth(38)
        self.btn_split=QToolButton(); self.btn_split.setText('Split'); self.btn_split.setCheckable(True)
        self.btn_side=QToolButton(); self.btn_side.setText('Side by side'); self.btn_side.setCheckable(True)
        view_h=max(self.btn_single.sizeHint().height(), self.btn_split.sizeHint().height(), self.btn_side.sizeHint().height())
        for _w in (self.btn_single,self.btn_eye,self.btn_split,self.btn_side): _w.setFixedHeight(view_h)
        self.view_group=QActionGroup(self); self.btn_single.clicked.connect(lambda:self.set_view('single')); self.btn_split.clicked.connect(lambda:self.set_view('split')); self.btn_side.clicked.connect(lambda:self.set_view('side')); self.btn_eye.toggled.connect(self.toggle_single_image)
        self.btn_fit=QToolButton(); self.btn_fit.setText('Fit'); self.btn_fit.clicked.connect(self.canvas.fit); self.btn_100=QToolButton(); self.btn_100.setText('100%'); self.btn_100.clicked.connect(self.canvas.zoom_100)
        self.btn_minus=QToolButton(); self.btn_minus.setText('−'); self.btn_minus.clicked.connect(lambda:self.canvas.set_zoom((self.canvas.zoom or .5)/1.2)); self.btn_plus=QToolButton(); self.btn_plus.setText('+'); self.btn_plus.clicked.connect(lambda:self.canvas.set_zoom((self.canvas.zoom or .5)*1.2))
        self.zoom_label=QLabel('Fit'); self.zoom_label.setObjectName('Muted'); self.zoom_label.setFixedWidth(48); self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.canvas.zoomChanged.connect(lambda z:self.zoom_label.setText('Fit' if z<=0 else f'{z*100:.0f}%'))
        # Pixel dimensions of the image currently on the canvas. Fixed width and
        # right alignment so the zoom buttons beside it do not shift as the
        # number of digits changes.
        self.resolution_label=QLabel(''); self.resolution_label.setObjectName('Muted'); self.resolution_label.setFixedWidth(180); self.resolution_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
        # Keep Single + eye as one tight visual group, then separate the
        # other comparison modes so they read as distinct choices.
        th.addWidget(self.btn_single)
        th.addWidget(self.btn_eye)
        th.addSpacing(12)
        th.addWidget(self.btn_split)
        th.addSpacing(12)
        th.addWidget(self.btn_side)
        th.addStretch(1)
        # Resolution readout, then zoom controls: Fit, 100%, minus, plus.
        th.addWidget(self.resolution_label)
        th.addSpacing(12)
        for w in [self.btn_fit,self.btn_100,self.btn_minus,self.btn_plus,self.zoom_label]: th.addWidget(w)
        canvas_col.addWidget(toolbar)
        cwrap=QWidget(); cwrap.setLayout(canvas_col); middle.addWidget(cwrap,1)
        self.settings_panel=SettingsPanel(); self.settings_panel.setFixedWidth(340); self.settings_panel.changed.connect(self.setting_changed); self.settings_panel.activationChanged.connect(self.activation_changed); self.settings_panel.exportSettingsRequested.connect(self.export_current_settings_json); self.settings_panel.importSettingsRequested.connect(self.import_current_settings_json); self.settings_panel.maskModeChanged.connect(self.mask_mode_changed); self.settings_panel.maskBrushChanged.connect(self.mask_brush_changed); self.settings_panel.maskCopyRequested.connect(self.copy_stage_mask); self.settings_panel.maskPasteRequested.connect(self.paste_stage_mask); self.settings_panel.maskInvertRequested.connect(self.invert_stage_mask); self.canvas.maskEdited.connect(self.mask_edited); middle.addWidget(self.settings_panel)
        main.addLayout(middle,1)
        bottom=QFrame(); bottom.setObjectName('BottomBar'); bh=QHBoxLayout(bottom); bh.setContentsMargins(8,7,8,7); bh.setSpacing(8)
        self.strip=ImageStrip(); self.strip.setObjectName('ImageStrip'); self.strip.setFixedHeight(108); self.strip.currentItemChanged.connect(self.current_item_changed); self.strip.itemSelectionChanged.connect(self.update_ui_state); self.strip.emptyContextRequested.connect(self.show_strip_context_menu); self.strip.itemContextRequested.connect(self.show_clip_context_menu); self.strip.droppedPaths.connect(self.add_paths); bh.addWidget(self.strip,1)
        self.btn_preview=QPushButton('Preview / Apply'); self.btn_preview.setObjectName('PrimaryButton'); self.btn_reset=QPushButton('Reset'); self.btn_export=QPushButton('Export'); self.btn_export.setObjectName('ExportMainButton'); self.btn_export_drop=DownChevronButton(); self.btn_export_drop.setObjectName('SplitDropButton'); self.btn_export_drop.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.export_menu=QMenu(self.btn_export_drop); self.exp_sel_drop=self.export_menu.addAction('Export selected...'); self.exp_all_drop=self.export_menu.addAction('Export all...'); self.btn_export_drop.setMenu(self.export_menu)
        for b in [self.btn_preview,self.btn_reset]: b.setFixedSize(132,38)
        self.btn_export.setFixedSize(96,38); self.btn_export_drop.setFixedSize(36,38)
        self.btn_preview.clicked.connect(self.preview_current); self.btn_reset.clicked.connect(self.reset_current); self.btn_export.clicked.connect(self.export_selected); self.exp_sel_drop.triggered.connect(self.export_selected); self.exp_all_drop.triggered.connect(self.export_all)
        bh.addWidget(self.btn_preview); bh.addWidget(self.btn_reset)
        expwrap=QWidget(); expwrap.setFixedSize(132,38); eh=QHBoxLayout(expwrap); eh.setContentsMargins(0,0,0,0); eh.setSpacing(0); eh.addWidget(self.btn_export); eh.addWidget(self.btn_export_drop); bh.addWidget(expwrap)
        main.addWidget(bottom)
    def _build_menus(self):
        mb=self.menuBar(); file=mb.addMenu('&File'); edit=mb.addMenu('&Edit'); select=mb.addMenu('&Select'); view=mb.addMenu('&View'); helpm=mb.addMenu('&Help')
        self.act_open=file.addAction('Open images...'); self.act_open.setShortcut(QKeySequence.StandardKey.Open); self.act_open.triggered.connect(self.open_images)
        self.menu_recent=file.addMenu('Open recent'); file.addSeparator(); self.act_close_sel=file.addAction('Close selected'); self.act_close_sel.triggered.connect(self.close_selected); self.act_close_all=file.addAction('Close all'); self.act_close_all.triggered.connect(self.close_all); file.addSeparator(); self.act_exp_sel=file.addAction('Export selected...'); self.act_exp_sel.triggered.connect(self.export_selected); self.act_exp_all=file.addAction('Export all...'); self.act_exp_all.triggered.connect(self.export_all); file.addSeparator(); file.addAction('Exit',self.close)
        self.act_copy=edit.addAction('Copy editing settings from selected'); self.act_copy.setShortcut(QKeySequence.StandardKey.Copy); self.act_copy.triggered.connect(self.copy_settings); self.act_paste=edit.addAction('Paste editing settings to selected'); self.act_paste.setShortcut(QKeySequence.StandardKey.Paste); self.act_paste.triggered.connect(self.paste_settings)
        edit.addSeparator()
        self.act_auto_settings=edit.addAction('Essential editing settings'); self.act_auto_settings.setCheckable(True)
        self.act_auto_settings.setChecked(bool(self.state.get('auto_editing_settings',True)))
        self.act_auto_settings.setToolTip('Determine the essential settings - band axis, period and cleanup mode - when an image is imported')
        self.act_auto_settings.toggled.connect(self._auto_settings_toggled)
        self.act_select_all=select.addAction('Select all'); self.act_select_all.setShortcut(QKeySequence.StandardKey.SelectAll); self.act_select_all.triggered.connect(self.select_all)
        self.act_deselect_all=select.addAction('Deselect all'); self.act_deselect_all.setShortcut(QKeySequence('Ctrl+Shift+A')); self.act_deselect_all.triggered.connect(self.deselect_all)
        self.act_fit=view.addAction('Zoom to fit'); self.act_fit.triggered.connect(self.canvas.fit); self.act_100=view.addAction('Zoom to 100%'); self.act_100.triggered.connect(self.canvas.zoom_100); view.addSeparator(); self.act_single=view.addAction('Single view'); self.act_single.triggered.connect(lambda:self.set_view('single')); self.act_split=view.addAction('Split view'); self.act_split.triggered.connect(lambda:self.set_view('split')); self.act_side=view.addAction('Side by side view'); self.act_side.triggered.connect(lambda:self.set_view('side'))
        helpm.addAction('About',self.about)
    def _update_recent(self):
        self.menu_recent.clear()
        for p in self.recent:
            a=self.menu_recent.addAction(Path(p).name); a.setToolTip(p); a.triggered.connect(lambda _=False,x=p:self.add_paths([x]) if Path(x).exists() else None)
        self.menu_recent.setEnabled(bool(self.recent))
    def show_strip_context_menu(self, global_pos):
        menu=QMenu(self)
        menu.addAction(self.act_open)
        menu.addSeparator()
        menu.addAction(self.act_close_all)
        menu.exec(global_pos)

    def show_clip_context_menu(self, global_pos):
        # Reuse the application's QAction objects so enabled/disabled state and
        # shortcuts always match the Edit/File menus. Selection/current-item
        # semantics are handled by ImageStrip before this signal is emitted.
        self.update_ui_state()
        menu=QMenu(self)
        menu.addAction(self.act_copy)
        menu.addAction(self.act_paste)
        menu.addSeparator()
        menu.addAction(self.act_close_sel)
        menu.exec(global_pos)
    def toggle_single_image(self, edited):
        edited=bool(edited)
        self.canvas.set_single_edited(edited)
        self.btn_eye.update()
        self.btn_eye.setToolTip('Showing edited image; click to show original' if edited else 'Showing original image; click to show edited')
    def open_images(self):
        paths,_=QFileDialog.getOpenFileNames(self,'Open images','',IMAGE_FILTER)
        if paths:self.add_paths(paths)
    def add_paths(self,paths):
        new_items=[]
        for s in paths:
            p=Path(s).resolve()
            # File dialogs already filter these, but drag-and-drop can contain
            # arbitrary files. Ignore directories and unsupported formats.
            if not p.is_file() or p.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
                continue
            if any(d.path==p for d in self.docs.values()):
                continue
            id_=uuid.uuid4().hex; doc=ImageDocument(id_,p); self.docs[id_]=doc
            item=ClipStripItem(p.name); item.setData(Qt.ItemDataRole.UserRole,id_); pm=QPixmap(str(p));
            # Always give the clip strip a full 72x54 thumbnail canvas. Without this,
            # KeepAspectRatio produces pixmaps with different actual heights/widths,
            # and Qt places the filename relative to that variable icon geometry.
            # A fixed transparent canvas keeps every label on the same baseline and
            # prevents portrait/landscape thumbnails from intruding into the text row.
            thumb=QPixmap(72,54); thumb.fill(Qt.GlobalColor.transparent)
            scaled=pm.scaled(72,54,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation)
            painter=QPainter(thumb); painter.drawPixmap((72-scaled.width())//2,(54-scaled.height())//2,scaled); painter.end()
            item.setThumbnail(thumb); item.setToolTip(str(p)); self.strip.addItem(item); new_items.append(item)
            if self.act_auto_settings.isChecked():
                self._queue_auto_settings(doc)
            ps=str(p); self.recent=[ps]+[x for x in self.recent if x!=ps]; self.recent=self.recent[:12]
        self.state['recent']=self.recent; save_state(self.state); self._update_recent()
        if new_items:
            # Import semantics: once an import batch finishes, only the clips
            # from that batch are selected. This applies equally to the file
            # dialog, drag/drop, and Open Recent because all feed add_paths().
            self.strip.clearSelection()
            for item in new_items:
                item.setSelected(True)
            # Make the first imported clip current without collapsing the
            # freshly-created multi-selection.
            self.strip.setCurrentItem(new_items[0], QItemSelectionModel.SelectionFlag.NoUpdate)
        self.update_ui_state()
    def select_all(self):
        if self.strip.count()==0:
            return
        self.strip.selectAll()
        if self.strip.currentItem() is None:
            self.strip.setCurrentItem(self.strip.item(0), QItemSelectionModel.SelectionFlag.NoUpdate)
        self.update_ui_state()

    def deselect_all(self):
        self.strip.clearSelection()
        self.update_ui_state()

    def current_doc(self): return self.docs.get(self.current_id)
    def selected_docs(self):
        out=[]
        for i in self.strip.selectedItems():
            d=self.docs.get(i.data(Qt.ItemDataRole.UserRole));
            if d:out.append(d)
        return out
    def current_item_changed(self,item,_prev):
        if self.settings_panel.active_mask_stage:
            self.settings_panel.exit_mask_mode()
        # Multi-selection and the current/canvas item are intentionally distinct.
        # ImageStrip paints a crisp red frame on the current tile.
        self.current_id=item.data(Qt.ItemDataRole.UserRole) if item else None; d=self.current_doc()
        if d:
            self.settings_panel.set_period_candidates((getattr(d,'auto',None) or {}).get('candidates'))
            self.settings_panel.load(d.settings,d.activated); self.load_canvas(d)
            self._maybe_warn_auto_settings(d)
        else:
            self.canvas.set_images(None,None); self._update_resolution_label(None)
        self.update_ui_state()
    def _auto_settings_toggled(self,checked):
        self.state['auto_editing_settings']=bool(checked); save_state(self.state)

    def _queue_auto_settings(self,doc):
        """Estimate settings for one freshly imported image, off the UI thread."""
        worker=AutoSettingsWorker(doc.id,doc.path)
        worker.signals.finished.connect(self._auto_settings_ready)
        worker.signals.error.connect(self._auto_settings_failed)
        self._auto_pending+=1
        self._sync_auto_overlay()
        self.thread_pool.start(worker)

    def _sync_auto_overlay(self):
        """Show while any estimate is outstanding, hide when the last finishes."""
        if self._auto_pending > 0:
            self.auto_overlay.setGeometry(self.rect())
            self.auto_overlay.show()
            self.auto_overlay.raise_()
        else:
            self.auto_overlay.hide()

    def _auto_settings_done(self):
        self._auto_pending=max(0,self._auto_pending-1)
        self._sync_auto_overlay()

    def _auto_settings_failed(self,doc_id,_msg):
        # An estimate that crashes must not block the image. Treat it as "could
        # not determine" and fall through to defaults.
        d=self.docs.get(doc_id)
        if d is not None:
            d.auto={'confidence':'low','reason':'the estimator failed on this image',
                    'candidates':[],'settings':{}}
        self._auto_settings_done()

    def _auto_settings_ready(self,doc_id,_src,result):
        self._auto_settings_done()
        d=self.docs.get(doc_id)
        if d is None or not isinstance(result,dict):
            return
        d.auto=result
        if d.dirty or d.processing:
            # The user has already edited or is mid-render; do not overwrite.
            return
        if result.get('confidence')=='low':
            # Defaults, not the previous image's settings: carrying a period
            # tuned for another photo across is worse than starting neutral.
            d.settings=dict(default_settings())
        else:
            merged=dict(d.settings); merged.update(result.get('settings') or {})
            d.settings=merged
        if d.id==self.current_id:
            self.settings_panel.set_period_candidates(result.get('candidates'))
            self.settings_panel.load(d.settings,d.activated)
            self._maybe_warn_auto_settings(d)

    def _maybe_warn_auto_settings(self,d):
        """Once per image, tell the user the estimate was inconclusive."""
        if d.auto is None or d.auto_notified:
            return
        if d.auto.get('confidence')!='low':
            return
        d.auto_notified=True
        box=QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle('Essential editing settings')
        box.setText('Essential editing settings could not be determined for the current image')
        # The estimator's reason is diagnostic ("colour ratios disagree by
        # 5.71x") and unhelpful to someone deciding what to do next. It stays in
        # doc.auto['reason'] for the CLI and for logs.
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    def load_canvas(self,d):
        orig=QImage(str(d.path)); edit=QImage(str(d.preview_path)) if d.preview_path and d.preview_path.exists() else QImage(); self.canvas.set_images(orig,edit)
        self._update_resolution_label(orig)

    def _update_resolution_label(self,image=None):
        """Show the source pixel dimensions of the image on the canvas.

        Source rather than preview: preview and export are full resolution so
        the two agree, but reading the source means the label is right before a
        preview exists and does not change while one is being rendered.
        """
        if image is None or image.isNull():
            self.resolution_label.setText('')
            return
        self.resolution_label.setText(f'Resolution: {image.width()} x {image.height()} px')
    def set_view(self,mode):
        if self.settings_panel.active_mask_stage and mode!='single':
            return
        self.canvas.set_mode(mode); self.btn_single.setChecked(mode=='single'); self.btn_split.setChecked(mode=='split'); self.btn_side.setChecked(mode=='side'); self.btn_eye.setVisible(True); self.toggle_single_image(self.btn_eye.isChecked()); self.update_ui_state()
    def activation_changed(self,checked):
        d=self.current_doc()
        if not d or d.processing or self._export_active():
            return
        d.activated=bool(checked)
        if not d.activated:
            if self.settings_panel.active_mask_stage:
                self.settings_panel.exit_mask_mode()
            with QSignalBlocker(self.btn_eye):
                self.btn_eye.setChecked(False)
            self.canvas.set_single_edited(False)
            if self.canvas.mode!='single':
                self.set_view('single')
        elif d.preview_path and d.preview_path.exists() and not d.dirty:
            with QSignalBlocker(self.btn_eye):
                self.btn_eye.setChecked(True)
            self.canvas.set_single_edited(True)
        self.update_ui_state()

    def _ensure_full_stage_mask(self,d,stage):
        """Lazily author a new cleanup mask as 100% selected everywhere."""
        if not d:
            return False
        stage=str(stage)
        if stage in d.mask_authored:
            return False
        if d is self.current_doc() and not self.canvas.original.isNull():
            size=self.canvas.original.size()
        else:
            source=QImage(str(d.path))
            if source.isNull():
                return False
            size=source.size()
        mask=QImage(size,QImage.Format.Format_RGBA8888)
        # RGB is only for the editor's red cast; alpha is the processing gate.
        mask.fill(QColor(255,32,32,255))
        d.masks[stage]=mask
        d.mask_authored.add(stage)
        return True

    def mask_mode_changed(self,stage,active):
        d=self.current_doc()
        if active:
            if not d or not d.activated or d.processing or self._export_active():
                if self.settings_panel.active_mask_stage:
                    QTimer.singleShot(0,self.settings_panel.exit_mask_mode)
                return
            # First use of a mask authors a full-coverage gate. This makes an
            # untouched mask explicit and removes the old ambiguity between
            # "never authored" and "authored then erased to zero".
            self._ensure_full_stage_mask(d,stage)
            if self._mask_previous_view is None:
                self._mask_previous_view=(self.canvas.mode,bool(self.btn_eye.isChecked()))
            if self.canvas.mode!='single':
                self.canvas.set_mode('single')
                self.btn_single.setChecked(True); self.btn_split.setChecked(False); self.btn_side.setChecked(False)
            mask=d.masks.get(stage)
            self.canvas.set_mask_mode(stage,mask,stage in d.mask_authored)
            params=self.settings_panel.current_mask_parameters(stage)
            self.canvas.set_mask_brush(params['tool'],params['size'],params['feather'],params['opacity'])
            self.update_ui_state()
            return

        self.canvas.set_mask_mode(None)
        previous=self._mask_previous_view
        self._mask_previous_view=None
        if previous is not None:
            mode,eye=previous
            d=self.current_doc()
            edited=bool(d and d.preview_path and d.preview_path.exists() and d.activated)
            if mode in {'split','side'} and not edited:
                mode='single'
            self.canvas.set_mode(mode)
            self.btn_single.setChecked(mode=='single'); self.btn_split.setChecked(mode=='split'); self.btn_side.setChecked(mode=='side')
            with QSignalBlocker(self.btn_eye): self.btn_eye.setChecked(bool(eye))
            self.canvas.set_single_edited(bool(eye))
        self.update_ui_state()

    def mask_brush_changed(self,stage,params):
        if self.canvas.mask_stage!=stage or not isinstance(params,dict):
            return
        self.canvas.set_mask_brush(params.get('tool','eraser'),params.get('size',160),params.get('feather',32),params.get('opacity',100))

    def mask_edited(self,stage,mask,authored):
        d=self.current_doc()
        if not d or not d.activated or not isinstance(mask,QImage):
            return
        stage=str(stage)
        d.masks[stage]=mask.copy()
        if authored:
            d.mask_authored.add(stage)
        else:
            d.mask_authored.discard(stage)
        d.dirty=True
        self.update_ui_state()

    def copy_stage_mask(self,stage):
        d=self.current_doc()
        if not d or not d.activated:
            return
        stage=str(stage)
        if self.canvas.mask_stage==stage:
            mask,authored=self.canvas.current_mask()
        else:
            mask=d.masks.get(stage,QImage()).copy()
            authored=stage in d.mask_authored
        self._mask_clipboard=(mask.copy(),bool(authored))
        self.settings_panel.set_mask_paste_available(True)

    def paste_stage_mask(self,stage):
        d=self.current_doc()
        if not d or not d.activated or self._mask_clipboard is None:
            return
        stage=str(stage)
        source,authored=self._mask_clipboard
        if self.canvas.mask_stage==stage:
            self.canvas.replace_current_mask(source,authored)
            return
        d.masks[stage]=source.copy()
        if authored:
            d.mask_authored.add(stage)
        else:
            d.mask_authored.discard(stage)
        d.dirty=True
        self.update_ui_state()

    def invert_stage_mask(self,stage):
        d=self.current_doc()
        if not d or not d.activated:
            return
        stage=str(stage)
        if self.canvas.mask_stage==stage:
            self.canvas.invert_current_mask()

    def _mask_paths_for_doc(self,d):
        if not d or not d.mask_authored:
            return {}
        out={}
        mask_dir=self.cache/'masks'; mask_dir.mkdir(parents=True,exist_ok=True)
        for stage in ('flat','profile','broad'):
            if stage not in d.mask_authored:
                continue
            image=d.masks.get(stage)
            if not isinstance(image,QImage) or image.isNull():
                continue
            target=mask_dir/f'{d.id}_{stage}.png'
            if not image.save(str(target),'PNG'):
                raise RuntimeError(f'Could not save {stage} mask for processing')
            out[stage]=target
        return out

    def setting_changed(self,dest,value):
        d=self.current_doc();
        if not d or not d.activated:return
        d.settings[dest]=value
        stage=MASK_STAGE_SETTING.get(str(dest))
        if stage is not None and bool(value):
            self._ensure_full_stage_mask(d,stage)
        d.dirty=True; self.update_ui_state()
    def reset_current(self):
        d=self.current_doc();
        if not d:return
        if not d.activated:return
        if self.settings_panel.active_mask_stage:
            self.settings_panel.exit_mask_mode()
        # Reset only the processing recipe. Keep the last rendered preview on
        # screen as a visual reference, but mark it stale so Preview/Export
        # recomputes from the reset settings instead of reusing that cache.
        had_cached_preview=self._has_cached_preview(d)
        d.settings=dict(default_settings()); d.dirty=had_cached_preview; d.masks.clear(); d.mask_authored.clear(); self.settings_panel.load(d.settings,d.activated); self.load_canvas(d); self.update_ui_state()
    def copy_settings(self):
        # Copy is intentionally a single-image operation. Multi-selection is
        # reserved for paste/export actions so there is never ambiguity about
        # which document supplied the settings payload.
        selected_docs=self.selected_docs()
        if len(selected_docs)!=1:
            return
        d=selected_docs[0]
        if not d.activated or d.processing or self._export_active():
            return
        self.copied_settings=dict(d.settings); QApplication.clipboard().setText(json.dumps(self.copied_settings))
        self.update_ui_state()
    def paste_settings(self):
        targets=self.selected_docs()
        if not targets or self._export_active() or any(d.processing for d in targets):
            return
        if self.copied_settings is None:
            try:self.copied_settings=json.loads(QApplication.clipboard().text())
            except Exception:return
        if not isinstance(self.copied_settings,dict):
            return
        # Paste is allowed onto activated and inactive images alike. Applying a
        # copied editing recipe opts every target into Flicker Suppressor, then
        # marks its cached preview stale so Preview/Export will recompute it.
        for d in targets:
            d.activated=True
            d.settings=dict(self.copied_settings)
            for setting,stage in MASK_STAGE_SETTING.items():
                if bool(d.settings.get(setting,False)):
                    self._ensure_full_stage_mask(d,stage)
            d.dirty=True
        current=self.current_doc()
        if current:
            self.settings_panel.load(current.settings,current.activated)
        self.update_ui_state()
    @staticmethod
    def _json_safe_setting(value):
        """Convert parser/namespace values to portable JSON primitives."""
        if isinstance(value,Path):
            return str(value)
        if isinstance(value,dict):
            return {str(k):MainWindow._json_safe_setting(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):
            return [MainWindow._json_safe_setting(v) for v in value]
        return value

    def export_current_settings_json(self):
        d=self.current_doc()
        if not d or not d.activated or d.processing or self._export_active():
            return

        # namespace() is the same merge used by inference: GUI-visible values
        # override authoritative CLI defaults, and GUI policy overrides such as
        # exposure_lock='all' are applied afterward. Thus the JSON contains the
        # actual complete processing configuration rather than only visible UI
        # controls. Input/output/model/debug plumbing is intentionally excluded
        # by settings_schema._all_parser_defaults().
        complete=vars(namespace(d.settings))
        payload={k:self._json_safe_setting(v) for k,v in complete.items()}

        suggested=d.path.parent/f'{d.path.stem}_settings.json'
        filename,_=QFileDialog.getSaveFileName(
            self,
            'Export processing settings',
            str(suggested),
            'JSON files (*.json);;All files (*)',
        )
        if not filename:
            return
        target=Path(filename)
        if target.suffix.lower()!='.json':
            target=Path(str(target)+'.json')
            # If the suffix was appended after the native dialog returned, Qt
            # may not have performed its normal overwrite confirmation for this
            # exact final path. Confirm explicitly in that case.
            if target.exists():
                answer=QMessageBox.question(
                    self,
                    'Export settings',
                    f'{target.name} already exists. Replace it?',
                    QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer!=QMessageBox.StandardButton.Yes:
                    return
        try:
            target.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        except Exception as exc:
            QMessageBox.critical(self,'Export settings',f'Could not export settings:\n{exc}')

    def import_current_settings_json(self):
        d=self.current_doc()
        if not d or not d.activated or d.processing or self._export_active():
            return

        filename,_=QFileDialog.getOpenFileName(
            self,
            'Import processing settings',
            str(d.path.parent),
            'JSON files (*.json);;All files (*)',
        )
        if not filename:
            return
        source=Path(filename)
        try:
            text=source.read_text(encoding='utf-8-sig')
        except Exception as exc:
            QMessageBox.critical(self,'Import settings',f'Could not read settings file:\n{exc}')
            return
        try:
            payload=json.loads(text)
        except json.JSONDecodeError as exc:
            QMessageBox.critical(
                self,
                'Import settings',
                f'Invalid JSON at line {exc.lineno}, column {exc.colno}:\n{exc.msg}',
            )
            return
        except Exception as exc:
            QMessageBox.critical(self,'Import settings',f'Could not parse JSON:\n{exc}')
            return

        try:
            imported=dict(validate_imported_settings(payload))
        except ValueError as exc:
            QMessageBox.critical(
                self,
                'Import settings',
                'This file is not a valid Flicker Suppressor processing-settings JSON.\n\n'
                f'{exc}',
            )
            return
        except Exception as exc:
            QMessageBox.critical(self,'Import settings',f'Could not validate settings:\n{exc}')
            return

        if self.settings_panel.active_mask_stage:
            self.settings_panel.exit_mask_mode()

        # Developer JSON contains processing values only. Preserve any authored
        # masks already attached to this image, but ensure newly enabled masked
        # stages have a full-coverage mask if no authored mask exists yet.
        d.settings=imported
        for setting,stage in MASK_STAGE_SETTING.items():
            if bool(d.settings.get(setting,False)):
                self._ensure_full_stage_mask(d,stage)
        d.dirty=True
        self.settings_panel.load(d.settings,d.activated)
        self.load_canvas(d)
        self.update_ui_state()

    def preview_current(self):
        d=self.current_doc()
        if not d or not d.activated or d.processing or self._export_active():
            return
        if self.settings_panel.active_mask_stage:
            self.settings_panel.exit_mask_mode()
        out=self.cache/f'{d.id}.png'
        self._run_preview_task(d,out)

    def _run_preview_task(self,d,out):
        d.processing=True
        self._preview_processing_ids.add(d.id)
        self._sync_thumbnail_processing(d.id)
        self._sync_processing_overlay()
        self.update_ui_state()
        w=InferenceWorker(d.id,self.engine,d.path,out,d.settings,masks=self._mask_paths_for_doc(d))
        self.tasks[d.id]=w
        w.signals.finished.connect(self.task_finished)
        w.signals.error.connect(self.task_error)
        w.signals.cancelled.connect(lambda _id:self.task_done(_id))
        self.thread_pool.start(w)

    def task_finished(self,id_,path,_stats):
        self._preview_processing_ids.discard(id_)
        d=self.docs.get(id_)
        if not d:
            self._sync_processing_overlay()
            self._sync_thumbnail_processing(id_)
            self.tasks.pop(id_,None)
            self.update_ui_state()
            return
        d.processing=False
        self._sync_thumbnail_processing(id_)
        self._sync_processing_overlay()
        d.preview_path=Path(path)
        d.dirty=False
        if d.id==self.current_id:
            with QSignalBlocker(self.btn_eye):
                self.btn_eye.setChecked(True)
            self.canvas.set_single_edited(True)
            self.load_canvas(d)
        self.tasks.pop(id_,None)
        self.update_ui_state()

    def task_error(self,id_,trace):
        self.task_done(id_)
        QMessageBox.critical(self,'Processing error',trace)

    def task_done(self,id_):
        self._preview_processing_ids.discard(id_)
        d=self.docs.get(id_)
        if d:
            d.processing=False
        self._sync_thumbnail_processing(id_)
        self._sync_processing_overlay()
        self.tasks.pop(id_,None)
        self.update_ui_state()

    def _sync_thumbnail_processing(self,id_=None):
        """Mirror document processing state on the corresponding clip tile."""
        for row in range(self.strip.count()):
            item=self.strip.item(row)
            if item is None:
                continue
            item_id=item.data(Qt.ItemDataRole.UserRole)
            if id_ is not None and item_id!=id_:
                continue
            d=self.docs.get(item_id)
            item.setProcessing(bool(d and d.processing))

    def _sync_processing_overlay(self):
        if not hasattr(self,'processing_overlay'):
            return
        self.processing_overlay.setGeometry(self.canvas.rect())
        if self._preview_processing_ids:
            self.processing_overlay.show()
            self.processing_overlay.raise_()
        else:
            self.processing_overlay.hide()

    def _export_active(self):
        if not hasattr(self,'export_overlay'):
            return False
        return self._export_current is not None or bool(self._export_jobs) or self._export_worker is not None or self.export_overlay.isVisible()

    @staticmethod
    def _has_cached_preview(d):
        return bool(d.preview_path and d.preview_path.exists())

    def _activated_docs(self):
        return [d for d in self.docs.values() if d.activated]

    def _choose_single_export_target(self,d):
        dlg=QFileDialog(self,'Export image',str(d.path.parent),'PNG image (*.png)')
        dlg.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dlg.setFileMode(QFileDialog.FileMode.AnyFile)
        dlg.setNameFilter('PNG image (*.png)')
        dlg.setDefaultSuffix('png')
        dlg.setOption(QFileDialog.Option.DontConfirmOverwrite,True)
        dlg.selectFile(d.path.stem+'.png')
        if dlg.exec()!=QDialog.DialogCode.Accepted:
            return None
        files=dlg.selectedFiles()
        if not files:
            return None
        target=Path(files[0])
        if target.suffix.lower()!='.png':
            target=target.with_suffix('.png')
        return target

    def _choose_batch_export(self,title):
        dlg=BatchExportDialog(title,self)
        if dlg.exec()!=QDialog.DialogCode.Accepted:
            return None
        return dlg.values()

    @staticmethod
    def _build_batch_jobs(docs,outdir,prefix,suffix):
        jobs=[]
        used=set()
        for d in docs:
            base=f'{prefix}{d.path.stem}{suffix}'
            candidate=outdir/f'{base}.png'
            key=candidate.name.casefold()
            counter=1
            while key in used:
                candidate=outdir/f'{base} ({counter}).png'
                key=candidate.name.casefold()
                counter+=1
            used.add(key)
            jobs.append(ExportJob(d.id,candidate))
        return jobs

    def _preflight_existing_outputs(self,jobs):
        approved=[]
        for job in jobs:
            if not job.target.exists():
                approved.append(job)
                continue
            box=QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle('File already exists')
            box.setText(f'An output file already exists:\n\n{job.target.name}')
            box.setInformativeText(f'{job.target.parent}\n\nReplace the existing file or skip this image?')
            replace_button=box.addButton('Replace',QMessageBox.ButtonRole.AcceptRole)
            skip_button=box.addButton('Skip',QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(skip_button)
            box.exec()
            if box.clickedButton() is replace_button:
                approved.append(job)
        return approved

    def export_selected(self):
        if self._export_active():
            return
        # Export Selected uses only selected documents for which Flicker
        # Suppressor has explicitly been activated. Previewing first is optional;
        # missing/stale cached results are rendered during export.
        docs=[d for d in self.selected_docs() if d.activated]
        if not docs or any(d.processing for d in docs):
            return
        if len(docs)==1:
            target=self._choose_single_export_target(docs[0])
            if target is None:
                return
            jobs=[ExportJob(docs[0].id,target)]
        else:
            values=self._choose_batch_export('Export selected images')
            if values is None:
                return
            outdir,prefix,suffix=values
            jobs=self._build_batch_jobs(docs,outdir,prefix,suffix)
        jobs=self._preflight_existing_outputs(jobs)
        if jobs:
            self._start_export_jobs(jobs)

    def export_all(self):
        if self._export_active():
            return
        # Export All uses every document for which Flicker Suppressor is
        # activated. A prior Preview / Apply is not required.
        docs=self._activated_docs()
        if not docs or any(d.processing for d in docs):
            return
        values=self._choose_batch_export('Export all activated images')
        if values is None:
            return
        outdir,prefix,suffix=values
        jobs=self._build_batch_jobs(docs,outdir,prefix,suffix)
        jobs=self._preflight_existing_outputs(jobs)
        if jobs:
            self._start_export_jobs(jobs)

    def _start_export_jobs(self,jobs):
        if self.settings_panel.active_mask_stage:
            self.settings_panel.exit_mask_mode()
        self._export_jobs=list(jobs)
        self._export_current=None
        self._export_worker=None
        self._export_temp=None
        self._export_cancel_requested=False
        self.centralWidget().setEnabled(False)
        self.menuBar().setEnabled(False)
        self.export_overlay.setGeometry(self.rect())
        self.export_overlay.show()
        self.export_overlay.raise_()
        self.update_ui_state()
        QTimer.singleShot(0,self._start_next_export_job)

    def _start_next_export_job(self):
        if self._export_cancel_requested or not self._export_jobs:
            self._finish_export_session()
            return
        job=self._export_jobs.pop(0)
        d=self.docs.get(job.doc_id)
        if d is None:
            QTimer.singleShot(0,self._start_next_export_job)
            return
        self._export_current=job
        d.processing=True
        self._sync_thumbnail_processing(d.id)

        if d.preview_path and d.preview_path.exists() and not d.dirty:
            self._start_export_copy(Path(d.preview_path),job.target)
            return

        cache_out=self.cache/f'{d.id}.png'
        self._export_temp=None
        task_id='export-process-'+uuid.uuid4().hex
        worker=InferenceWorker(task_id,self.engine,d.path,cache_out,d.settings,masks=self._mask_paths_for_doc(d))
        self._export_worker=worker
        worker.signals.finished.connect(self._export_inference_finished)
        worker.signals.error.connect(self._export_worker_error)
        worker.signals.cancelled.connect(self._export_worker_cancelled)
        self.thread_pool.start(worker)

    def _start_export_copy(self,src,target):
        if self._export_cancel_requested:
            self._export_worker_cancelled('export-copy-cancelled')
            return
        task_id='export-copy-'+uuid.uuid4().hex
        worker=CopyWorker(task_id,src,target)
        self._export_worker=worker
        worker.signals.finished.connect(self._export_copy_finished)
        worker.signals.error.connect(self._export_worker_error)
        worker.signals.cancelled.connect(self._export_worker_cancelled)
        self.thread_pool.start(worker)

    def _export_inference_finished(self,_task_id,path,_stats):
        self._export_worker=None
        if self._export_current is None:
            return
        # A render completed during export is a valid preview cache too. Keep it
        # even if the user cancels before/during the final file copy so it can
        # be reused later rather than needlessly recomputed.
        d=self.docs.get(self._export_current.doc_id)
        if d is not None:
            d.preview_path=Path(path)
            d.dirty=False
            if d.id==self.current_id:
                self.load_canvas(d)
        if self._export_cancel_requested:
            self._export_worker_cancelled('export-process-cancelled')
            return
        self._start_export_copy(Path(path),self._export_current.target)

    def _export_copy_finished(self,_task_id,_path,_stats):
        self._export_worker=None
        self._complete_current_export_job()
        QTimer.singleShot(0,self._start_next_export_job)

    def _complete_current_export_job(self):
        if self._export_current is not None:
            d=self.docs.get(self._export_current.doc_id)
            if d:
                d.processing=False
                self._sync_thumbnail_processing(d.id)
        if self._export_temp is not None:
            try:
                self._export_temp.unlink(missing_ok=True)
            except OSError:
                pass
        self._export_temp=None
        self._export_current=None

    def _export_worker_error(self,_task_id,trace):
        self._export_worker=None
        self._complete_current_export_job()
        self._finish_export_session()
        QMessageBox.critical(self,'Export error',trace)

    def _export_worker_cancelled(self,_task_id):
        self._export_worker=None
        self._complete_current_export_job()
        self._export_jobs.clear()
        self._finish_export_session()

    def cancel_export(self):
        if not self._export_active():
            return
        self._export_cancel_requested=True
        self._export_jobs.clear()
        self.export_overlay.set_cancelling(True)
        worker=self._export_worker
        if worker is not None and hasattr(worker,'cancel'):
            worker.cancel()
        elif self._export_current is None:
            self._finish_export_session()

    def _finish_export_session(self):
        self._complete_current_export_job()
        self._export_jobs.clear()
        self._export_worker=None
        self._export_cancel_requested=False
        if hasattr(self,'export_overlay'):
            self.export_overlay.hide()
        self.centralWidget().setEnabled(True)
        self.menuBar().setEnabled(True)
        self.update_ui_state()

    def close_selected(self):
        if self.settings_panel.active_mask_stage:
            self.settings_panel.exit_mask_mode()
        ids={d.id for d in self.selected_docs()}
        for r in range(self.strip.count()-1,-1,-1):
            it=self.strip.item(r)
            if it.data(Qt.ItemDataRole.UserRole) in ids:self.strip.takeItem(r)
        for i in ids:self.docs.pop(i,None)
        self.current_id=self.strip.currentItem().data(Qt.ItemDataRole.UserRole) if self.strip.currentItem() else None; self.update_ui_state()
    def close_all(self):
        if self.settings_panel.active_mask_stage:
            self.settings_panel.exit_mask_mode()
        self.strip.clear(); self.docs.clear(); self.current_id=None; self.canvas.set_images(None,None); self.update_ui_state()
    def update_ui_state(self):
        any_docs=bool(self.docs)
        selected_docs=self.selected_docs()
        selected=bool(selected_docs)
        d=self.current_doc()
        current=d is not None
        current_selected=bool(d and any(sel.id == d.id for sel in selected_docs))
        active=current and current_selected
        activated=bool(d and d.activated)
        cached_preview=bool(d and d.preview_path and d.preview_path.exists())
        edited=activated and cached_preview
        busy=bool(d and d.processing)
        any_processing=any(doc.processing for doc in self.docs.values())
        exporting=self._export_active()
        masking=bool(self.settings_panel.active_mask_stage)
        export_all_available=bool(self._activated_docs())
        export_selected_available=any(doc.activated for doc in selected_docs)
        can_export_selected=export_selected_available and not any_processing and not exporting
        can_export_all=export_all_available and not any_processing and not exporting

        can_toggle_activation=active and not busy and not exporting
        can_edit=can_toggle_activation and activated
        self.settings_panel.set_interaction_state(can_toggle_activation,can_edit)

        self.act_close_sel.setEnabled(selected and not exporting)
        self.act_close_all.setEnabled(any_docs and not exporting)
        self.act_exp_sel.setEnabled(can_export_selected)
        self.act_exp_all.setEnabled(can_export_all)
        can_copy_settings=(
            len(selected_docs)==1
            and selected_docs[0].activated
            and not selected_docs[0].processing
            and not exporting
        )
        can_paste_settings=(
            selected
            and self.copied_settings is not None
            and not any(doc.processing for doc in selected_docs)
            and not exporting
        )
        self.act_copy.setEnabled(can_copy_settings)
        self.act_paste.setEnabled(can_paste_settings)
        self.act_select_all.setEnabled(any_docs and not exporting)
        self.act_deselect_all.setEnabled(selected and not exporting)

        self.btn_preview.setEnabled(can_edit)
        self.btn_reset.setEnabled(can_edit and bool(d.dirty or cached_preview))
        self.btn_export.setEnabled(can_export_selected)
        self.btn_export_drop.setEnabled(can_export_selected or can_export_all)
        self.exp_sel_drop.setEnabled(can_export_selected)
        self.exp_all_drop.setEnabled(can_export_all)

        self.btn_single.setEnabled(active and not masking)
        for w in [self.btn_fit,self.btn_100,self.btn_minus,self.btn_plus]:
            w.setEnabled(active)
        self.btn_eye.setEnabled(active and edited and self.canvas.mode=='single')
        self.btn_split.setEnabled(active and edited and not masking)
        self.btn_side.setEnabled(active and edited and not masking)
        self.act_split.setEnabled(active and edited and not masking)
        self.act_side.setEnabled(active and edited and not masking)
        self.act_single.setEnabled(active and not masking)
        if not edited:
            with QSignalBlocker(self.btn_eye):
                self.btn_eye.setChecked(False)
            self.canvas.set_single_edited(False)
            if not activated:
                self.btn_eye.setToolTip('Flicker Suppressor is not activated for this image')
            else:
                self.btn_eye.setToolTip('Original image (edited preview not available yet)')
            self.btn_eye.update()
        else:
            self.btn_eye.setToolTip('Showing edited image; click to show original' if self.btn_eye.isChecked() else 'Showing original image; click to show edited')
            self.btn_eye.update()

        self.act_fit.setEnabled(active)
        self.act_100.setEnabled(active)
        self.act_single.setEnabled(active and not masking)
        self.act_split.setEnabled(active and edited and not masking)
        self.act_side.setEnabled(active and edited and not masking)
        if not edited and self.canvas.mode!='single':
            self.set_view('single')
    def resizeEvent(self,event):
        super().resizeEvent(event)
        if hasattr(self,'processing_overlay'):
            self.processing_overlay.setGeometry(self.canvas.rect())
            if self.processing_overlay.isVisible():
                self.processing_overlay.raise_()
        if hasattr(self,'export_overlay'):
            self.export_overlay.setGeometry(self.rect())
            if self.export_overlay.isVisible():
                self.export_overlay.raise_()
        if hasattr(self,'auto_overlay'):
            self.auto_overlay.setGeometry(self.rect())
            if self.auto_overlay.isVisible():
                self.auto_overlay.raise_()

    def closeEvent(self,event):
        if self._export_active():
            event.ignore()
            return
        super().closeEvent(event)

    def about(self):
        dlg=QDialog(self)
        dlg.setWindowTitle('About Flicker Suppressor')
        dlg.setMinimumWidth(560)
        v=QVBoxLayout(dlg)

        about_text=QLabel(
            '<div style="line-height: 1.35;">'
            '<b>Flicker Suppressor by mattaja</b><br><br>'
            'contributions by:<br>'
            '- Restormer: <a href="https://github.com/swz30/Restormer">https://github.com/swz30/Restormer</a> - MIT License.<br>'
            '- BurstDeflicker: <a href="https://github.com/qulishen/BurstDeflicker">https://github.com/qulishen/BurstDeflicker</a> - Apache License 2.0.<br>'
            '- BasicSR: <a href="https://github.com/XPixelGroup/BasicSR">https://github.com/XPixelGroup/BasicSR</a> - Apache License 2.0.<br><br>'
            'Flicker Suppressor is released under the Apache License 2.0<br><br>'
            'Souce code and releases hosted at<br>'
            '<a href="https://github.com/GianSegugio/flicker_suppressor">https://github.com/GianSegugio/flicker_suppressor</a>'
            '</div>'
        )
        about_text.setTextFormat(Qt.TextFormat.RichText)
        about_text.setOpenExternalLinks(True)
        about_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        about_text.setWordWrap(True)
        about_text.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        v.addWidget(about_text)

        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dlg.accept)
        v.addWidget(buttons)
        dlg.exec()
