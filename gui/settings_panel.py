from __future__ import annotations

from collections import defaultdict

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QScrollArea, QSlider, QSpinBox, QToolButton,
    QVBoxLayout, QWidget,
)

from .settings_schema import GROUP_ORDER, SettingSpec, default_settings, specs


class CollapsibleSection(QWidget):
    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self.toggle = QToolButton(text=title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.toggle.setStyleSheet("QToolButton { border: none; font-weight: 600; padding: 8px 2px; text-align: left; }")
        self.content = QWidget()
        self.form = QFormLayout(self.content)
        self.form.setContentsMargins(4, 0, 4, 8)
        self.form.setHorizontalSpacing(10)
        self.form.setVerticalSpacing(7)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.content.setVisible(expanded)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toggle)
        layout.addWidget(self.content)
        self.toggle.toggled.connect(self._toggle)

    def _toggle(self, checked: bool) -> None:
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self.content.setVisible(checked)


class FloatSlider(QWidget):
    valueChanged = Signal(float)

    def __init__(self, minimum: float, maximum: float, step: float = 0.01, parent=None):
        super().__init__(parent)
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self._factor = max(1, round(1.0 / step))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(round(minimum * self._factor), round(maximum * self._factor))
        self.spin = QDoubleSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(3)
        self.spin.setSingleStep(step)
        self.spin.setFixedWidth(78)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        self.slider.valueChanged.connect(self._slider_changed)
        self.spin.valueChanged.connect(self._spin_changed)

    def _slider_changed(self, raw: int) -> None:
        value = raw / self._factor
        if abs(self.spin.value() - value) > 1e-12:
            self.spin.blockSignals(True)
            self.spin.setValue(value)
            self.spin.blockSignals(False)
        self.valueChanged.emit(value)

    def _spin_changed(self, value: float) -> None:
        raw = round(value * self._factor)
        if self.slider.value() != raw:
            self.slider.blockSignals(True)
            self.slider.setValue(raw)
            self.slider.blockSignals(False)
        self.valueChanged.emit(value)

    def setValue(self, value: float) -> None:
        self.spin.setValue(float(value))
        self.slider.setValue(round(float(value) * self._factor))

    def value(self) -> float:
        return float(self.spin.value())


class SettingsPanel(QFrame):
    settingsChanged = Signal(dict)
    previewRequested = Signal()
    resetRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SettingsPanel")
        self.setMinimumWidth(330)
        self.setMaximumWidth(430)
        self._loading = False
        self._specs = specs()
        self._widgets: dict[str, QWidget] = {}
        self._sections: dict[str, CollapsibleSection] = {}

        title = QLabel("Restoration settings")
        title.setStyleSheet("font-size: 12pt; font-weight: 600;")
        subtitle = QLabel("Changes are stored per image. Click Preview to render them.")
        subtitle.setObjectName("Muted")
        subtitle.setWordWrap(True)
        self.preview_btn = QPushButton("Preview / Apply")
        self.preview_btn.setObjectName("PreviewButton")
        self.preview_btn.clicked.connect(self.previewRequested)

        self.reset_btn = QPushButton("Reset current image")
        self.reset_btn.clicked.connect(self.resetRequested)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(8, 4, 8, 12)
        inner_layout.setSpacing(2)

        grouped: dict[str, list[SettingSpec]] = defaultdict(list)
        for spec in self._specs:
            grouped[spec.group].append(spec)

        for group in GROUP_ORDER:
            if not grouped[group]:
                continue
            expanded = group in {"General", "Neural correction", "Flat-region cleanup", "Residual profile"}
            section = CollapsibleSection(group, expanded=expanded)
            self._sections[group] = section
            for spec in grouped[group]:
                widget = self._make_widget(spec)
                widget.setToolTip(f"{spec.option}\n{spec.help}".strip())
                label = QLabel(spec.label)
                label.setToolTip(widget.toolTip())
                label.setWordWrap(True)
                section.form.addRow(label, widget)
                self._widgets[spec.dest] = widget
            inner_layout.addWidget(section)
        inner_layout.addStretch(1)
        scroll.setWidget(inner)

        top = QVBoxLayout(self)
        top.setContentsMargins(12, 12, 12, 12)
        top.setSpacing(8)
        top.addWidget(title)
        top.addWidget(subtitle)
        top.addWidget(self.preview_btn)
        top.addWidget(self.reset_btn)
        top.addWidget(scroll, 1)

        self.load_settings(dict(default_settings()))

    def _slider_range(self, spec: SettingSpec) -> tuple[float, float, float]:
        if spec.dest.startswith("orthogonal_profile"):
            return -1.0, 2.0, 0.01
        if spec.dest == "highlight_recovery_strength":
            return 0.0, 1.0, 0.01
        return 0.0, 2.0, 0.01

    def _make_widget(self, spec: SettingSpec) -> QWidget:
        if spec.value_type is bool:
            w = QCheckBox()
            w.stateChanged.connect(self._changed)
            return w
        if spec.choices:
            w = QComboBox()
            for choice in spec.choices:
                w.addItem(str(choice), choice)
            w.currentIndexChanged.connect(self._changed)
            return w
        if spec.dest == "device":
            w = QComboBox()
            for choice in ("auto", "cuda", "cpu"):
                w.addItem(choice, choice)
            w.currentIndexChanged.connect(self._changed)
            return w
        if spec.slider:
            lo, hi, step = self._slider_range(spec)
            w = FloatSlider(lo, hi, step)
            w.valueChanged.connect(self._changed)
            return w
        if spec.value_type is int:
            w = QSpinBox()
            w.setRange(-1000000, 1000000)
            if spec.dest == "processing_size":
                w.setRange(64, 4096)
                w.setSingleStep(8)
            elif "degree" in spec.dest:
                w.setRange(0, 12)
            elif "passes" == spec.dest:
                w.setRange(1, 2)
            else:
                w.setSingleStep(1)
            w.valueChanged.connect(self._changed)
            return w
        if spec.value_type is float:
            w = QDoubleSpinBox()
            w.setRange(-1000000.0, 1000000.0)
            w.setDecimals(6)
            w.setSingleStep(0.01)
            if "period" in spec.dest or "sigma" in spec.dest or "analysis" in spec.dest:
                w.setSingleStep(1.0)
            w.valueChanged.connect(self._changed)
            return w
        w = QLineEdit()
        w.editingFinished.connect(self._changed)
        return w

    def _widget_value(self, spec: SettingSpec, widget: QWidget):
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QComboBox):
            return widget.currentData()
        if isinstance(widget, FloatSlider):
            return widget.value()
        if isinstance(widget, QSpinBox):
            return int(widget.value())
        if isinstance(widget, QDoubleSpinBox):
            return float(widget.value())
        if isinstance(widget, QLineEdit):
            return widget.text().strip()
        return spec.default

    def _set_widget_value(self, spec: SettingSpec, widget: QWidget, value) -> None:
        widget.blockSignals(True)
        try:
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                idx = widget.findData(value)
                if idx < 0:
                    idx = widget.findText(str(value))
                widget.setCurrentIndex(max(0, idx))
            elif isinstance(widget, FloatSlider):
                widget.setValue(float(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
        finally:
            widget.blockSignals(False)

    def load_settings(self, values: dict) -> None:
        self._loading = True
        merged = dict(default_settings())
        merged.update(values or {})
        for spec in self._specs:
            self._set_widget_value(spec, self._widgets[spec.dest], merged[spec.dest])
        self._loading = False

    def current_settings(self) -> dict:
        return {
            spec.dest: self._widget_value(spec, self._widgets[spec.dest])
            for spec in self._specs
        }

    def reset_defaults(self) -> None:
        self.load_settings(dict(default_settings()))
        self.settingsChanged.emit(self.current_settings())

    def set_busy(self, busy: bool) -> None:
        self.preview_btn.setEnabled(not busy)
        self.preview_btn.setText("Processing..." if busy else "Preview / Apply")

    def _changed(self, *args) -> None:
        if self._loading:
            return
        # The profile and surface equalizer are implemented inside the flat-filter stage.
        values = self.current_settings()
        if (values.get("flat_profile") or values.get("flat_surface_equalizer")) and not values.get("flat_filter"):
            self._loading = True
            spec = next(s for s in self._specs if s.dest == "flat_filter")
            self._set_widget_value(spec, self._widgets["flat_filter"], True)
            self._loading = False
            values["flat_filter"] = True
        self.settingsChanged.emit(values)
