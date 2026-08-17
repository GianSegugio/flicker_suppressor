DARK_STYLESHEET = r'''
QMainWindow, QWidget { background:#1f2023; color:#e7e8ea; font-family:"Segoe UI"; font-size:10pt; }
QMenuBar { background:#27282b; padding:3px; }
QMenuBar::item { padding:5px 9px; }
QMenuBar::item:selected { background:#3a3c40; border-radius:3px; }
QMenu { background:#292a2e; border:1px solid #45474d; padding:5px; }
QMenu::item { padding:6px 30px 6px 22px; }
QMenu::item:selected { background:#3a6ea5; }
QMenu::item:disabled { color:#6f737a; }
QFrame#SettingsPanel { background:#252629; border-left:1px solid #34363a; }
QFrame#CanvasToolbar, QFrame#BottomBar { background:#222326; border-top:1px solid #34363a; }
QLabel#Muted { color:#9da1a8; }
QPushButton, QToolButton { background:#303237; border:1px solid #44474d; border-radius:4px; padding:5px 9px; }
QPushButton:hover, QToolButton:hover { background:#393c42; }
QPushButton:pressed, QToolButton:pressed { background:#292b2f; }
QToolButton#MaskToolButton:checked { background:#3c8df6; border-color:#3c8df6; color:white; }
QToolButton#MaskToolButton:checked:hover { background:#529bfa; border-color:#529bfa; }
QPushButton:disabled, QToolButton:disabled { color:#70747b; background:#27282b; border-color:#34363a; }
QPushButton#PrimaryButton, QPushButton#ExportMainButton, QToolButton#SplitDropButton { background:#3c8df6; border-color:#3c8df6; color:white; font-weight:600; }
QPushButton#PrimaryButton:hover, QPushButton#ExportMainButton:hover, QToolButton#SplitDropButton:hover { background:#529bfa; }
QPushButton#PrimaryButton:disabled, QPushButton#ExportMainButton:disabled, QToolButton#SplitDropButton:disabled { background:#30445d; border-color:#30445d; color:#78889b; }
QToolButton#SplitDropButton { border-top-left-radius:0; border-bottom-left-radius:0; border-left:1px solid #6aa9fa; padding:0; }
QToolButton#SplitDropButton::menu-indicator { image:none; width:0px; height:0px; }
QPushButton#ExportMainButton { border-top-right-radius:0; border-bottom-right-radius:0; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background:#18191b; border:1px solid #45474d; border-radius:3px; padding:4px 6px; min-height:22px; }
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled { color:#6f737a; background:#202124; border-color:#34363a; }
QFrame#ModernSpinField {
    background:#18191b;
    border:1px solid #45474d;
    border-radius:3px;
}
QFrame#ModernSpinField:disabled { background:#202124; border-color:#34363a; }
QSpinBox#ModernSpinEditor, QDoubleSpinBox#ModernSpinEditor {
    background:transparent;
    border:none;
    border-radius:0;
    padding:3px 7px;
    min-height:0;
}
QSpinBox#ModernSpinEditor:disabled, QDoubleSpinBox#ModernSpinEditor:disabled {
    color:#6f737a;
    background:transparent;
    border:none;
}
QFrame#NumericStepper { background:transparent; border:none; }
QToolButton#NumericStepUp, QToolButton#NumericStepDown {
    background:#25272b;
    border:1px solid #646973;
    border-radius:0;
    padding:0;
    margin:0;
}
QToolButton#NumericStepUp { border-top-right-radius:2px; }
QToolButton#NumericStepDown {
    border-top:0px;
    border-bottom-right-radius:2px;
}
QToolButton#NumericStepUp:hover, QToolButton#NumericStepDown:hover {
    background:#34373c;
    border-color:#7a808a;
}
QToolButton#NumericStepUp:pressed, QToolButton#NumericStepDown:pressed {
    background:#1d1f22;
    border-color:#707680;
}
QToolButton#NumericStepUp:disabled, QToolButton#NumericStepDown:disabled {
    background:#202124;
    border-color:#41444a;
}

QFrame#ModernComboField {
    background:#18191b;
    border:1px solid #45474d;
    border-radius:3px;
}
QFrame#ModernComboField:disabled { background:#202124; border-color:#34363a; }
QComboBox#ModernComboEditor {
    background:transparent;
    border:none;
    border-radius:0;
    padding:3px 7px;
    min-height:0;
}
QComboBox#ModernComboEditor:disabled { color:#6f737a; background:transparent; border:none; }
QComboBox#ModernComboEditor::drop-down { width:0px; border:none; background:transparent; }
QComboBox#ModernComboEditor::down-arrow { image:none; width:0px; height:0px; }
QToolButton#ModernComboDrop {
    background:#25272b;
    border:none;
    border-left:1px solid #646973;
    border-radius:0;
    border-top-right-radius:2px;
    border-bottom-right-radius:2px;
    padding:0;
    margin:0;
}
QToolButton#ModernComboDrop:hover { background:#34373c; border-left-color:#7a808a; }
QToolButton#ModernComboDrop:pressed { background:#1d1f22; border-left-color:#707680; }
QToolButton#ModernComboDrop:disabled { background:#202124; border-left-color:#41444a; }

QCheckBox:disabled { color:#6f737a; }
QWidget#ResetControlRow, QWidget#SliderRow { background:transparent; border:none; }
QToolButton#SettingResetButton {
    background:#292b2f;
    border:1px solid #45484e;
    border-radius:4px;
    padding:0;
}
QToolButton#SettingResetButton:hover { background:#35383d; border-color:#656a73; }
QToolButton#SettingResetButton:pressed { background:#202226; border-color:#565b63; }
QToolButton#SettingResetButton:disabled { background:#242529; border-color:#34363a; }
QSlider, QSlider#FlatSlider { background:transparent; border:none; padding:0; margin:0; }
QSlider::groove:horizontal { height:4px; background:#45474d; border-radius:2px; }
QSlider::handle:horizontal { width:14px; margin:-5px 0; background:#d9dde3; border-radius:7px; }
QSlider::sub-page:horizontal { background:#4e91df; }
/* Disabled state must be attached to the subcontrol, not before it. */
QSlider#FlatSlider::groove:horizontal:disabled { background:#3b3d42; }
QSlider#FlatSlider::sub-page:horizontal:disabled { background:#5a5d62; }
QSlider#FlatSlider::add-page:horizontal:disabled { background:#3b3d42; }
QSlider#FlatSlider::handle:horizontal:disabled { background:#73767b; }
QScrollArea { border:none; }
/* Compact modern scrollbar used by the right-hand settings panel. */
QScrollArea#SettingsScrollArea QScrollBar:vertical {
    background:transparent;
    width:10px;
    margin:3px 1px 3px 1px;
}
QScrollArea#SettingsScrollArea QScrollBar::handle:vertical {
    background:#565960;
    min-height:34px;
    border:none;
    border-radius:4px;
    margin:0 2px;
}
QScrollArea#SettingsScrollArea QScrollBar::handle:vertical:hover { background:#70747b; }
QScrollArea#SettingsScrollArea QScrollBar::handle:vertical:pressed { background:#858a92; }
QScrollArea#SettingsScrollArea QScrollBar::add-line:vertical,
QScrollArea#SettingsScrollArea QScrollBar::sub-line:vertical {
    height:0px;
    background:transparent;
    border:none;
}
QScrollArea#SettingsScrollArea QScrollBar::add-page:vertical,
QScrollArea#SettingsScrollArea QScrollBar::sub-page:vertical { background:transparent; }
/* Match the clip strip's horizontal scrollbar to the settings scrollbar. */
QAbstractScrollArea#ImageStrip QScrollBar:horizontal {
    background:transparent;
    height:10px;
    margin:1px 3px 1px 3px;
}
QAbstractScrollArea#ImageStrip QScrollBar::handle:horizontal {
    background:#565960;
    min-width:34px;
    border:none;
    border-radius:4px;
    margin:2px 0;
}
QAbstractScrollArea#ImageStrip QScrollBar::handle:horizontal:hover { background:#70747b; }
QAbstractScrollArea#ImageStrip QScrollBar::handle:horizontal:pressed { background:#858a92; }
QAbstractScrollArea#ImageStrip QScrollBar::add-line:horizontal,
QAbstractScrollArea#ImageStrip QScrollBar::sub-line:horizontal {
    width:0px;
    background:transparent;
    border:none;
}
QAbstractScrollArea#ImageStrip QScrollBar::left-arrow:horizontal,
QAbstractScrollArea#ImageStrip QScrollBar::right-arrow:horizontal { width:0px; height:0px; }
QAbstractScrollArea#ImageStrip QScrollBar::add-page:horizontal,
QAbstractScrollArea#ImageStrip QScrollBar::sub-page:horizontal { background:transparent; }
QListWidget { background:#202124; border:none; outline:none; }
QAbstractScrollArea#ImageStrip { background:#252629; border:1px solid #34363a; outline:none; }
QListWidget::item { border:1px solid transparent; padding:4px; }
QListWidget::item:selected { background:#2d3744; border:1px solid #4d96e8; }
QGroupBox { font-weight:600; border:1px solid #34363a; border-radius:4px; margin-top:12px; padding-top:8px; }
QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 4px; }
QToolTip { background:#33353a; color:white; border:1px solid #575a61; padding:4px; }
'''



def build_stylesheet(assets_dir=None) -> str:
    """Return the application stylesheet.

    Numeric steppers are real Qt child widgets now, not native QSpinBox
    subcontrols, so no platform-dependent arrow assets are required.
    """
    return DARK_STYLESHEET
