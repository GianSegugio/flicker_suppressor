from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QCursor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsDropShadowEffect,
    QLabel,
    QVBoxLayout,
    QWidget,
)


def _early_resource_path(*parts: str) -> Path:
    """Resolve bundled assets without importing the application package yet."""
    if "__compiled__" in globals() or getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
    else:
        root = Path(__file__).resolve().parent
    return root.joinpath(*parts)


class StartupSplash(QWidget):
    """Small transparent startup splash shown before heavy application imports."""

    def __init__(self, logo_path: Path):
        flags = (
            Qt.WindowType.SplashScreen
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        super().__init__(None, flags)
        self.setObjectName("StartupSplash")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        # Keep the splash transparent even after the application-wide dark
        # stylesheet is installed while the main window is still loading.
        self.setStyleSheet(
            "QWidget#StartupSplash { background: transparent; } "
            "QLabel { background: transparent; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 34)
        layout.setSpacing(14)

        logo = QLabel(self)
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = QPixmap(str(logo_path))
        if not pixmap.isNull():
            logo.setPixmap(
                pixmap.scaled(
                    256,
                    256,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        logo.setFixedSize(256, 256)
        layout.addWidget(logo, 0, Qt.AlignmentFlag.AlignHCenter)

        loading = QLabel("loading...", self)
        loading.setObjectName("StartupSplashLoading")
        loading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Keep typography local to this label. The application-wide stylesheet
        # is installed later during startup and can otherwise override QLabel
        # font metrics for the final few frames of the splash.
        loading.setStyleSheet(
            "QLabel#StartupSplashLoading { "
            "color: white; "
            "background: transparent; "
            "font-size: 13pt; "
            "font-weight: 700; "
            "}"
        )
        loading.setMinimumWidth(220)

        shadow = QGraphicsDropShadowEffect(loading)
        shadow.setBlurRadius(18.0)
        shadow.setOffset(0.0, 3.0)
        shadow.setColor(QColor(0, 0, 0, 235))
        loading.setGraphicsEffect(shadow)
        layout.addWidget(loading, 0, Qt.AlignmentFlag.AlignHCenter)

        self.adjustSize()
        self._center_on_current_screen()

    def _center_on_current_screen(self) -> None:
        screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        frame = self.frameGeometry()
        frame.moveCenter(area.center())
        self.move(frame.topLeft())


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Flicker Suppressor")
    app.setOrganizationName("Flicker Suppressor")

    # Show something immediately, before importing MainWindow and its heavy
    # NumPy/PyTorch/CUDA dependency chain.
    logo_path = _early_resource_path("assets", "logo_256.png")
    if logo_path.exists():
        app.setWindowIcon(QIcon(str(logo_path)))

    splash = StartupSplash(logo_path)
    splash.show()
    splash.raise_()
    app.processEvents()

    # Keep all application-package imports below the splash so cold starts on a
    # clean machine provide visual feedback while the heavy runtime initializes.
    from gui.main_window import MainWindow
    from gui.theme import DARK_STYLESHEET

    app.setStyleSheet(DARK_STYLESHEET)
    app.processEvents()

    win = MainWindow()
    win.show()
    app.processEvents()

    splash.close()
    splash.deleteLater()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
