from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QImageReader, QPixmap
from PySide6.QtWidgets import QListWidget, QListWidgetItem


ROLE_ID = Qt.ItemDataRole.UserRole + 1
ROLE_PATH = Qt.ItemDataRole.UserRole + 2


class ThumbnailStrip(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setFlow(QListWidget.Flow.LeftToRight)
        self.setWrapping(False)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Static)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setIconSize(QSize(112, 76))
        self.setGridSize(QSize(154, 116))
        self.setHorizontalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.setSpacing(3)

    def make_icon(self, path: Path) -> QIcon:
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid():
            size.scale(self.iconSize(), Qt.AspectRatioMode.KeepAspectRatio)
            reader.setScaledSize(size)
        image = reader.read()
        if image.isNull():
            return QIcon()
        return QIcon(QPixmap.fromImage(image))

    def add_image(self, job_id: str, path: Path, status: str = "Original") -> QListWidgetItem:
        item = QListWidgetItem(self.make_icon(path), f"{path.name}\n{status}")
        item.setData(ROLE_ID, job_id)
        item.setData(ROLE_PATH, str(path))
        item.setToolTip(str(path))
        self.addItem(item)
        return item

    def find_by_id(self, job_id: str) -> QListWidgetItem | None:
        for i in range(self.count()):
            item = self.item(i)
            if item.data(ROLE_ID) == job_id:
                return item
        return None

    def update_status(self, job_id: str, status: str) -> None:
        item = self.find_by_id(job_id)
        if item is None:
            return
        path = Path(item.data(ROLE_PATH))
        item.setText(f"{path.name}\n{status}")

    def selected_ids(self) -> list[str]:
        return [item.data(ROLE_ID) for item in self.selectedItems()]
