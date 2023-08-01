from typing import Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout


class BaseView(QWidget):
    """
    TODO: UNDER DEV
    """

    def __init__(self, uid: int, title: str, background: Optional[QColor] = None) -> None:
        """
        Create an empty view with the specified title.

        :param uid: The view's identifier.
        :param title: The view's title.
        """
        super(BaseView, self).__init__()

        self._uid = uid
        """ The view's identifier. """
        self._title = title
        """ The view's title. """

        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, background)
        self.setPalette(palette)
        self.setContentsMargins(0, 0, 0, 0)
        self.setMinimumSize(QSize(100, 100))
        self.setWindowTitle(title)

        label = QLabel(f"This is the '{title}' view.")
        label.setMargin(1)

        layout = QVBoxLayout()
        layout.addWidget(label, alignment=Qt.AlignVCenter | Qt.AlignmentFlag.AlignHCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    @property
    def uid(self) -> int:
        """ The view's ID. """
        return self._uid

    @property
    def title(self) -> str:
        """ The view's title. """
        return self._title
