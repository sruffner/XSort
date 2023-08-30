from abc import ABC, abstractmethod
from typing import Optional

from PySide6.QtCore import QSize
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QWidget

from xsort.data.analyzer import Analyzer


class BaseView(ABC):
    """
    An abstract base class defining functionality common to all XSort view widgets.
    """
    def __init__(self, title: str, background: Optional[QColor], data_manager: Analyzer):
        """
        Create an empty view with the specified title.

        :param title: The view's title.
        :param background: An alternative background color (intended for debug use when testing view layout).
        :param data_manager: The source for all raw recorded data and analysis results presented in any XSort view.
        """
        self.view_container = QWidget()
        """ The widget that contains this view. """
        self._title = title
        """ The view's title. """
        self.data_manager = data_manager
        """ Each view will query this object for raw recorded data and analyis results. """

        if isinstance(background, QColor):
            self.view_container.setAutoFillBackground(True)
            palette = self.view_container.palette()
            palette.setColor(QPalette.Window, background)
            self.view_container.setPalette(palette)
        self.view_container.setContentsMargins(0, 0, 0, 0)
        self.view_container.setMinimumSize(QSize(100, 100))
        self.view_container.setWindowTitle(title)

    @property
    def title(self) -> str:
        """ The view's title. """
        return self._title

    @property
    def container(self) -> QWidget:
        """ The widget container for this view. """
        return self.view_container

    @abstractmethod
    def on_working_directory_changed(self) -> None:
        """ Refresh view contents after the current working directory has changed. """
        ...
