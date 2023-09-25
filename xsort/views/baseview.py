from typing import Optional

from PySide6.QtCore import QSize, QObject, Signal
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QWidget

from xsort.data.analyzer import Analyzer


class BaseView(QObject):
    """
    A base class defining functionality common to all XSort view widgets as well as Qt-style signals that can be
    used for inter-view communications
    """

    selected_neuron_changed: Signal = Signal(str)
    """ 
    Signals a change in the neural unit selected within the NeuronView, which is considered the unit with the display 
    focus across all XSort views. Arg (str): The label uniquely identifying the unit. If empty, no unit is selected.
    """

    def __init__(self, title: str, background: Optional[QColor], data_manager: Analyzer):
        """
        Create an empty view with the specified title.

        :param title: The view's title.
        :param background: An alternative background color (intended for debug use when testing view layout).
        :param data_manager: The source for all raw recorded data and analysis results presented in any XSort view.
        """
        super().__init__(parent=None)
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

    def on_working_directory_changed(self) -> None:
        """
        Refresh view contents after the current working directory has changed. Default implementation takes no action,
        but all views should re-initialize their contents appropriately.
        """
        pass

    def on_neuron_metrics_updated(self, unit_label: str) -> None:
        """
        Refresh view contents after the metrics for a neural unit have been updated or retrieved from the
        working directory contents.  Default implementation takes no action.

        :param unit_label: The unique label identifying the neural unit for which updated metrics are available.
        """
        pass

    def on_channel_trace_segment_updated(self, idx: int) -> None:
        """
        Refresh view contents after the data for an analog channel trace segment has been retrieved from the working
        directory contents. Degault implementation takes no action.

        :param idx: Index of the analog channel for which data is available.
        """
        pass

    def on_focus_neuron_changed(self, unit_label: str) -> None:
        """
        Refresh view contents after the user changes the display focus to a different neural unit. Default
        implementation takes no action.

        :param unit_label: The unique label identifying the neural unit receiving the focus. Will be an empty string if
            no unit has the display focus.
        """
        pass
