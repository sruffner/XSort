from typing import Optional

from PySide6.QtCore import QSize, QObject, QSettings
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QWidget

from xsort.data.analyzer import Analyzer


class BaseView(QObject):
    """
    A base class defining functionality common to all XSort view widgets as well as Qt-style signals that can be
    used for inter-view communications
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

    def on_focus_neurons_changed(self) -> None:
        """
        Refresh view contents after a change in the list of neurons selected for display/comparison purposes. Default
        implementation takes no action.
        """
        pass

    def on_focus_neurons_stats_updated(self) -> None:
        """
        Refresh view contents after some statistics are updated/recomputed for any or all of the neurons currently
        selected for display/comparison purposes. Default implementation takes no action.
        """
        pass

    def on_channel_trace_segment_start_changed(self) -> None:
        """
        Refresh view contents after a change in the elapsed starting time (relative to the beginning of the
        electrophysiological recording) for all analog channel 1-second trace segments. Default implementation takes no
        action.
        """
        pass

    def save_settings(self, settings: QSettings) -> None:
        """
        Save any view-specific user preferences that should be restored the next time XSort is launched. This method
        is called just prior to application exit, but before the GUI is destroye. Default implementation does nothing.
        :param settings: The application settings object.
        """
        pass

    def restore_settings(self, settings: QSettings) -> None:
        """
        Load any view-specific user preferences and refresh the view state accordingly. This method is called during
        application startup, after the main window and all views have been realized but not shown. Default
        implementation does nothing.
        :param settings: The application settings object.
        """
        pass
