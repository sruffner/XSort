from typing import Optional, List

from PySide6.QtCore import QSize, QObject, QSettings
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QWidget, QDockWidget

from xsort.data.analyzer import Analyzer
from xsort.data.neuron import DataType


class BaseView(QObject):
    """
    A base class defining functionality common to all XSort view widgets as well as Qt-style signals that can be
    used for inter-view communications
    """

    def __init__(self, title: str, background: Optional[QColor], data_manager: Analyzer, settings: QSettings):
        """
        Create an empty view with the specified title.

        :param title: The view's title.
        :param background: An alternative background color (intended for debug use when testing view layout).
        :param data_manager: The source for all raw recorded data and analysis results presented in any XSort view.
        :param settings: The user's XSort application settings.
        """
        super().__init__(parent=None)
        self.view_container = QWidget()
        """ The widget that contains this view. """
        self._title = title
        """ The view's title. """
        self.data_manager = data_manager
        """ Each view will query this object for raw recorded data and analyis results. """
        self.settings = settings
        """ 
        View-specific and application-wide user preferences. Each view is responsible for maintaining its own
        settings.
        """

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
    def is_parent_dock_hidden(self) -> bool:
        """ True if the docking widget containing this view is currently hidden. """
        widget = self.view_container.parentWidget()
        if isinstance(widget, QDockWidget):
            return widget.isHidden()
        return True

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

    def on_neuron_metrics_updated(self, uid: str) -> None:
        """
        Refresh view contents after the metrics for a neural unit have been updated or retrieved from the
        working directory contents.  Default implementation takes no action.

        :param uid: UID identifying the neural unit for which updated metrics are available.
        """
        pass

    def on_channel_traces_updated(self) -> None:
        """
        Refresh view contents after the channel trace segments have been retrieved for the current set of displayable
        analog channels. The trace segments are retrieved by a background task each time the user changes the segment
        start time, or whenever the displayable channel set changes because of a change in the current unit focus list.
        Default implementation takes no action.
        """
        pass

    def on_focus_neurons_changed(self, channels_changed: bool) -> None:
        """
        Refresh view contents after a change in the list of neurons selected for display/comparison purposes. Default
        implementation takes no action.

        :param channels_changed: True if the set of displayable analog channels has changed as a result of the change
            in the unit focus list. This can only happen if the number of recorded analog channels > 16, in which case
            the displayable set is the 16 channels on which the primary unit's spike templates were computed.
        """
        pass

    def on_neuron_labels_updated(self, uids: List[str]) -> None:
        """
        Refresh view contents after one or more units in the neural unit list is relabeled. Default implementation
        takes no action.
        :param uids: List of UIDs identifying the neural units that were re-labeled.
        """
        pass

    def on_focus_neurons_stats_updated(self, data_type: DataType, uid: str) -> None:
        """
        Refresh view contents after some statistic is computed for a specified neuron within the list of neurons
        currently seelected for display/comparison purposes -- aka the "focus list". Default implementation takes no
        action. Note that the statistic is cached in the :class:`Neuron` instance.

        :param data_type: Indicates the type of statistic that was computed or recomputed.
        :param uid: UID identifying the neural unit for which a computed statistic is available.
        """
        pass

    def on_channel_trace_segment_start_changed(self) -> None:
        """
        Refresh view contents after a change in the elapsed starting time (relative to the beginning of the
        electrophysiological recording) for all analog channel 1-second trace segments. Default implementation takes no
        action.
        """
        pass

    def save_settings(self) -> None:
        """
        Save any view-specific user preferences that should be restored the next time XSort is launched. The application
        settings object is an instance member of the view (a reference shared by all views).

        This method is called just prior to application exit, but before the GUI is destroyed. Default implementation
        does nothing.
        """
        pass

    def restore_settings(self) -> None:
        """
        Load any view-specific user preferences and refresh the view state accordingly. The application settings object
        is an instance member of the view (a reference shared by all views).
        This method is called during application startup, after the main window and all views have been realized but not
        shown. Default implementation does nothing.
        """
        pass
