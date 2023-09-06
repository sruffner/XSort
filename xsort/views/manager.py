from pathlib import Path
from typing import List

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QMainWindow

from xsort.data.analyzer import Analyzer
from xsort.constants import APP_NAME
from xsort.views.baseview import BaseView
from xsort.views.channelview import ChannelView
from xsort.views.neuronview import NeuronView
from xsort.views.pcaview import PCAView
from xsort.views.similarityview import SimilarityView
from xsort.views.statisticsview import StatisticsView
from xsort.views.templateview import TemplateView
from xsort.views.umapview import UMAPView


class ViewManager:
    def __init__(self, main_window: QMainWindow):
        self._main_window = main_window
        """ Reference to the main application window -- to update standard UIlike status bar and window title."""
        self.data_analyzer = Analyzer()
        """
        The master data model. It encapsulates the notion of XSort's 'current working directory, mediates access to 
        data stored in the various files within that directory, performs analyses triggered by view actions, and so on.
        """
        self._neuron_view = NeuronView(self.data_analyzer)
        self._similarity_view = SimilarityView(self.data_analyzer)
        self._templates_view = TemplateView(self.data_analyzer)
        self._statistics_view = StatisticsView(self.data_analyzer)
        self._pca_view = PCAView(self.data_analyzer)
        self._channels_view = ChannelView(self.data_analyzer)
        self._umap_view = UMAPView(self.data_analyzer)

        self._all_views = [self._neuron_view, self._similarity_view, self._templates_view, self._statistics_view,
                           self._pca_view, self._channels_view, self._umap_view]
        """ List of all managed views. """

        # connect to Analyzer signals
        self.data_analyzer.working_directory_changed.connect(self.on_working_directory_changed)
        self.data_analyzer.background_task_updated.connect(self.on_background_task_updated)

    @property
    def central_view(self) -> BaseView:
        """ The XSort view which should be installed as the central widget in the main application window. """
        return self._neuron_view

    @property
    def dockable_views(self) -> List[BaseView]:
        """
        List of dockable XSort views -- ie, all views other than the central view.
        """
        return [self._similarity_view, self._templates_view, self._statistics_view, self._pca_view, self._channels_view,
                self._umap_view]

    @property
    def main_window_title(self) -> str:
        """
        String to be displayed in the title bar of the main application window. This reflects the current working
        directory, if defined.
        """
        if isinstance(self.data_analyzer.working_directory, Path):
            return f"{APP_NAME} ({str(self.data_analyzer.working_directory)})"
        else:
            return APP_NAME

    @Slot()
    def on_working_directory_changed(self) -> None:
        """ Handler updates all views when the current working directory has changed. """
        for v in self._all_views:
            v.on_working_directory_changed()

    @Slot()
    def on_background_task_updated(self, msg: str) -> None:
        self._main_window.statusBar().showMessage(msg if isinstance(msg, str) and (len(msg) > 0) else "Ready")
