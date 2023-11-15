from typing import List, Optional

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QVBoxLayout
import pyqtgraph as pg
import numpy as np

from xsort.data.analyzer import Analyzer
from xsort.data.neuron import Neuron, DataType
from xsort.views.baseview import BaseView


class PCAView(BaseView):
    """
    This view renders the results of principal component analysis (PCA) on the spike clips of the neural units in the
    current display/focus list. The purpose of the analysis is to "map" each spike in each unit's spike train to a
    point in a 2D space, offering a "scatter plot" to help assess whether the units are truly distinct from each other.
    This view is responsible for rendering the scatter plots.
        PCA is a time-consuming operation that is always performed on a background thread. It can take many seconds to
    complete, especially if the unit spike trains are very long. In the analysis approach, each spike is represented by
    a 2-ms "centered" on the spike occurrence time. For each spike, one such clip is extracted from the M Omniplex
    analog channels recorded, and the clips are concatenated end-to-end, yielding a P=2M "multi-clip" for each spike.
    The goal of PCA is to reduce the dimension P to 2 -- so that each spike is represented by a single point in 2D
    space.
        If there are 3 units in the focus list, there are a total of (N1 + N2 + N3) multi-clips of length P samples. The
    first step in PCA is to select a random sampling of 1000 multi-clips across the 3 units (in proportion to each
    unit's contribution to the total number of clips), then find the eigenvalues/vectors for the resulting 1000xP
    matrix. The eigenvectors associated with the two highest eigenvalues represent the two directions in P-space along
    which there's the most variance in the dataset -- these are the two principal components which preserve the most
    information in the original data. A Px2 matrix is formed from these two eigenvectors. In the second, longer step,
    all N multi-clips for each units are concatenated to form a NxP matrix, which is multiplied by the Px2 principal
    component matrix (this is done in smaller chunks to conserve memory) to generate the projection of that unit's
    spikes onto the 2D plane defined by the two principal components.
        NOTE that it should be clear that, whenver the composition of the focus list changes, the principal component
    analysis must be redone.
        As with many of the views rendering statistics plots for neurons in the display list, a plot item is pre-created
    to render the PCA projection (as a scatter plot) for the unit in each possible "slot" in the display list.
    Initially, these plot items contain empty data arrays. The view is "refreshed" by updating these data arrays
    whenever a PCA projection array becomes available (the statistic is cached within the :class:`Neuron` instance), or
    whenever the focus list changes -- in which case any previously plotted projections are reset to empty arrays.
    """

    _SYMBOL_SIZE: int = 4
    """ Fixed size of all scatter plot symbols in this view, in pixels. """

    def __init__(self, data_manager: Analyzer) -> None:
        super().__init__('PCA', None, data_manager)
        self._pca_plot_widget = pg.PlotWidget()
        """ PCA projections for all neurons in the current display list are rendered in this widget. """
        self._pca_plot_item: pg.PlotItem = self._pca_plot_widget.getPlotItem()
        """ The graphics item that manages plotting of the PCA projections. """
        self._pca_data: List[pg.PlotDataItem] = list()
        """ 
        The plot data item at pos N renders the PCA projection for the N-th neuron in the display focus list. The 
        trace color matches the color assigned to that position in the display list. Each of these plot data items
        represent scatter plots.
        """

        # some configuration
        self._pca_plot_item.setMenuEnabled(False)

        # pre-create the plot data items for the maximum number of units in the display focus list
        for i in range(Analyzer.MAX_NUM_FOCUS_NEURONS):
            color = QColor.fromString(Analyzer.FOCUS_NEURON_COLORS[i])
            color.setAlpha(128)
            self._pca_data.append(
                self._pca_plot_item.plot(
                    np.empty((0, 2)), pen=None, symbol='o',  symbolPen=pg.mkPen(None),
                    symbolSize=PCAView._SYMBOL_SIZE, symbolBrush=color, clipToView=True,
                    downsample=10, downsampleMethod='subsample')
            )

        main_layout = QVBoxLayout()
        main_layout.addWidget(self._pca_plot_widget)

        self.view_container.setLayout(main_layout)

    def _refresh(self) -> None:
        """
        Refresh the PCA projections rendered as scatter plots in this view in response to a change in the neuron display
        list, or when notified that the PCA projection for a particular unit in the display list has been computed and
        is ready for plotting.
            This method simply updates the data item corresponding to the PCA projection for each "slot" in the neuron
        display list. When a display slot is unused, the corresponding data item contains an empty array and thus
        renders nothing.
            By design, the PCA projection is cached in the :class:`Neuron` instance. All previous projections are
        cleared whenever the composition of the display list changes, and the new PCA projections are computed on a
        background thread; if not yet available, the corresponding plot data item is set to an empty data array and
        therefore renders nothing.
        """
        displayed = self.data_manager.neurons_with_display_focus

        for k in range(Analyzer.MAX_NUM_FOCUS_NEURONS):
            unit: Optional[Neuron] = displayed[k] if k < len(displayed) else None
            prj: np.ndarray = np.empty((0, 2)) if unit is None else unit.cached_pca_projection()
            self._pca_data[k].setData(prj)

    def on_working_directory_changed(self) -> None:
        self._refresh()

    def on_focus_neurons_changed(self) -> None:
        self._refresh()

    def on_focus_neurons_stats_updated(self, data_type: DataType, unit_label: str) -> None:
        if data_type == DataType.PCA:
            displayed = self.data_manager.neurons_with_display_focus
            for k in range(len(displayed)):
                if displayed[k].label == unit_label:
                    self._pca_data[k].setData(displayed[k].cached_pca_projection())
                    break
