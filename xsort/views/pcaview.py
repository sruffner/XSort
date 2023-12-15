from typing import List, Optional

from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QVBoxLayout, QComboBox, QLabel, QHBoxLayout, QPushButton
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
    a 2-ms clip "centered" on the spike occurrence time. For each spike, one such clip is extracted from the M Omniplex
    analog channels recorded, and the clips are concatenated end-to-end, yielding a P=2M "multi-clip" for each spike.
    The goal of PCA is to reduce the dimension P to 2 -- so that each spike is represented by a single point in 2D
    space.
        If there are 3 units in the focus list, there are a total of (N1 + N2 + N3) multi-clips of length P samples. The
    first step in PCA is to form the 3xP matrix containing the first 2ms clip of each unit's spike waveform as computed
    on each analog channel, then find the eigenvalues/vectors for the matrix. The eigenvectors associated with the two
    highest eigenvalues represent the two directions in P-space along which there's the most variance in the dataset --
    these are the two principal components which preserve the most information in the original data. A Px2 matrix is
    formed from these two eigenvectors. (Initially, we selected a random sampling of 1000 multi-clips across the 3 units
    -- in proportion to each unit's contribution to the total number of clips, resulting in a 1000xP matrix. But the
    results were unsatisfactory because many of the clips were mostly noise.) In the second, longer step, all N
    multi-clips for each units are concatenated to form a NxP matrix, which is multiplied by the Px2 principal component
    matrix (this is done in smaller chunks to conserve memory) to generate the projection of that unit's spikes onto the
    2D plane defined by the two principal components.
        NOTE that it should be clear that, whenver the composition of the focus list changes, the principal component
    analysis must be redone.

        As with many of the views rendering statistics plots for neurons in the display list, a plot item is pre-created
    to render the PCA projection (as a scatter plot) for the unit in each possible "slot" in the display list.
    Initially, these plot items contain empty data arrays. The view is "refreshed" by updating these data arrays
    whenever a PCA projection array becomes available (the statistic is cached within the :class:`Neuron` instance), or
    whenever the focus list changes -- in which case any previously plotted projections are reset to empty arrays.
        Initial testing has shown that, when there are a great many points to draw, it takes a noticeable amount of
    time to render the scatter plots. And with a great many points and overlapping projections, one unit's projection
    can obscure another's. At the bottom of the view are two controls to help address these issues:
        - A "Toggle Z Order" button changes the Z-order of the displayed scatter plots.
        - A "Downsample" combo box lets the user choose the downsampling factor for the rendered plots. A value of 1
          disables downsampling.
    """

    _SYMBOL_SIZE: int = 4
    """ Fixed size of all scatter plot symbols in this view, in pixels. """
    _DOWNSAMPLE_CHOICES: List[int] = [200, 100, 50, 20, 10, 5, 1]
    """ Choice list for downsampling factor N: every Nth point is plotted (1=all points plotted). """
    _DEF_DOWNSAMPLE: int = 50
    """ Default downsampling factor. """

    def __init__(self, data_manager: Analyzer) -> None:
        super().__init__('PCA', None, data_manager)
        self._plot_widget = pg.PlotWidget()
        """ PCA projections for all neurons in the current display list are rendered in this widget. """
        self._plot_item: pg.PlotItem = self._plot_widget.getPlotItem()
        """ The graphics item that manages plotting of the PCA projections. """
        self._pca_data: List[pg.PlotDataItem] = list()
        """ 
        The plot data item at pos N renders the PCA projection for the N-th neuron in the display focus list. The 
        trace color matches the color assigned to that position in the display list. Each of these plot data items
        represent scatter plots.
        """
        self._downsample_combo = QComboBox()
        """ 
        Combo box selects the downsample factor (render every Nth point) for the view, since the PCA projections
        will include one point per spike in a unit's spike train -- potentially 100K+ points! 
        """
        self._toggle_z_btn = QPushButton("Toggle Z Order")
        """ Clicking this pushbutton toggles the display order of the PCA projections drawn in this view. """

        # some configuration. We hide both axes because the units of the two principal components are meaningless.
        self._plot_item.setMenuEnabled(False)
        self._plot_item.getViewBox().setMouseEnabled(x=False, y=False)
        self._plot_item.hideButtons()
        self._plot_item.hideAxis('left')
        self._plot_item.hideAxis('bottom')

        # pre-create the plot data items for the maximum number of units in the display focus list
        for i in range(Analyzer.MAX_NUM_FOCUS_NEURONS):
            color = QColor.fromString(Analyzer.FOCUS_NEURON_COLORS[i])
            color.setAlpha(128)
            self._pca_data.append(
                self._plot_item.plot(
                    np.empty((0, 2)), pen=None, symbol='o',  symbolPen=pg.mkPen(None),
                    symbolSize=PCAView._SYMBOL_SIZE, symbolBrush=color)
            )
            self._pca_data[i].setDownsampling(ds=PCAView._DEF_DOWNSAMPLE, auto=False, method='subsample')
            self._pca_data[i].setZValue(i)   # from QGraphicsItem base class

        # set up the combo box that selects downsample factor (note we have to convert to/from str)
        self._downsample_combo.addItems([str(k) for k in PCAView._DOWNSAMPLE_CHOICES])
        self._downsample_combo.setCurrentText(str(PCAView._DEF_DOWNSAMPLE))
        self._downsample_combo.currentTextChanged.connect(self._on_downsample_factor_changed)

        label = QLabel("Downsample (1=disabled):")
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._toggle_z_btn.clicked.connect(self._on_toggle_z_btn_clicked)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self._plot_widget)
        control_line = QHBoxLayout()
        control_line.addWidget(self._toggle_z_btn)
        control_line.addStretch(1)
        control_line.addWidget(label)
        control_line.addWidget(self._downsample_combo)
        main_layout.addLayout(control_line)

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
                if displayed[k].uid == unit_label:
                    self._pca_data[k].setData(displayed[k].cached_pca_projection())
                    break

    @Slot(str)
    def _on_downsample_factor_changed(self, _: str) -> None:
        """
         Handler called when the user changes the downsampling factor in the combo box.
        :param _: (Unused) The text selected in the combo box.
        """
        # delay update by 100ms so the combo box has time to close -- because it can take a while to draw
        # scatter plots with 100K+ points!
        QTimer.singleShot(100, self._update_downsample_factor)

    def _update_downsample_factor(self) -> None:
        """
        Get the current downsample factor from the corresponding combo box and update each of the plot data items
        in this view accordingly.
        """
        ds = int(self._downsample_combo.currentText())
        for i in range(Analyzer.MAX_NUM_FOCUS_NEURONS):
            self._pca_data[i].setDownsampling(ds, False, 'subsample')

    @Slot(bool)
    def _on_toggle_z_btn_clicked(self, _: bool) -> None:
        """
        Handler called when the user clicks the "Toggle Z Order" push button.
        :param _: (Unused) The checked state of the button, which is not applicable here.
        """
        if len(self.data_manager.neurons_with_display_focus) > 1:
            QTimer.singleShot(100, self._toggle_z_order)

    def _toggle_z_order(self) -> None:
        """
        Change the display order -- aka "Z order" -- of the plot data items that currently render PCA projections for
        neural units in the current display/focus list. If the focus list is empty, there's nothing to do!
        """
        num_displayed = len(self.data_manager.neurons_with_display_focus)
        if num_displayed == 2:
            z = self._pca_data[0].zValue()
            self._pca_data[0].setZValue(self._pca_data[1].zValue())
            self._pca_data[1].setZValue(z)
        elif num_displayed == 3:
            z = self._pca_data[0].zValue()
            self._pca_data[0].setZValue(self._pca_data[1].zValue())
            self._pca_data[1].setZValue(self._pca_data[2].zValue())
            self._pca_data[2].setZValue(z)
