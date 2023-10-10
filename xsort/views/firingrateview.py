from typing import List, Optional

from PySide6.QtCore import Qt, Slot, QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QVBoxLayout, QComboBox, QHBoxLayout
import pyqtgraph as pg
import numpy as np

from xsort.data.analyzer import Analyzer
from xsort.data.neuron import Neuron
from xsort.views.baseview import BaseView


class FiringRateView(BaseView):
    """
    This simple view plots the variation in firing rate of each unit in the current display/focus list over the
    course of the electrophysiological recording.
        As with many of the views rendering plots for neurons in the display list, a plot item is pre-created to render
    the firing rate histogram for the unit in each possible "slot" in the display list. Initially, these plot items
    contain empty (X,Y) data arrays. The view is "refreshed" by updating those data arrays IAW the current state --
    including the selected bin size and what neurons currently occupy the display list.
        The firing rate histogram is computed on the fly via :method:`firing_rate_histogram()` in :class:`Neuron`. The
    computation is fast, so it does not impact GUI responsiveness.
    """

    _BIN_SIZE_CHOICES: List[int] = [20, 30, 60, 90, 120, 300]
    """ Choice list for firing rate histogram bin size, in seconds. """
    _DEF_BIN_SIZE: int = 60
    """ Default bin size for firing rate histograms in seconds. """
    _BIN_SIZE_STEP: int = 20
    """ Bin step size in seconds. """
    _PEN_WIDTH: int = 3
    """ Width of pen used to draw the firing rate histograms. """

    def __init__(self, data_manager: Analyzer) -> None:
        super().__init__('Firing Rate', None, data_manager)

        self._plot_widget = pg.PlotWidget()
        """ The firing rate histograms for all neurons in the current display list are rendered in this widget. """
        self._plot_item: pg.PlotItem = self._plot_widget.getPlotItem()
        """ The graphics item that manages plotting of the firing rate histograms. """
        self._histograms: List[pg.PlotDataItem] = list()
        """ 
        The plot data item at pos N renders the firing histogram for the N-th neuron in the display focus list. The 
        trace color matches the color assigned to each neuron in the display list.
        """
        self._bin_size_combo = QComboBox()
        """ Combo box selects the bin size for the firing rate histograms, in seconds. """

        # one-time configuration
        self._plot_item.setMenuEnabled(enableMenu=False)
        self._plot_item.getViewBox().setMouseEnabled(x=False, y=False)
        self._plot_item.hideButtons()
        x_axis: pg.AxisItem = self._plot_item.getAxis('bottom')
        x_axis.setLabel(text='time (seconds)')

        # pre-create the plot data items for the maximum number of units in the display focus list
        for i in range(Analyzer.MAX_NUM_FOCUS_NEURONS):
            color = QColor.fromString(Analyzer.FOCUS_NEURON_COLORS[i])
            pen = pg.mkPen(color=color, width=self._PEN_WIDTH)
            self._histograms.append(self._plot_item.plot(x=[], y=[], pen=pen))

        # set up the combo box that selects bin size (note we have to convert to/from str
        self._bin_size_combo.addItems([str(k) for k in self._BIN_SIZE_CHOICES])
        self._bin_size_combo.setCurrentText(str(self._DEF_BIN_SIZE))
        self._bin_size_combo.currentTextChanged.connect(self._on_bin_size_changed)

        label = QLabel("Bin size (s):")
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self._plot_widget)
        control_line = QHBoxLayout()
        control_line.addStretch(1)
        control_line.addWidget(label)
        control_line.addWidget(self._bin_size_combo)
        main_layout.addLayout(control_line)

        self.view_container.setLayout(main_layout)

    def _refresh(self) -> None:
        """
        Refresh the firing rate histogram data rendered in this view in response to a change in the set of displayed
        neurons, or a change in the histogram bin size.
            This method simply updates the data item corresponding to the firing rate histogram for the neuron in each
        'slot' in the display list. When a display slot is unused, the corresponding data items contain empty arrays
        and thus renders nothing.
            The firing rate histograms are computed on the fly in the :class:`Neuron` instance. However, this
        computation is fast and should not impact GUI responiveness.
        """
        displayed = self.data_manager.neurons_with_display_focus

        bin_size = int(self._bin_size_combo.currentText())
        for k in range(Analyzer.MAX_NUM_FOCUS_NEURONS):
            unit: Optional[Neuron] = displayed[k] if k < len(displayed) else None
            if unit is None:
                self._histograms[k].setData(x=[], y=[])
            else:
                fr_hist = unit.firing_rate_histogram(bin_size, self.data_manager.channel_recording_duration_seconds)
                n = len(fr_hist)
                self._histograms[k].setData(x=np.linspace(start=bin_size/2, stop=bin_size*n, num=n), y=fr_hist)

    def on_working_directory_changed(self) -> None:
        end = int(self.data_manager.channel_recording_duration_seconds)
        if end > 0:
            x_axis: pg.AxisItem = self._plot_item.getAxis('bottom')
            x_axis.setTicks([
                [(0, "0"), (end, str(end))], []
            ])
        self._refresh()

    def on_focus_neurons_changed(self) -> None:
        self._refresh()

    @Slot(str)
    def _on_bin_size_changed(self, _: str) -> None:
        """ Handler refreshes the plotted firing rate histograms whenever the user changes the bin size. """
        self._refresh()

    def save_settings(self, settings: QSettings) -> None:
        """ Overridden to preserve the current bin sizer for computing the firing rate histograms. """
        settings.setValue('firing_rate_view_bin_size', self._bin_size_combo.currentText())

    def restore_settings(self, settings: QSettings) -> None:
        """ Overridden to restore the current firing rate histogram bin size from user settings. """
        try:
            bin_size_str = settings.value('firing_rate_view_bin_size')
            if int(bin_size_str) in self._BIN_SIZE_CHOICES:
                self._bin_size_combo.setCurrentText(bin_size_str)
                self._on_bin_size_changed()
        except Exception:
            pass
