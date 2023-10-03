from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHBoxLayout, QSpinBox, QLabel, QVBoxLayout
import pyqtgraph as pg
import numpy as np

from xsort.data.analyzer import Analyzer, MAX_NUM_FOCUS_NEURONS, FOCUS_NEURON_COLORS
from xsort.data.neuron import Neuron
from xsort.views.baseview import BaseView


class StatisticsView(BaseView):
    """ TODO: UNDER DEV """

    _MIN_TSPAN_MS: int = 20
    """ The minimum allowed time span for the histograms computed in this view, in milliseconds. """
    _MAX_TSPAN_MS: int = 500
    """ The maximum allowed time span for the histograms computed in this view, in milliseconds. """
    _DEF_TSPAN_MS: int = 100
    """ The default time span for the histograms computed in this view, in milliseconds. """
    _PEN_WIDTH: int = 1
    """ Width of pen used to draw histograms. """

    def __init__(self, data_manager: Analyzer) -> None:
        super().__init__('Statistics', None, data_manager)
        self._isi_plot_widget = pg.PlotWidget()
        """ Interspike interval histograms for all neurons in the current display list are rendered in this widget. """
        self._isi_plot_item: pg.PlotItem = self._isi_plot_widget.getPlotItem()
        """ The graphics item that manages plotting of interspike interval histograms. """
        self._acg_plot_widget = pg.PlotWidget()
        """ Autocorrelograms for all neurons in the current display list are rendered in this widget. """
        self._acg_plot_item: pg.PlotItem = self._acg_plot_widget.getPlotItem()
        """ The graphics item that manages plotting of autocorrelograms. """
        self._ccg_plot_widget = pg.PlotWidget()
        """ Crosscorrelograms of the first neuron vs the other neurons in the display list are rendered here. """
        self._ccg_plot_item: pg.PlotItem = self._ccg_plot_widget.getPlotItem()
        """ The graphics item that manages plotting of crosscorrelograms. """
        self._isi_data: List[pg.PlotDataItem] = list()
        """ 
        The plot data item at pos N renders the ISI histogram for the N-th neuron in the display focus list. The 
        trace color matches the color assigned to each neuron in the display list, albeit semi-transparent to help 
        visualize trace overlaps.
        """
        self._acg_data: List[pg.PlotDataItem] = list()
        """ 
        The plot data item at pos N renders the autocorrelogram for the N-th neuron in the display focus list. The 
        trace color matches the color assigned to each neuron in the display list, albeit semi-transparent to help 
        visualize trace overlaps.
        """
        self._ccg_data: List[pg.PlotDataItem] = list()
        """ 
        The plot data item at pos N renders the crosscorrelogram for the first neuron in the display list against the
        N-th neuron in that list. The first plot, of course, is the autocorrelogram for the first neuron. The trace 
        color matches the color assigned to the N-th neuron in the list, albeit semi-transparent to help visualize
        trace overlaps.
        """
        self._tspan_spinner = QSpinBox()
        """ Spinner sets/displays the time span used for all 3 histograms in the view. """

        # some configuration
        for t in [(self._isi_plot_item, "ISI"), (self._acg_plot_item, "ACG"), (self._ccg_plot_item, "CCG")]:
            pi, title = t[0], t[1]
            pi.setMenuEnabled(False)
            pi.getViewBox().setMouseEnabled(x=False, y=False)
            pi.hideButtons()
            pi.getAxis("bottom").setLabel("msec")
            pi.setTitle(title, size='14pt', color='#FFFFFF', bold=True, italic=True)

        # pre-create the plot data items for the maximum number of units in the display focus list
        for i in range(MAX_NUM_FOCUS_NEURONS):
            color = QColor.fromString(FOCUS_NEURON_COLORS[i])
            # color.setAlpha(128)
            pen = pg.mkPen(color=color, width=self._PEN_WIDTH)
            self._isi_data.append(self._isi_plot_item.plot(x=[], y=[], pen=pen))
            self._acg_data.append(self._acg_plot_item.plot(x=[], y=[], pen=pen))
            self._ccg_data.append(self._ccg_plot_item.plot(x=[], y=[], pen=pen))

        self._tspan_spinner.setRange(self._MIN_TSPAN_MS, self._MAX_TSPAN_MS)
        self._tspan_spinner.setValue(self._DEF_TSPAN_MS)
        span_label = QLabel("Span (msec): ")
        span_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        plot_layout = QHBoxLayout()
        plot_layout.addWidget(self._isi_plot_widget)
        plot_layout.addWidget(self._acg_plot_widget)
        plot_layout.addWidget(self._ccg_plot_widget)
        control_line = QHBoxLayout()
        control_line.addStretch(1)
        control_line.addWidget(span_label)
        control_line.addWidget(self._tspan_spinner)
        control_line.addStretch(1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(plot_layout)
        main_layout.addLayout(control_line)
        self.view_container.setLayout(main_layout)

    def _refresh(self) -> None:
        """
        Refresh the histogram data rendered in this view in response to a change in the set of displayed neurons, a
        change in the span over which the histograms are computed, or upon retrieval of the metrics for a neuron.
            In all cases, the data associated with all plot data items is recomputed. If this proves too slow, we will
        have to revisit the implementation of this method.
        """
        t_span_ms = self._tspan_spinner.value()
        displayed = self.data_manager.neurons_with_display_focus

        primary_neuron: Optional[Neuron] = displayed[0] if len(displayed) > 0 else None
        for k in range(MAX_NUM_FOCUS_NEURONS):
            unit: Optional[Neuron] = displayed[k] if k < len(displayed) else None
            if unit is None:
                self._isi_data[k].setData(x=[], y=[])
                self._acg_data[k].setData(x=[], y=[])
                self._ccg_data[k].setData(x=[], y=[])
            else:
                isi = unit.isi_histogram(max_isi_ms=t_span_ms)
                self._isi_data[k].setData(x=np.linspace(start=0, stop=t_span_ms, num=len(isi)), y=isi)
                acg = unit.crosscorrelogram(span_ms=t_span_ms)
                self._acg_data[k].setData(x=np.linspace(start=-t_span_ms, stop=t_span_ms, num=len(acg)), y=acg)
                ccg = primary_neuron.crosscorrelogram(other=unit.spike_times, span_ms=t_span_ms)
                self._ccg_data[k].setData(x=np.linspace(start=-t_span_ms, stop=t_span_ms, num=len(ccg)), y=ccg)

    def on_working_directory_changed(self) -> None:
        self._refresh()

    def on_neuron_metrics_updated(self, unit_label: str) -> None:
        pass  # self._refresh()

    def on_focus_neurons_changed(self) -> None:
        self._refresh()
