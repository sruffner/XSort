from typing import List, Optional

from PySide6.QtCore import Qt, Slot, QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QSlider
import pyqtgraph as pg
import numpy as np
from pyqtgraph import ViewBox

from xsort.data.analyzer import Analyzer
from xsort.data.neuron import Neuron, DataType
from xsort.views.baseview import BaseView


class ISIView(BaseView):
    """
    This view renders the interspike interval (ISI) histogram for each unit in the current display/focus list.
        Like autocorrelograms and cross-correlograms, the ISIs are computed and cached lazily during application
    runtime -- as a unit gets added to display list. :class:`Neuron` contains the infrastructure for computing these
    histograms with a fixed span of 200ms that should be sufficient for our purposes. Analyzer queues a background task
    to do the computations whenever the histograms have not yet been computed and cached for any unit on the current
    display list.
        As with many of the views rendering plots for neurons in the display list, a plot item is pre-created to render
    the ISI for the unit in each possible "slot" in the display list. Initially, these plot items contain empty (X,Y)
    data arrays. The view is "refreshed" by updating these data arrays whenever there's a change in the focus list or in
    the stats of any neurons in the focus list.
        A slider control at the bottom of the view allows the user to zoom in on the ISI histogram timescale, between
    20 and 200ms.
    """

    _MIN_TSPAN_MS: int = 20
    """ Minimum time span of the ISI histograms displayed in this view. """
    _MAX_TSPAN_MS: int = Neuron.FIXED_HIST_SPAN_MS
    """ Maximum time span of the ISI histograms displayed in this view. """
    _PEN_WIDTH: int = 3
    """ Width of pen used to draw ISI histograms. """

    def __init__(self, data_manager: Analyzer) -> None:
        super().__init__('Interspike Interval Histograms', None, data_manager)
        self._isi_plot_widget = pg.PlotWidget()
        """ Interspike interval histograms for all neurons in the current display list are rendered in this widget. """
        self._isi_plot_item: pg.PlotItem = self._isi_plot_widget.getPlotItem()
        """ The graphics item that manages plotting of interspike interval histograms. """
        self._isi_data: List[pg.PlotDataItem] = list()
        """ 
        The plot data item at pos N renders the ISI histogram for the N-th neuron in the display focus list. The 
        trace color matches the color assigned to each neuron in the display list.
        """
        self._hist_span_slider = QSlider(orientation=Qt.Orientation.Horizontal)
        """ 
        Slider sets the X-axis range for ISI histogram plot, letting user adjust the displayed span (in milliseconds).
        This will be some or all of the computed span, which is fixed.
        """

        # some configuration
        self._isi_plot_item.setMenuEnabled(False)
        self._isi_plot_item.hideButtons()
        self._isi_plot_item.getAxis("bottom").setLabel("msec")
        vb = self._isi_plot_item.getViewBox()
        vb.setMouseEnabled(x=False, y=False)
        vb.disableAutoRange(axis=ViewBox.XAxis)
        vb.setXRange(min=0, max=self._MAX_TSPAN_MS)

        # pre-create the plot data items for the maximum number of units in the display focus list
        for i in range(Analyzer.MAX_NUM_FOCUS_NEURONS):
            color = QColor.fromString(Analyzer.FOCUS_NEURON_COLORS[i])
            pen = pg.mkPen(color=color, width=self._PEN_WIDTH)
            self._isi_data.append(self._isi_plot_item.plot(x=[], y=[], pen=pen))

        self._hist_span_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self._hist_span_slider.setRange(self._MIN_TSPAN_MS, self._MAX_TSPAN_MS)
        self._hist_span_slider.setSliderPosition(self._MAX_TSPAN_MS)
        self._hist_span_slider.valueChanged.connect(self._on_hist_span_changed)

        min_label = QLabel(f"{self._MIN_TSPAN_MS} ms")
        min_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        max_label = QLabel(f"{self._MAX_TSPAN_MS} ms")
        max_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self._isi_plot_widget)
        control_line = QHBoxLayout()
        control_line.addStretch(2)
        control_line.addWidget(min_label)
        control_line.addWidget(self._hist_span_slider, stretch=1)
        control_line.addWidget(max_label)
        main_layout.addLayout(control_line)

        self.view_container.setLayout(main_layout)

    def _refresh(self) -> None:
        """
        Refresh the ISI histogram data rendered in this view in response to a change in the neuron display list, or
        upon retrieval of the metrics for a neuron.
            This method simply updates the data item corresponding to the ISI histogram for each "slot" in the neuron
        display list. When a display slot is unused, the corresponding data item contains empty (X,Y) arrays and thus
        renders nothing.
            By design, the ISI histograms are cached in the :class:`Neuron` instance. They are lazily computed on a
        background thread; if not yet available, the histograms will be empty arrays, so again nothing is rendered.
        """
        displayed = self.data_manager.neurons_with_display_focus

        span_ms = Neuron.FIXED_HIST_SPAN_MS
        for k in range(Analyzer.MAX_NUM_FOCUS_NEURONS):
            unit: Optional[Neuron] = displayed[k] if k < len(displayed) else None
            if unit is None:
                self._isi_data[k].setData(x=[], y=[])
            else:
                isi = unit.cached_isi
                self._isi_data[k].setData(x=np.linspace(start=0, stop=span_ms, num=len(isi)), y=isi)

    def on_working_directory_changed(self) -> None:
        self._refresh()

    def on_focus_neurons_changed(self) -> None:
        self._refresh()

    def on_focus_neurons_stats_updated(self, data_type: DataType, unit_label: str) -> None:
        if data_type == DataType.ISI:
            self._refresh()

    @Slot()
    def _on_hist_span_changed(self):
        span = self._hist_span_slider.sliderPosition()
        self._isi_plot_item.getViewBox().setXRange(min=0, max=span)

    def save_settings(self, settings: QSettings) -> None:
        """ Overridden to preserve the current ISI histogram span, which is user selectable between 20-200ms. """
        settings.setValue('isi_view_span', self._hist_span_slider.sliderPosition())

    def restore_settings(self, settings: QSettings) -> None:
        """ Overridden to restore the current histogram span from user settings. """
        try:
            span = int(settings.value('isi_view_span'))
            self._hist_span_slider.setSliderPosition(span)
            self._on_hist_span_changed()
        except Exception:
            pass
