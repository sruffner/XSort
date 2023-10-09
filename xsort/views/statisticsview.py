from typing import List, Optional

from PySide6.QtCore import Qt, Slot, QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QSlider
import pyqtgraph as pg
import numpy as np
from pyqtgraph import ViewBox

from xsort.data.analyzer import Analyzer
from xsort.data.neuron import Neuron
from xsort.views.baseview import BaseView


class StatisticsView(BaseView):
    """
    TODO: CONTINUE HERE WITH REIMPLEMENTATION

    TODO: UNDER DEV

    The histograms displayed in this view can take a noticeable amount of time to prepare, but they don't take up
    much memory. Rather than cacheing them with the other unit metrics in a file cache, they are computed and cached
    lazily during application runtime -- as a unit gets added to the display/focus list. Neuron contains the
    infrastructure for computing the histograms with a fixed span of 200ms that should be sufficient for our purposes.
    Analyzer queues a background task to do the computations whenever the histograms have not yet been computed and
    cached for any unit on the current display/focus list.

    The StatisticsView simply refreshes whenever there's a change in the focus list or in the stats of any neurons in
    the focus list. It also allows the user to zoom in on the histograms's timescale....
    """

    _MIN_TSPAN_MS: int = 20
    """ Minimum time span of the histograms (ISI, ACG, CCG) displayed in this view. """
    _MAX_TSPAN_MS: int = Neuron.FIXED_HIST_SPAN_MS
    """ Maximum time span of the histograms displayed in this view. """
    _PEN_WIDTH: int = 3
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
        trace color matches the color assigned to each neuron in the display list.
        """
        self._ccg_data: List[pg.PlotDataItem] = list()
        """ 
        The plot data item at pos N renders the crosscorrelogram for the first neuron in the display list against the
        N-th neuron in that list. The first plot, of course, would be the autocorrelogram for the first neuron -- that
        plot is ALWAYS empty. The trace color matches the color assigned to the N-th neuron in the list.
        """
        self._hist_span_slider = QSlider(orientation=Qt.Orientation.Horizontal)
        """ 
        Slider sets the X-axis range for all plots, letting user adjust the displayed span of the ISI/ACG/CCGs (in 
        milliseconds_. This will be some or all of the computed span, which is fixed.
        """

        # some configuration
        for t in [(self._isi_plot_item, "ISI"), (self._acg_plot_item, "ACG"), (self._ccg_plot_item, "CCG")]:
            pi, title = t[0], t[1]
            pi.setMenuEnabled(False)

            vb = pi.getViewBox()
            vb.setMouseEnabled(x=False, y=False)
            vb.disableAutoRange(axis=ViewBox.XAxis)
            vb.setXRange(min=(0 if title == "ISI" else -self._MIN_TSPAN_MS), max=self._MAX_TSPAN_MS)

            pi.hideButtons()
            pi.getAxis("bottom").setLabel("msec")
            pi.setTitle(title, size='14pt', color='#FFFFFF', bold=True, italic=True)

        # pre-create the plot data items for the maximum number of units in the display focus list
        for i in range(Analyzer.MAX_NUM_FOCUS_NEURONS):
            color = QColor.fromString(Analyzer.FOCUS_NEURON_COLORS[i])
            # color.setAlpha(128)
            pen = pg.mkPen(color=color, width=self._PEN_WIDTH)
            self._isi_data.append(self._isi_plot_item.plot(x=[], y=[], pen=pen))
            self._acg_data.append(self._acg_plot_item.plot(x=[], y=[], pen=pen))
            self._ccg_data.append(self._ccg_plot_item.plot(x=[], y=[], pen=pen))

        self._hist_span_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self._hist_span_slider.setRange(self._MIN_TSPAN_MS, self._MAX_TSPAN_MS)
        self._hist_span_slider.setSliderPosition(self._MAX_TSPAN_MS)
        self._hist_span_slider.valueChanged.connect(self._on_hist_span_changed)

        min_label = QLabel(f"{self._MIN_TSPAN_MS} ms")
        min_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        max_label = QLabel(f"{self._MAX_TSPAN_MS} ms")
        max_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        plot_layout = QHBoxLayout()
        plot_layout.addWidget(self._isi_plot_widget)
        plot_layout.addWidget(self._acg_plot_widget)
        plot_layout.addWidget(self._ccg_plot_widget)
        control_line = QHBoxLayout()
        control_line.addStretch(2)
        control_line.addWidget(min_label)
        control_line.addWidget(self._hist_span_slider, stretch=1)
        control_line.addWidget(max_label)

        main_layout = QVBoxLayout()
        main_layout.addLayout(plot_layout)
        main_layout.addLayout(control_line)
        self.view_container.setLayout(main_layout)

    def _refresh(self) -> None:
        """
        Refresh the histogram data rendered in this view in response to a change in the set of displayed neurons, or
        upon retrieval of the metrics for a neuron.
            This method simply updates the data item corresponding to each histogram type (ISI, ACG, CCG) and each
        'slot' in the display list. When a display slot is unused, the corresponding data items contain empty arrays
        and thus render nothing.
            By design, the histograms are cached in the :class:`Neuron` instance. They are lazily computed on a
        background thread; if not yet available, the histograms will be empty arrays, so again nothing is rendered.
        """
        displayed = self.data_manager.neurons_with_display_focus

        span_ms = Neuron.FIXED_HIST_SPAN_MS
        primary_neuron: Optional[Neuron] = displayed[0] if len(displayed) > 0 else None
        for k in range(Analyzer.MAX_NUM_FOCUS_NEURONS):
            unit: Optional[Neuron] = displayed[k] if k < len(displayed) else None
            if unit is None:
                self._isi_data[k].setData(x=[], y=[])
                self._acg_data[k].setData(x=[], y=[])
                self._ccg_data[k].setData(x=[], y=[])
            else:
                isi = unit.cached_isi
                self._isi_data[k].setData(x=np.linspace(start=0, stop=span_ms, num=len(isi)), y=isi)
                acg = unit.cached_acg
                self._acg_data[k].setData(x=np.linspace(start=-span_ms, stop=span_ms, num=len(acg)), y=acg)
                if unit.label != primary_neuron.label:
                    ccg = primary_neuron.get_cached_ccg(unit.label)
                    self._ccg_data[k].setData(x=np.linspace(start=-span_ms, stop=span_ms, num=len(ccg)), y=ccg)

    def on_working_directory_changed(self) -> None:
        self._refresh()

    def on_neuron_metrics_updated(self, unit_label: str) -> None:
        self._refresh()

    def on_focus_neurons_changed(self) -> None:
        self._refresh()

    def on_focus_neurons_stats_updated(self) -> None:
        self._refresh()

    @Slot()
    def _on_hist_span_changed(self):
        span = self._hist_span_slider.sliderPosition()
        self._isi_plot_item.getViewBox().setXRange(min=0, max=span)
        self._acg_plot_item.getViewBox().setXRange(min=-span, max=span)
        self._ccg_plot_item.getViewBox().setXRange(min=-span, max=span)

    def save_settings(self, settings: QSettings) -> None:
        """ Overridden to preserve the current histogram span, which is user selectable between 20-200ms. """
        settings.setValue('statistics_view_span', self._hist_span_slider.sliderPosition())

    def restore_settings(self, settings: QSettings) -> None:
        """ Overridden to restore the current histogram span from user settings. """
        try:
            span = int(settings.value('statistics_view_span'))
            self._hist_span_slider.setSliderPosition(span)
            self._on_hist_span_changed()
        except Exception:
            pass
