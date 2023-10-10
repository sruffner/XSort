from typing import Optional

from PySide6.QtCore import Qt, Slot, QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QSlider
import pyqtgraph as pg
import numpy as np
from pyqtgraph import ViewBox

from xsort.data.analyzer import Analyzer
from xsort.data.neuron import Neuron
from xsort.views.baseview import BaseView


class CorrelogramView(BaseView):
    """
    This view displays autocorrelograms and crosscorrelograms for the neurons selected for display/comparison.

        The neuron display list, managed by :class:`Analyzer`, may contain up to N different neural units. This
    view displays an NxN grid of subplots. Each subplot along the major diagonal -- **row == column == i** -- renders
    the autocorrelogram (ACG) for the i-th unit in the display list. Each off-diagonal subplot (i, j) renders the
    crosscorrelogram for the i-th unit vs the j-th unit in the display list.
        ACGs are drawn using the display color assigned to the corresponding neuron, while CCGs are drawn using a 50-50
    blend of the two colors assigned to the units being compared.
        Plot axes are are hidden entirely; a single horizontal scale bar installed in the bottom right subplot provides
    a time scale indication. Whenever the display list is empty, the view is empty except for a centered message label
    indicating that "No neurons are selected".
        ACGs and CCGs are computed lazily as needed, then cached in the :class:`Neuron` instances. This view is
    refreshed whenever the ACG/CCGs of a unit in the current display list are updated.
        Finally, a slider at the bottom of the view allows the user to change the visible span of all ACG/CCGs
    currently drawn. A change in the slider position merely changes the X-axis range on all subplots.
    """

    _MIN_TSPAN_MS: int = 20
    """ Minimum time span of the histograms (ISI, ACG, CCG) displayed in this view. """
    _MAX_TSPAN_MS: int = Neuron.FIXED_HIST_SPAN_MS
    """ Maximum time span of the histograms displayed in this view. """
    _PEN_WIDTH: int = 3
    """ Width of pen used to draw histograms. """

    def __init__(self, data_manager: Analyzer) -> None:
        super().__init__('Correlograms', None, data_manager)
        self._layout_widget = pg.GraphicsLayoutWidget()
        """ Layout widget in which the ACGs and CCGs for all neurons in the current display list are arranged. """
        self._hist_span_slider = QSlider(orientation=Qt.Orientation.Horizontal)
        """ 
        Slider sets the X-axis range for all plots, letting user adjust the displayed span of the ACG/CCGs (in 
        milliseconds). This will be some or all of the computed span, which is fixed.
        """
        self._scale_bar: Optional[pg.ScaleBar] = None
        """ Horizontal calibration bar indicating time scale in lieu of a visible X-axis. """

        self._reset()

        self._hist_span_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self._hist_span_slider.setRange(self._MIN_TSPAN_MS, self._MAX_TSPAN_MS)
        self._hist_span_slider.setSliderPosition(self._MAX_TSPAN_MS)
        self._hist_span_slider.valueChanged.connect(self._on_hist_span_changed)

        min_label = QLabel(f"{self._MIN_TSPAN_MS} ms")
        min_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        max_label = QLabel(f"{self._MAX_TSPAN_MS} ms")
        max_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self._layout_widget)
        control_line = QHBoxLayout()
        control_line.addStretch(2)
        control_line.addWidget(min_label)
        control_line.addWidget(self._hist_span_slider, stretch=1)
        control_line.addWidget(max_label)
        main_layout.addLayout(control_line)
        self.view_container.setLayout(main_layout)

    def _reset(self) -> None:
        """
        Whenever the working directory changes or the neuron display list is modified, clear all graphic items from
        the layout widget central to this view, and repopulate it.
        """
        # before clearing the graphics layout widget, must carefully disconnect our scale bar from the item hierarchy,
        # else the pyqtgraph library throws unpleasant exceptions.
        if self._scale_bar is not None:
            vb = self._scale_bar.parentItem()
            self._scale_bar.offset = None
            vb.removeItem(self._scale_bar)
            self._scale_bar = None
        self._layout_widget.ci.clear()

        displayed = self.data_manager.neurons_with_display_focus
        if len(displayed) == 0:
            self._layout_widget.ci.addLabel("No neurons selected")
            return

        num_units = len(displayed)
        span_ms = Neuron.FIXED_HIST_SPAN_MS
        visible_span = self._hist_span_slider.sliderPosition()
        for i, unit in enumerate(displayed):
            for j in range(num_units):
                hist = displayed[i].cached_acg if i == j else displayed[i].get_cached_ccg(displayed[j].label)
                color = QColor.fromString(Analyzer.FOCUS_NEURON_COLORS[i])
                if i != j:
                    color2 = QColor.fromString(Analyzer.FOCUS_NEURON_COLORS[j])
                    r = min(255, int(color.red() * 0.5 + color2.red() * 0.5))
                    g = min(255, int(color.green() * 0.5 + color2.green() * 0.5))
                    b = min(255, int(color.blue() * 0.5 + color2.blue() * 0.5))
                    color = QColor(r, g, b)
                pen = pg.mkPen(color=color, width=self._PEN_WIDTH)

                pi = self._layout_widget.ci.addPlot(
                    row=i, col=j, pen=pen, x=np.linspace(start=-span_ms, stop=span_ms, num=len(hist)), y=hist)
                pi.setMenuEnabled(False)
                vb = pi.getViewBox()
                vb.setMouseEnabled(x=False, y=False)
                vb.disableAutoRange(axis=ViewBox.XAxis)
                vb.setXRange(min=-visible_span, max=visible_span, padding=0.05)
                pi.hideButtons()
                pi.hideAxis('left')
                pi.hideAxis('bottom')
                if (i == num_units - 1) and (j == num_units - 1):
                    self._scale_bar = pg.ScaleBar(size=20, suffix='ms')
                    self._scale_bar.setParentItem(pi.getViewBox())
                    self._scale_bar.anchor(itemPos=(0, 1), parentPos=(0.75, 1), offset=(0, -20))

    def on_working_directory_changed(self) -> None:
        self._reset()

    def on_focus_neurons_changed(self) -> None:
        self._reset()

    def on_focus_neurons_stats_updated(self) -> None:
        """
        Whenever ACG/CCG stats are updated for neurons in the current display list, we need to update the plot data
        items for all subplots currently installed in this view.
        """
        displayed = self.data_manager.neurons_with_display_focus
        num_units = len(displayed)
        span_ms = Neuron.FIXED_HIST_SPAN_MS
        for i, unit in enumerate(displayed):
            for j in range(num_units):
                hist = displayed[i].cached_acg if i == j else displayed[i].get_cached_ccg(displayed[j].label)
                plot_item = self._layout_widget.ci.getItem(row=i, col=j)
                if isinstance(plot_item, pg.PlotItem):
                    pdi = next(iter(plot_item.listDataItems()), None)
                    if isinstance(pdi, pg.PlotDataItem):
                        pdi.setData(x=np.linspace(start=-span_ms, stop=span_ms, num=len(hist)), y=hist)

    @Slot()
    def _on_hist_span_changed(self):
        """
        Update the X-axis range for all subplots IAW a change in the position of the slider that sets the visible
        span of the displayed ACG/CCGs.
        """
        span = self._hist_span_slider.sliderPosition()
        num_units = len(self.data_manager.neurons_with_display_focus)
        if num_units == 0:
            return
        for i in range(num_units):
            for j in range(num_units):
                plot_item = self._layout_widget.ci.getItem(row=i, col=j)
                if isinstance(plot_item, pg.PlotItem):
                    plot_item.getViewBox().setXRange(min=-span, max=span, padding=0.05)

    def save_settings(self, settings: QSettings) -> None:
        """ Overridden to preserve the current histogram span, which is user selectable between 20-200ms. """
        settings.setValue('correlogram_view_span', self._hist_span_slider.sliderPosition())

    def restore_settings(self, settings: QSettings) -> None:
        """ Overridden to restore the current histogram span from user settings. """
        try:
            span = int(settings.value('correlogram_view_span'))
            self._hist_span_slider.setSliderPosition(span)
            self._on_hist_span_changed()
        except Exception:
            pass
