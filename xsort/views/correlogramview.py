from typing import Optional

from PySide6.QtCore import Qt, Slot, QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QSlider, QCheckBox
import pyqtgraph as pg
import numpy as np
from pyqtgraph import ViewBox

from xsort.data.analyzer import Analyzer
from xsort.data.neuron import Neuron, DataType
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
        The ACG/CCG plots include two annotations: a dashed white horizontal line at zero correlation, and a translucent
    white vertical band spanning the range T=-1.5 to +1.5ms.
        There are two controls at the bottom of the view that affect the appearance of the ACG/CCG plots. A slider lets
    the user change the visible span of all ACG/CCGs currently drawn. A change in the slider position merely changes the
    X-axis range on all subplots. A checkbox toggles the visibility of the zero correlation line.
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
        self._show_zero_cb = QCheckBox("Show zero correlation")
        """ If checked, the plots include a horizontal dashed line at Y=0 (zero correlation). """
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

        self._show_zero_cb.stateChanged.connect(self._reset)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self._layout_widget)
        control_line = QHBoxLayout()
        control_line.addWidget(self._show_zero_cb)
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
                hist = displayed[i].cached_acg if i == j else displayed[i].get_cached_ccg(displayed[j].uid)
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
                pi.hideAxis('bottom')

                # annotations: a horizontal line at zero correlation, and a linear vertical region spanning +/-1.5ms.
                # The zero correlation line may be toggled on/off
                if self._show_zero_cb.isChecked():
                    pi.addItem(pg.InfiniteLine(
                        pos=0, angle=0, movable=False,
                        pen=pg.mkPen(color='w', style=Qt.PenStyle.DashLine, width=self._PEN_WIDTH)))
                pi.addItem(pg.LinearRegionItem(values=(-1.5, 1.5), pen=pg.mkPen(None), brush=pg.mkBrush("#FFFFFF40"),
                                               movable=False), ignoreBounds=True)

                # put horizontal time scale bar in the bottom right plot
                if (i == num_units - 1) and (j == num_units - 1):
                    self._scale_bar = pg.ScaleBar(size=10, suffix='ms')
                    self._scale_bar.setParentItem(pi.getViewBox())
                    self._scale_bar.anchor(itemPos=(0, 1), parentPos=(1, 1), offset=(-20, -30))

    def on_working_directory_changed(self) -> None:
        self._reset()

    def on_focus_neurons_changed(self, _: bool) -> None:
        self._reset()

    def on_focus_neurons_stats_updated(self, data_type: DataType, uid: str) -> None:
        """
        Whenever ACG/CCG stats are updated for a neural unit in the current display list, we need to update the
        corresponding plot data items within the subplots currently installed in this view.
        """
        if (data_type != DataType.ACG) and (data_type != DataType.CCG):
            return
        displayed = self.data_manager.neurons_with_display_focus
        num_units = len(displayed)
        span_ms = Neuron.FIXED_HIST_SPAN_MS
        unit: Neuron
        for i, unit in enumerate(displayed):
            if unit.uid == uid:
                for j in range(num_units):
                    hist = unit.cached_acg if i == j else unit.get_cached_ccg(displayed[j].uid)
                    plot_item = self._layout_widget.ci.getItem(row=i, col=j)
                    if isinstance(plot_item, pg.PlotItem):
                        pdi = next(iter(plot_item.listDataItems()), None)
                        if isinstance(pdi, pg.PlotDataItem):
                            pdi.setData(x=np.linspace(start=-span_ms, stop=span_ms, num=len(hist)), y=hist)
                break

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
        """
        Overridden to preserve view-specific settings: (1) the current histogram span, which is user selectable between
        20-200ms; and (2) whether or not the zero-correlation line is shown.
        """
        settings.setValue('correlogram_view_span', self._hist_span_slider.sliderPosition())
        settings.setValue('correlogram_view_show_zero', self._show_zero_cb.isChecked())

    def restore_settings(self, settings: QSettings) -> None:
        """
        Overridden to restore the current histogram span and "show zero correlation line" flag from user settings.
        """
        try:
            shown: bool = (settings.value('correlogram_view_show_zero', defaultValue="true") == "true")
            self._show_zero_cb.setChecked(shown)
            span = int(settings.value('correlogram_view_span'))
            self._hist_span_slider.setSliderPosition(span)
            self._on_hist_span_changed()
        except Exception:
            pass
