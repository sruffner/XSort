from typing import Optional

from PySide6.QtCore import Qt, Slot, QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QSlider
import pyqtgraph as pg
from pyqtgraph import ViewBox

from xsort.data.analyzer import Analyzer
from xsort.data.neuron import Neuron, DataType
from xsort.views.baseview import BaseView


class ACGRateView(BaseView):
    """
    This view displays -- for each neural unit currently selected for display/comparision -- a 3D histogram representing
    the unit's autocorrelogram as a function of instantaneous firing rate.
        The sampled firing rates (as observed at each spike occurrence) are distributed across 10 bins, and the
    autocorrelograms for each firing rate bin span the time range [-100 .. 100] milliseconds. These parameters cannot be
    changed by the user. The series of 10 ACGs are plotted as a heatmap image with 10 rows and 201 columns.
        The neuron display list, managed by :class:`Analyzer`, may contain up to 3 different neural units. This view
    renders a single row of up to 3 subplots, with the ACG-vs-firing-rate histogram for each unit in the list in a
    separate subplot. The subplot title reflects the unit UID and the range of firing rates spanned along the Y-axis.
        The ACG-vs-rate historgrams are computed lazily as needed, then cached in the :class:`Neuron` instances. This
    view is refreshed whenever the ACG-vs-rate histogram of a unit in the current display list is updated.
        Finally, a slider at the bottom of the view allows the user to change the visible correlogram span of all
    ACG-vs-rate histograms currently drawn. A change in the slider position merely changes the X-axis range on all
    subplots.
    """

    _MIN_TSPAN_MS: int = 20
    """ Minimum time span of the ACG-vs-firing rate histograms displayed in this view, in milliseconds. """
    _MAX_TSPAN_MS: int = Neuron.ACG_VS_RATE_SPAN_MS
    """ Maximum time span of the ACG-vs-firing rate histograms displayed in this view. """
    _PEN_WIDTH: int = 3
    """ Width of pen used to draw histograms. """
    _COLORMAP: str = 'viridis'
    """ The PyQtGraph colormap used to render the 3D histograms as heatmaps. """
    _TITLE_FONT_SZ: str = '14pt'
    """ Sets font size of plot title which displays range of firing rates for the ACG-vs-FR histogram in that plot. """

    def __init__(self, data_manager: Analyzer, settings: QSettings) -> None:
        super().__init__('ACG-vs-Firing Rate', None, data_manager, settings)
        self._layout_widget = pg.GraphicsLayoutWidget()
        """ Layout widget in which the ACG-vs-rate histogram for all unit in the current display list are arranged. """
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
        span_ms = Neuron.ACG_VS_RATE_SPAN_MS
        visible_span = self._hist_span_slider.sliderPosition()
        for i in range(len(displayed)):
            # create the subplot item
            pi: pg.PlotItem = self._layout_widget.addPlot(row=0, col=i)
            pi.setMenuEnabled(False)
            vb = pi.getViewBox()
            vb.setMouseEnabled(x=False, y=False)
            vb.disableAutoRange(axis=ViewBox.XAxis)
            vb.setXRange(min=-visible_span, max=visible_span, padding=0.05)
            pi.hideButtons()
            pi.hideAxis('bottom')
            pi.hideAxis('left')

            color = QColor.fromString(Analyzer.FOCUS_NEURON_COLORS[i])
            rate_bins, acgs = displayed[i].cached_acg_vs_rate
            if len(rate_bins) > 0:
                n_bins = len(rate_bins)
                rate_bin_size = (rate_bins[-1] - rate_bins[0]) / n_bins
                img_rect = [-span_ms, rate_bins[0] - rate_bin_size/2.0, 2.0*span_ms, rate_bin_size*n_bins]
                img = pg.ImageItem(image=acgs, axisOrder='row-major', rect=img_rect, colorMap=self._COLORMAP)
                pi.addItem(img)
                title = f"Unit {displayed[i].uid} : {rate_bins[0]:.1f}-{rate_bins[-1]:.1f} Hz"
                pi.setTitle(title, color=color, bold=True, size=self._TITLE_FONT_SZ)

            if i == num_units - 1:
                self._scale_bar = pg.ScaleBar(size=20, suffix='ms')
                self._scale_bar.setParentItem(pi.getViewBox())
                self._scale_bar.anchor(itemPos=(1, 1), parentPos=(1, 1), offset=(-10, -10))

    def on_working_directory_changed(self) -> None:
        self._reset()

    def on_focus_neurons_changed(self, _: bool) -> None:
        self._reset()

    def on_focus_neurons_stats_updated(self, data_type: DataType, uid: str) -> None:
        """
        Whenever the ACG-vs-rate histogram is updated for a neural unit in the current display list, we need to update
        the corresponding plot data item within the subplots currently installed in this view.
        """
        if data_type != DataType.ACG_VS_RATE:
            return
        displayed = self.data_manager.neurons_with_display_focus
        span_ms = Neuron.FIXED_HIST_SPAN_MS
        for i in range(len(displayed)):
            if displayed[i].uid == uid:
                plot_item: pg.PlotItem = self._layout_widget.ci.getItem(row=0, col=i)
                color = QColor.fromString(Analyzer.FOCUS_NEURON_COLORS[i])
                rate_bins, acgs = displayed[i].cached_acg_vs_rate
                if len(rate_bins) > 0:
                    n_bins = len(rate_bins)
                    rate_bin_size = (rate_bins[-1] - rate_bins[0]) / n_bins
                    img_rect = [-span_ms, rate_bins[0] - rate_bin_size / 2.0, 2.0 * span_ms, rate_bin_size * n_bins]
                    img = pg.ImageItem(image=acgs, axisOrder='row-major', rect=img_rect, colorMap=self._COLORMAP)
                    plot_item.addItem(img)
                    title = f"Unit {displayed[i].uid} : {rate_bins[0]:.1f}-{rate_bins[-1]:.1f} Hz"
                    plot_item.setTitle(title, color=color, bold=True,  size=self._TITLE_FONT_SZ)

    @Slot()
    def _on_hist_span_changed(self):
        """
        Update the X-axis range for all subplots IAW a change in the position of the slider that sets the visible
        span of the displayed ACG-vs-firing-rate histograms.
        """
        span = self._hist_span_slider.sliderPosition()
        num_units = len(self.data_manager.neurons_with_display_focus)
        if num_units == 0:
            return
        for i in range(num_units):
            plot_item = self._layout_widget.ci.getItem(row=0, col=i)
            if isinstance(plot_item, pg.PlotItem):
                plot_item.getViewBox().setXRange(min=-span, max=span, padding=0.05)

    def save_settings(self) -> None:
        """ Overridden to preserve the current histogram span, which is user selectable between 20-100ms. """
        self.settings.setValue('acg_vs_rate_view_span', self._hist_span_slider.sliderPosition())

    def restore_settings(self) -> None:
        """ Overridden to restore the current histogram span from user settings. """
        try:
            span = int(self.settings.value('acg_vs_rate_view_span'))
            self._hist_span_slider.setSliderPosition(span)
            self._on_hist_span_changed()
        except Exception:
            pass
