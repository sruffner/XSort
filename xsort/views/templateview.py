from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import Slot, QSettings
from PySide6.QtGui import QFont, QColor, Qt
from PySide6.QtWidgets import QHBoxLayout, QGraphicsTextItem, QSlider, QVBoxLayout, QLabel
import pyqtgraph as pg

from xsort.data.analyzer import Analyzer
from xsort.views.baseview import BaseView


class TemplateView(BaseView):
    """
    This view displays the per-channel spike template waveforms for all neurons currently selected for display in
    XSort. The displayed template span is user-configurable between 3-10ms. The per-channel templates are rendered in
    a single :class:`pyqtgraph.PlotWidget`, populated in order from the bottom to top and left to right by source
    channel index, with 4 templates per 'row'.
        Our strategy is to pre-create a list of :class:`pyqtgraph.PlotDataItem` representing the per-channel templates
    for the the first neuron in the display focus set, another list for the second unit, and so on. Each data item is
    assigned a pen color depending on the unit's position in the display focus list. All of the data items will have
    empty data sets initially,  and it is only the data sets that get updated when the display focus list changes or
    neuron metrics are updated.
    """

    trace_pen_width: int = 3
    """ Width of pen used to draw template plots. """
    bkg_color: str | QColor = 'default'   # pg.mkColor('#404040')
    """ Background color for the plot widget in which all spike templates are drawn. """

    _NUM_TEMPLATES_PER_ROW: int = 4
    """ Number of per-channel templates plotted per 'row' in the plot widget. """
    _H_OFFSET_REL: float = 1.2
    """ Multiply current template span by this factor to get horizontal offset between adjacent templates in a row. """
    _MIN_TSPAN_MS: int = 3
    """ Minimum template span in milliseconds. """
    _MAX_TSPAN_MS: int = 10
    """ Maximum template span in milliseconds. """
    _MIN_VSPAN_UV: int = 30
    """ Minimum +/- voltage range for templates, in microvolts. """
    _MAX_VSPAN_UV: int = 250
    """ Msximum +/- voltage range for templates, in microvolts. """
    _DEF_VSPAN_UV: int = 75
    """ Default +/- voltage range for templates, in microvolts. """
    def __init__(self, data_manager: Analyzer) -> None:
        super().__init__('Templates', None, data_manager)

        self._plot_widget = pg.PlotWidget(background=self.bkg_color)
        """ This plot widget fills the entire view (minus border) and contains all plotted spike templates. """
        self._plot_item: pg.PlotItem = self._plot_widget.getPlotItem()
        """ The graphics item within the plot widget that manages the plotting. """
        self._message_label: pg.LabelItem = pg.LabelItem("", size="24pt", color="#808080")
        """ This label is displayed centrally when template data is not available. """
        self._scale_bar: pg.ScaleBar = pg.ScaleBar(size=10, suffix='ms')
        """ Horizontal calibration bar indicating time scale in lieu of a visible X-axis. """
        self._channel_labels: Dict[int, pg.TextItem] = dict()
        """ 
        Dictionary maps analog channel index to corresponding text item that renders the channel label in the template
        plot. We keep track of these text items so we can reposition them when the user changes the template span.
        """
        self._spike_templates: List[Dict[int, pg.PlotDataItem]] = list()
        """ 
        Each dictionary in this list represents the plotted spike templates for a neural unit with the display focus,
        keyed by index of the analog source channel for each channel on which templates are computed. Position in the 
        list matches the position of the corresponding neuron in the display focus list.
        """
        self._tspan_slider = QSlider(orientation=Qt.Orientation.Horizontal)
        """ Slider controls the displayed span of each template in milliseconds. """
        self._vspan_slider = QSlider(orientation=Qt.Orientation.Horizontal)
        """ Slider controls the displayed height of each template in microvolts. """
        self._vspan_readout = QLabel(f"+/-{self._DEF_VSPAN_UV} \u00b5v")

        # one-time only configuration of the plot: disable the context menu; hide axes; position message
        # label centered over the plot (to indicate when no data is available); use a scale bar to indicate timescale.
        # initially: plot is empty with the message label indicating that no data is available
        self._plot_item.setMenuEnabled(enableMenu=False)
        self._plot_item.getViewBox().setMouseEnabled(x=False, y=False)
        self._plot_item.hideButtons()
        self._plot_item.hideAxis('left')
        self._plot_item.hideAxis('bottom')
        self._plot_item.setXRange(0, self._NUM_TEMPLATES_PER_ROW * (self._MAX_TSPAN_MS * self._H_OFFSET_REL))
        self._message_label.setParentItem(self._plot_item)
        self._message_label.anchor(itemPos=(0.5, 0.5), parentPos=(0.5, 0.5))
        self._scale_bar.setParentItem(self._plot_item.getViewBox())
        self._scale_bar.anchor(itemPos=(1, 1), parentPos=(1, 1), offset=(-20, -10))

        self._tspan_slider.setTickPosition(QSlider.TickPosition.TicksAbove)
        self._tspan_slider.setRange(self._MIN_TSPAN_MS, self._MAX_TSPAN_MS)
        self._tspan_slider.setTickInterval(1)
        self._tspan_slider.setSliderPosition(self._MAX_TSPAN_MS)
        self._tspan_slider.valueChanged.connect(self._on_template_span_change)

        min_label = QLabel(f"{self._MIN_TSPAN_MS} ms")
        min_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        max_label = QLabel(f"{self._MAX_TSPAN_MS} ms")
        max_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self._vspan_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self._vspan_slider.setRange(self._MIN_VSPAN_UV, self._MAX_VSPAN_UV)
        self._vspan_slider.setSliderPosition(self._DEF_VSPAN_UV)
        self._vspan_slider.valueChanged.connect(self._on_voltage_range_change)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self._plot_widget)
        control_line = QHBoxLayout()
        control_line.addWidget(self._vspan_readout)
        control_line.addWidget(self._vspan_slider, stretch=1)
        control_line.addStretch(1)
        control_line.addWidget(min_label)
        control_line.addWidget(self._tspan_slider, stretch=1)
        control_line.addWidget(max_label)
        main_layout.addLayout(control_line)
        self.view_container.setLayout(main_layout)

    def _reset(self) -> None:
        """
        Reset and reinitialize this view. All plot data items rendering per-channel spike templates are removed and then
        repopulated. If there are no analog channels, a static centered message label is shown indicating that there is
        no data. If the available analog channels are known, empty plot data items are created as 'placeholders' for
        all the per-channel spike templates for each of MAX_NUM_FOCUS_NEURONS neural units. The trace color for each
        plot data item matches the display color assigned to the corresponding unit (based on its position in the
        displayed neurons list). The centered message label will indicate that template data is not yet available. The
        actual template data are set later, as the data is retrieved.
            All template plots are distributed across the view box from bottom to top and left to right, with four
        templates per row, in ascending order by source channel index. Channel labels are created and positioned near
        the end of the template waveforms computed from that channel.
        """
        self._plot_item.clear()  # this does not remove _message_label or scale bar, which are in the internal viewbox
        self._channel_labels.clear()
        self._spike_templates.clear()
        if len(self.data_manager.channel_indices) == 0:
            self._message_label.setText("No channel data available")
        else:
            self._message_label.setText("No template data available")

            pen_colors: List[QColor] = list()
            for k in range(Analyzer.MAX_NUM_FOCUS_NEURONS):
                self._spike_templates.append(dict())
                color = QColor.fromString(Analyzer.FOCUS_NEURON_COLORS[k])
                pen_colors.append(color)

            t_span_ms = self._tspan_slider.sliderPosition()
            v_span_uv = self._vspan_slider.sliderPosition()
            row: int = 0
            col: int = 0
            for idx in self.data_manager.channel_indices:
                x = col * (t_span_ms * self._H_OFFSET_REL)
                y = row * (v_span_uv * 2)
                for k in range(Analyzer.MAX_NUM_FOCUS_NEURONS):
                    pen = pg.mkPen(pen_colors[k], width=self.trace_pen_width)
                    self._spike_templates[k][idx] = self._plot_item.plot(x=[], y=[], pen=pen)

                label = pg.TextItem(text=str(idx), anchor=(1, 1))
                ti: QGraphicsTextItem = label.textItem
                font: QFont = ti.font()
                font.setBold(True)
                label.setFont(font)
                label.setPos(x + t_span_ms, y + 5)
                self._plot_item.addItem(label)
                self._channel_labels[idx] = label

                col = col + 1
                if col == self._NUM_TEMPLATES_PER_ROW:
                    col = 0
                    row = row + 1
            num_channels = len(self.data_manager.channel_indices)
            num_rows = (int(num_channels/self._NUM_TEMPLATES_PER_ROW) +
                        (0 if (num_channels % self._NUM_TEMPLATES_PER_ROW == 0) else 1))
            self._plot_item.setYRange(-v_span_uv, (num_rows - 0.5) * (2 * v_span_uv))

    def on_working_directory_changed(self) -> None:
        """
        When the working directory changes, the number of analog data channels should be known, but neural unit spike
        template data will not be ready. Here we initialize the plot to show per-channel placeholder spike templates as
        "flat lines", each spanning 10 ms.
        """
        self._reset()

    def on_neuron_metrics_updated(self, uid: str) -> None:
        """
        If the unit metrics are updated for a neural unit currently displayed in this view, update the rendered spike
        templates for the unit across all available analog channels.

        :param uid: Label identifying the updated neural unit.
        """
        self._refresh(uid_updated=uid)

    def on_focus_neurons_changed(self) -> None:
        """
        Whenever the subset of neural units holding the display focus changes, update the plot to show the spike
        templates for each unit in the focus set that across all available analog channels.
        """
        self._refresh()

    @Slot()
    def _on_template_span_change(self) -> None:
        """ Refresh the entire view whenever the user changes the template span. """
        self._refresh(tspan_changed=True)

    @Slot()
    def _on_voltage_range_change(self) -> None:
        """ Refresh the entire view whenever the user changes the voltage range for the templates. """
        self._vspan_readout.setText(f"+/-{self._vspan_slider.sliderPosition()} \u00b5v")
        self._refresh(vspan_changed=True)

    def _refresh(self, tspan_changed: bool = False, vspan_changed: bool = False,
                 uid_updated: Optional[str] = None) -> None:
        """
        Refresh the view in response to a change in the set of displayed neurons, a change in the template span, or
        upon retrieval of the metrics for a neuron.
            If the template span has changed, the plot data items for all templates are updated IAW the new span, the
        channel labels in the plot are repositioned accordingly, and the size of the time scale bar adjusted. If the
        set of displayed neurons has changed, all plot data items are updated accordingly. When a neuron's metrics
        (including spike template waveforms) are retrieved, no action is taken it that neuron is not currently selected
        for display. If it is, then only the plot data items associated with that unit's templates are refreshed.
        Note that only one of these changes will occur at a time.

        :param tspan_changed: True if template time (X) span was changed. Default is False.
        :param vspan_changed: True if template voltage (Y) span was changed. Default is False.
        :param uid_updated: UID  identifying the neural unit for which metrics have just been retrieved from the
            application cache; otherwise None. Default is None
        """
        if len(self.data_manager.channel_indices) == 0:
            return
        t_span_ms = self._tspan_slider.sliderPosition()
        t_span_ticks = int(self.data_manager.channel_samples_per_sec * t_span_ms * 0.001)
        v_span_uv = self._vspan_slider.sliderPosition()
        displayed = self.data_manager.neurons_with_display_focus

        # if refresh is because a unit's metrics were updated, but that unit is not displayed, there's nothing to do.
        if isinstance(uid_updated, str) and not (uid_updated in [u.uid for u in displayed]):
            return

        for k, template_dict in enumerate(self._spike_templates):
            # when refresing because a unit's metrics were updated, only update the templates for that unit
            if isinstance(uid_updated, str) and ((k >= len(displayed)) or (displayed[k].uid != uid_updated)):
                continue
            row: int = 0
            col: int = 0
            for idx in self.data_manager.channel_indices:
                t0 = col * (t_span_ms * self._H_OFFSET_REL)
                y0 = row * (2 * v_span_uv)
                pdi = template_dict[idx]
                template = displayed[k].get_template_for_channel(idx) if (k < len(displayed)) else None
                if template is None:
                    pdi.setData(x=[], y=[])
                else:
                    template = template[0:t_span_ticks]
                    pdi.setData(x=np.linspace(start=t0, stop=t0+t_span_ms, num=len(template)),
                                y=y0 + template)

                # fix corresponding channel label when template time or voltage span changes
                if (tspan_changed or vspan_changed) and (k == 0):
                    label = self._channel_labels[idx]
                    label.setPos(t0 + t_span_ms, y0 + 5)

                col = col + 1
                if col == self._NUM_TEMPLATES_PER_ROW:
                    col = 0
                    row = row + 1

        num_channels = len(self.data_manager.channel_indices)
        num_rows = (int(num_channels / self._NUM_TEMPLATES_PER_ROW) +
                    (0 if (num_channels % self._NUM_TEMPLATES_PER_ROW == 0) else 1))
        self._plot_item.setYRange(-v_span_uv, (num_rows - 0.5) * (2 * v_span_uv))

        # fix x-axis range and scale bar when template span changes
        if tspan_changed:
            self._scale_bar.size = t_span_ms
            self._scale_bar.text.setText(f"{t_span_ms} ms")
            self._scale_bar.updateBar()
            self._plot_item.setXRange(0, self._NUM_TEMPLATES_PER_ROW * (t_span_ms * self._H_OFFSET_REL))

        self._message_label.setText("" if len(displayed) > 0 else "No units selected for display")

    def save_settings(self, settings: QSettings) -> None:
        """ Overridden to preserve the current template time and voltage spans, which are user selectable. """
        settings.setValue('template_view_span', self._tspan_slider.sliderPosition())
        settings.setValue('template_view_voltage_span', self._vspan_slider.sliderPosition())

    def restore_settings(self, settings: QSettings) -> None:
        """ Overridden to restore the template time and voltage spans from user settings. """
        try:
            t_span_ms = int(settings.value('template_view_span', self._MAX_TSPAN_MS))
            v_span_uv = int(settings.value('template_view_voltage_span', self._DEF_VSPAN_UV))
            t_span_ms = max(self._MIN_TSPAN_MS, min(t_span_ms, self._MAX_TSPAN_MS))
            v_span_uv = max(self._MIN_VSPAN_UV, min(v_span_uv, self._MAX_VSPAN_UV))

            # don't need to fix view if voltage span is not the default, since no templates are displayed when
            # this method is invoked.
            if v_span_uv != self._DEF_VSPAN_UV:
                self._vspan_slider.setSliderPosition(v_span_uv)
            if t_span_ms != self._MAX_TSPAN_MS:
                self._tspan_slider.setSliderPosition(t_span_ms)
                self._plot_item.setXRange(0, self._NUM_TEMPLATES_PER_ROW * (t_span_ms * self._H_OFFSET_REL))
                self._scale_bar.size = t_span_ms
                self._scale_bar.text.setText(f"{t_span_ms} ms")
                self._scale_bar.updateBar()
        except Exception:
            pass
