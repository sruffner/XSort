from typing import List, Optional

import numpy as np
from PySide6.QtCore import Slot, QSettings
from PySide6.QtGui import QFont, QColor, Qt
from PySide6.QtWidgets import QHBoxLayout, QGraphicsTextItem, QSlider, QVBoxLayout, QLabel
import pyqtgraph as pg

from xsort.data.analyzer import Analyzer
from xsort.data.neuron import MAX_CHANNEL_TRACES, Neuron
from xsort.views.baseview import BaseView


class TemplateView(BaseView):
    """
    This view displays the per-channel spike template waveforms for all neurons currently selected for display in
    XSort. The displayed template span is user-configurable between 3-10ms. The per-channel templates are rendered in
    a single :class:`pyqtgraph.PlotWidget`, populated in order from the bottom to top and left to right by source
    channel index, with 4 templates per 'row'.
        Our strategy is to pre-create a list of 48 :class:`pyqtgraph.PlotDataItem` representing up to 16 per-channel
    templates for each of up to 3 neural units in the current display focus list.  All of the plot data items will have
    empty data sets initially,  and it is only the data sets that get updated when the display focus list changes or
    neuron metrics are updated.
        **NOTE**: When there are more than 16 recorded analog channels, XSort computes templates on a maximum of
    16 channels "near" a unit's primary channel. If two different units do not have the same primary channel, their
    templates are not computed on the same set of channels. By design, this view displays all 16 templates for the
    primary unit (the first unit in the focus list), as well any templates for other units in the focus list that were
    computed on any channel among the primary unit's 16 template channel indices. This makes sense -- it's highly
    unlikely the user will need to compare units with very different template channel sets.
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
    _MAX_VSPAN_UV: int = 1000
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
        self._channel_labels: List[pg.TextItem] = list()
        """ The text items that render the channel label in each of the up to MAX_CHANNEL_TRACES template plots. """
        self._template_pdis: List[pg.PlotDataItem] = list()
        """ 
        Plot data items rendering up to M*N template plots, where M=MAX_FOCUS_NEURONS and N=MAX_CHANNEL_TRACES. These
        are precreated with empty data sets. The first N correspond to the templates for the first unit in the current 
        display/focus list, and so on. Position within a bank of N serves as a index into the sorted list of displayable
        channel indices -- so we can get the unit template that is rendered by that particular data item.
        """
        self._tspan_slider = QSlider(orientation=Qt.Orientation.Horizontal)
        """ Slider controls the displayed span of each template in milliseconds. """
        self._vspan_slider = QSlider(orientation=Qt.Orientation.Horizontal)
        """ Slider controls the displayed height of each template in microvolts. """
        self._vspan_readout = QLabel(f"+/-{self._DEF_VSPAN_UV} \u00b5v")
        """ A label reflecting the current height of each template in microvolts. """

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

        self._reset()  # precreate all graphics items we need (text items and plot data items)

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
        Reset and reinitialize this view.

        On first call, all plot data items rendering up to 3*16 per-channel spike templates are created, initially with
        empty data sets (up to 16 per-channel spike templates for up to 3 units in the unit display/focus list). In
        addition, 16 text items are created that will display the analog channel index for each template plot. The trace
        color for each plot data item matches the display color assigned to the corresponding unit (based on the unit's
        position in the focus list).

        On all future calls, the data set for every plot data item is emptied (if it wasn't already), and the position
        and contents of each text item are updated IAW the current set of displayable channels and the current
        composition of the unit focus list. If there are no displayable analog channels, a static centered message label
        is shown indicating that there is no data. Otherwise the centered message label will indicate that template data
        is not yet available. The actual template data are set later, as the data is retrieved.
            All template plots are distributed across the view box from bottom to top and left to right, with four
        templates per row, in ascending order by source channel index. Channel labels are positioned near the end of the
        per-channel template waveforms for the primary unit.
        """
        # one-time only: create all the template plot data items and channel label text items
        if len(self._template_pdis) == 0:
            for unit_idx in range(Analyzer.MAX_NUM_FOCUS_NEURONS):
                pen_color = QColor.fromString(Analyzer.FOCUS_NEURON_COLORS[unit_idx])
                pen = pg.mkPen(pen_color, width=self.trace_pen_width)
                for template_idx in range(MAX_CHANNEL_TRACES):
                    self._template_pdis.append(self._plot_item.plot(x=[], y=[], pen=pen))
            for _ in range(MAX_CHANNEL_TRACES):
                label = pg.TextItem(text="", anchor=(1, 1))
                ti: QGraphicsTextItem = label.textItem
                font: QFont = ti.font()
                font.setBold(True)
                label.setFont(font)
                label.setPos(0, 0)
                self._plot_item.addItem(label)
                self._channel_labels.append(label)
            self._message_label.setText("No channel data available")
            return

        # all future calls: empty all data items and configure channel labels for the current displayable channel list
        for pdi in self._template_pdis:
            # only empty the PlotDataItem if it is not already empty!
            if not (pdi.xData is None):
                pdi.setData(x=[], y=[])

        displayable = self.data_manager.channel_indices

        if len(displayable) == 0:
            self._message_label.setText("No channel data available")
            for label in self._channel_labels:
                label.setText("")
        else:
            self._message_label.setText("No template data available")
            t_span_ms = self._tspan_slider.sliderPosition()
            v_span_uv = self._vspan_slider.sliderPosition()
            row: int = 0
            col: int = 0
            for pos in range(MAX_CHANNEL_TRACES):
                x = col * (t_span_ms * self._H_OFFSET_REL)
                y = row * (v_span_uv * 2)
                if pos < len(displayable):
                    self._channel_labels[pos].setText(str(displayable[pos]))
                    self._channel_labels[pos].setPos(x + t_span_ms, y + 5)
                    col = col + 1
                    if col == self._NUM_TEMPLATES_PER_ROW:
                        col = 0
                        row = row + 1
                else:
                    self._channel_labels[pos].setText("")
                    self._channel_labels[pos].setPos(0, 0)

            num_channels = len(displayable)
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

    def on_focus_neurons_changed(self, channels_changed: bool) -> None:
        """
        Whenever the subset of neural units holding the display focus changes, update the plot to show the spike
        templates for each unit in the focus set that across all displayable analog channels.
        """
        # if the set of displayable channels has changed, then we need to do a full reset
        if channels_changed:
            self._reset()
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
        Refresh the view in response to a change in the set of displayed neurons, a change in the template span, a
        user-initiated change in the voltage span, or upon retrieval of the metrics for a neuron.
            If the template span or voltage span has changed, the plot data items for all templates are updated
        accordingly, the channel labels in the plot are repositioned, and the size of the time scale bar is adjusted if
        necessary. If the set of displayed neurons has changed, all plot data items are updated accordingly. When a
        neuron's metrics (including spike template waveforms) are retrieved, no action is taken it that neuron is not
        currently selected for display. If it is, then only the plot data items associated with that unit's templates
        are refreshed. Note that only one of these changes will occur at a time.
            When the primary neuron is changed or its metrics updated, the current voltage span is auto-adjusted to
        one-half of that unit's peak-to-peak amplitude.

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
        displayed_units = self.data_manager.neurons_with_display_focus

        # if refresh is because a unit's metrics were updated, but that unit is not displayed, there's nothing to do.
        if isinstance(uid_updated, str) and not (uid_updated in [u.uid for u in displayed_units]):
            return

        # update all template plot data item and channel label text items in use.
        for unit_idx in range(Analyzer.MAX_NUM_FOCUS_NEURONS):
            u: Optional[Neuron] = displayed_units[unit_idx] if unit_idx < len(displayed_units) else None
            # when refresing because a unit's metrics were updated, only update the templates for that unit
            if isinstance(uid_updated, str) and isinstance(u, Neuron) and (u.uid != uid_updated):
                continue

            # if the focus list has changed or the metrics of the primary neuron were updated, auto-adjust voltage
            # span to 1/2 that unit's peak-to-peak amplitude
            if (unit_idx == 0) and (u is not None) and not (vspan_changed or tspan_changed):
                vspan_auto = int(u.amplitude / 2.0)
                vspan_auto = max(self._MIN_VSPAN_UV, min(vspan_auto, self._MAX_VSPAN_UV))
                if v_span_uv != vspan_auto:
                    self._vspan_slider.valueChanged.disconnect(self._on_voltage_range_change)
                    self._vspan_slider.setSliderPosition(vspan_auto)
                    self._vspan_readout.setText(f"+/-{vspan_auto} \u00b5v")
                    self._vspan_slider.valueChanged.connect(self._on_voltage_range_change)
                    v_span_uv = vspan_auto
                    vspan_changed = True  # to fix channel labels as a result of auto adjustment of voltage span

            row: int = 0
            col: int = 0
            for ch_pos, ch_idx in enumerate(self.data_manager.channel_indices):
                t0 = col * (t_span_ms * self._H_OFFSET_REL)
                y0 = row * (2 * v_span_uv)
                pdi = self._template_pdis[unit_idx*MAX_CHANNEL_TRACES + ch_pos]
                template = u.get_template_for_channel(ch_idx) if isinstance(u, Neuron) else None
                if template is None:
                    # only empty the PlotDataItem if it is not already empty!
                    if not (pdi.xData is None):
                        pdi.setData(x=[], y=[])
                else:
                    template = template[0:t_span_ticks]
                    pdi.setData(x=np.linspace(start=t0, stop=t0+t_span_ms, num=len(template)),
                                y=y0 + template)

                # fix corresponding channel label when template time or voltage span changes
                if (tspan_changed or vspan_changed) and (unit_idx == 0):
                    label = self._channel_labels[ch_pos]
                    label.setPos(t0 + t_span_ms, y0 + 5)

                col = col + 1
                if col == self._NUM_TEMPLATES_PER_ROW:
                    col = 0
                    row = row + 1

        # fix y-axis range IAW the # of template rows and the current voltage span
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

        self._message_label.setText("" if len(displayed_units) > 0 else "No units selected for display")

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
