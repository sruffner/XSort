from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QSlider, QVBoxLayout, QLabel, QHBoxLayout, QFrame
import pyqtgraph as pg

from xsort.data.analyzer import Analyzer
from xsort.data.neuron import Neuron, ChannelTraceSegment, MAX_CHANNEL_TRACES
from xsort.views.baseview import BaseView


class ChannelView(BaseView):
    """
    This view displays a 1-sec trace for each of up to MAX_CHANNEL_TRACES channels in the displayable channel set. For
    each neuron currently selected for display in XSort, the view superimposes -- on the trace for the neuron's
    'primary channel' -- 10-ms clips indicating the occurrence of spikes on that neuron.

        A slider at the bottom of the view lets the user choose any 1-sec segment over the entire EPhys recording. The
    companion readouts reflect the elapsed recording time -- in the format MM:SS.mmm (minutes, seconds, milliseconds) --
    at the start and end of the currently visible portion of the traces. Using the mouse scroll wheel, the user can
    zoom in on the plotted traces both horizontally in time and vertically in voltage. The plot's x- and y-axis range
    limits are configured so the user can zoom in on ony 100ms portion of the 1-second segment, and on any 2 adjacent
    channel traces.

        Our strategy is to pre-create a list of MAX_CHANNEL_TRACES (16) :class:`pyqtgraph.PlotDataItem`s representing
    the channel trace segments (:class:`ChannelTraceSegment`) for each of the analog channels in the current displayable
    chanel set. Similarly, we pre-create a list of Analyzer.MAX_FOCUS_NEURONS (3) data items that render the 10-ms spike
    clips for neurons in the display focus list. All of the plot data items will have empty data sets initially, and it
    is only the data sets that get updated when channel trace segments or neural metrics are retrieved from XSort's
    internal cache, or when the neuron focus list changes in any way.
    """

    spike_clip_pen_width: int = 1
    """ 
    Width of pen used to render spike clips on top of channel traces. IMPORTANT - Any value other than 1
    dramatically worsened rendering times!
    """
    _DEFAULT_TRACE_OFFSET: int = 200
    """ Default vertical separation between channel trace segments, in microvolts. """
    _MIN_TRACE_OFFSET: int = 100
    """ Minimum vertical separation between channel trace segments, in microvolts. """
    _MAX_TRACE_OFFSET: int = 2000
    """ Minimum vertical separation between channel trace segments, in microvolts. """

    def __init__(self, data_manager: Analyzer) -> None:
        super().__init__('Channels', None, data_manager)

        self._plot_widget = pg.PlotWidget()
        """ This plot widget fills the entire view (minus border) and contains all plotted channel traces. """
        self._plot_item: pg.PlotItem = self._plot_widget.getPlotItem()
        """ The graphics item within the plot widget that manages the plotting. """
        self._message_label: pg.LabelItem = pg.LabelItem("", size="24pt", color="#808080")
        """ This label is displayed centrally when channel data is not available. """
        self._trace_offset: int = self._DEFAULT_TRACE_OFFSET
        """ Vertical separation between channel trace segments in the view, in microvolts. """
        self._trace_start: int = 0
        """ Sample index, relative to start of recording, for the first sample in each trace segment. """
        self._trace_pdis: List[pg.PlotDataItem] = list()
        """ 
        List of plot data items rendering channel trace segments for up to MAX_CHANNEL_TRACES analog channels.
        """
        self._spike_clip_pdis: List[pg.PlotDataItem] = list()
        """ 
        The plot data item at pos N renders the spike clips for the N-th neuron in the list of neurons with the display
        focus. The 10ms clips **[T-1 .. T_9]** are plotted on top of the trace segment for the unit's primary analog 
        source channel at each spike occurrence time T that falls within that trace segment. The clip color matches
        the color assigned to each neuron in the display list, albeit semi-transparent to help visualize overlaps among
        units having the same primary channel.
        """
        self._t0_slider = QSlider(orientation=Qt.Orientation.Horizontal)
        """ Slider controls the elapsed start time (in whole seconds) for the channel trace segments on display. """
        self._t0_readout = QLabel()
        """ 
        Readout displays current elapsed recording time at the left side of the plot, in MM:SS:mmm format. It reflects
        the current X-axis range of the plot, which changes as user scales/pans the plot.
        """
        self._t1_readout = QLabel()
        """ 
        Readout displays current elapsed recording time at the right side of the plot, in MM:SS:mmm format. It reflects
        the current X-axis range of the plot, which changes as user scales/pans the plot.
        """
        self._vspan_slider = QSlider(orientation=Qt.Orientation.Horizontal)
        """ Slider controls vertical separation between channel traces in microvolts. """
        self._vspan_readout = QLabel(f"+/-{int(self._trace_offset / 2)} \u00b5v")
        """ Readout display +/- range for each channel trace, in microvolts (same for all). """

        # one-time only configuration of the plot: disable the context menu, don't draw y-axis line; position a message
        # label centered over the plot; x-axis is hidden; use a scale bar instead. initially: plot is empty with no axes
        # and message label indicates that no channel data is available.
        self._plot_item.setMenuEnabled(enableMenu=False)
        self._plot_item.sigXRangeChanged.connect(self.on_x_range_changed)
        left_axis: pg.AxisItem = self._plot_item.getAxis('left')
        left_axis.setPen(pg.mkPen(None))
        font: QFont = left_axis.label.font()
        font.setBold(True)
        left_axis.setStyle(tickFont=font, tickTextOffset=8)
        left_axis.setTextPen(pg.mkPen('w'))

        self._plot_item.hideAxis('bottom')

        self._message_label.setParentItem(self._plot_item)
        self._message_label.anchor(itemPos=(0.5, 0.5), parentPos=(0.5, 0.5))
        scale_bar = pg.ScaleBar(size=0.05, suffix='s')
        scale_bar.setParentItem(self._plot_item.getViewBox())
        scale_bar.anchor(itemPos=(1, 1), parentPos=(1, 1), offset=(-10, -20))

        self._t0_readout.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        self._t0_readout.setFrameStyle(QFrame.Shadow.Sunken | QFrame.Shape.Panel)
        self._t1_readout.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self._t1_readout.setFrameStyle(QFrame.Shadow.Sunken | QFrame.Shape.Panel)
        self._t0_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self._t0_slider.setTracking(False)
        self._t0_slider.valueChanged.connect(self.on_t0_slider_value_changed)
        self._t0_slider.sliderMoved.connect(self._on_t0_slider_moved)

        self._vspan_slider.setRange(self._MIN_TRACE_OFFSET, self._MAX_TRACE_OFFSET)
        self._vspan_slider.setSliderPosition(self._trace_offset)
        self._vspan_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self._vspan_slider.setTracking(False)
        self._vspan_slider.valueChanged.connect(self._on_vspan_slider_changed)
        self._vspan_slider.sliderMoved.connect(lambda x: self._vspan_readout.setText(f"+/-{int(x/2)} \u00b5v"))

        self._reset()

        main_layout = QVBoxLayout()
        main_layout.addWidget(self._plot_widget)

        control_line = QHBoxLayout()
        control_line.addWidget(self._t0_readout)
        control_line.addWidget(self._t0_slider, stretch=4)
        control_line.addWidget(self._t1_readout)
        control_line.addStretch(1)
        control_line.addWidget(self._vspan_slider, stretch=2)
        control_line.addWidget(self._vspan_readout)
        main_layout.addLayout(control_line)

        self.view_container.setLayout(main_layout)

    def _reset(self) -> None:
        """
        Reset and reinitialize this view.

        On first call, all plot data items (PDI) rendering up to MAX_CHANNEL_TRACES analog channel trace segments are
        created, initially with empty data sets. Additional empty PDIs are created to render the spike clips for up to
        Analyzer.MAX_FOCUS_NEURONS in the display/focus list. The trace color for each spike clip PDI matches the
        display color assigned to the corresponding position in the focus list.

        On all future calls, the view is reset IAW the number of displayable analog channels.
         - The data set for each trace segment PDI is set to the corresponding channel trace segment, a "flat line"
           trace if that trace segment is not yet available, or an empty set if that PDI will not be used because there
           are fewer than MAX_CHANNEL_TRACES displayable channels. The traces are offset vertically IAW the view's
           current trace offset.
         - The data set for every spike clip PDI is refreshed if it is available, else it is cleared if necessary.
         - If there are no displayable analog channels, the Y-axis is hidden and a static centered message label is
           shown indicating that there is no data. Otherwise, the Y-axis tick labels are updated to reflect the analog
           channel indices of the corresponding "flat line" traces.
         - The slider and label widgets that set and display the elapsed start time for the trace segments are reset
           for a start time of 0 and configured IAW the analog channel duration (if known).
        """
        # one-time only: create all the trace segment and spike clip plot data items. The latter PDIs are configured so
        # that NaN introduces gaps between the individual clips, with a semi-transparent version of the color assigned
        # to each slot in the unit focus list.
        if len(self._trace_pdis) == 0:
            for _ in range(MAX_CHANNEL_TRACES):
                self._trace_pdis.append(self._plot_item.plot(x=[], y=[]))
            for k in range(Analyzer.MAX_NUM_FOCUS_NEURONS):
                color = QColor.fromString(Analyzer.FOCUS_NEURON_COLORS[k])
                color.setAlpha(128)
                pen = pg.mkPen(color, width=self.spike_clip_pen_width)
                self._spike_clip_pdis.append(self._plot_item.plot(x=[], y=[], name=f"spike-clips-{k}", pen=pen,
                                                                  connect='finite'))

        n_traces = len(self.data_manager.channel_indices)
        if n_traces == 0:
            self._plot_item.hideAxis('left')
            self._message_label.setText("No channels available")
        else:
            self._message_label.setText("")
            self._plot_item.showAxis('left')

        self._refresh(traces=True, clips=True, vspan_changed=True, reset_zoom=True)

        self._t0_readout.setText(ChannelView.digital_readout(0))
        self._t1_readout.setText(ChannelView.digital_readout(1))
        dur = int(self.data_manager.channel_recording_duration_seconds)
        if dur == 0:
            self._t0_slider.setRange(0, 10)
            self._t0_slider.setSliderPosition(0)
            self._t0_slider.setTickInterval(10)
            self._t0_slider.setEnabled(False)
        else:
            self._t0_slider.setRange(0, dur)
            self._t0_slider.setSliderPosition(0)
            self._t0_slider.setTickInterval(dur)
            self._t0_slider.setEnabled(True)

    def _refresh(self, traces: bool = False, clips: bool = False, vspan_changed: bool = False,
                 reset_zoom: bool = False) -> None:
        """
        Helper method optionally refreshes various aspects of this view.

        :param traces: If True, the data sets for all channel trace plot items are refreshed.
        :param clips: If True, the data sets for all spike clip plot items are refreshed.
        :param vspan_changed: If True, the tick labels on the Y-axis and the plot's X/Y range are refreshed IAW a
            change in the vertical range. (The tick labels identify the channel indices for the traces displayed.)
        :param reset_zoom: If True, resets the plot's zoom and pan state.
        :return:
        """
        ch_indices = self.data_manager.channel_indices
        if len(ch_indices) == 0:
            return

        if traces:
            offset = 0
            for k, pdi in enumerate(self._trace_pdis):
                if k >= len(ch_indices):
                    if not (pdi.xData is None):
                        pdi.setData(x=[], y=[])
                else:
                    ch_idx = ch_indices[k]
                    seg = self.data_manager.channel_trace(ch_idx)
                    if seg is None:
                        pdi.setData(x=[0, 1], y=[offset, offset])
                    else:
                        pdi.setData(x=np.linspace(start=0, stop=seg.duration, num=seg.length),
                                    y=offset + seg.trace_in_microvolts)
                    offset += self._trace_offset

        if clips:
            displayed = self.data_manager.neurons_with_display_focus
            for k, pdi in enumerate(self._spike_clip_pdis):
                x = []
                y = []
                if k < len(displayed):
                    u = displayed[k]
                    # NOTE: Set of 16 displayable channels is governed by primary channel of the primary neuron. The
                    # primary channel of other units in the focus list may not be in the set of displayable channels!
                    if u.primary_channel in ch_indices:
                        offset = self._trace_offset * ch_indices.index(u.primary_channel)
                        seg = self.data_manager.channel_trace(u.primary_channel)
                        if seg is not None:
                            x, y = self._prepare_clip_data(u, seg, offset)
                # update PDI data set if we got clip data, OR there's no clip data and the PDI has a non-empty data set
                if (len(x) > 0) or not (pdi.xData is None):
                    pdi.setData(x=x, y=y)

        if vspan_changed:
            ticks = []
            offset = 0
            for ch_idx in ch_indices:
                ticks.append((offset, str(ch_idx)))
                offset += self._trace_offset
            self._plot_item.getAxis('left').setTicks([ticks])

            self._plot_item.getViewBox().setLimits(
                xMin=0, xMax=1, minXRange=0.1, maxXRange=1, yMin=-self._trace_offset, yMax=offset,
                minYRange=2 * self._trace_offset, maxYRange=offset + self._trace_offset)

        if reset_zoom:
            self._plot_item.getViewBox().enableAutoRange()  # resets the zoom (if user previously zoomed in on view)

    def on_working_directory_changed(self) -> None:
        """
        Whenever the working directory has changed, the unit focus list will be empty and the initial set of analog data
        channel traces should be ready. The view is reset accordingly.
        """
        if self._auto_adjust_trace_offset():
            self._vspan_readout.setText(f"+/-{int(self._trace_offset / 2)} \u00b5v")
        self._reset()

    def on_channel_traces_updated(self) -> None:
        """
        Handler updates the X,Y data for each :class:`pyqtgraph.PlotDataItem` that renders a trace segment or a set
        of spike clips for a neuron in the current focus list.
        """
        # auto adjust trace offset IAW worst-case peak-to-peak amplitude across channel traces
        if self._auto_adjust_trace_offset():
            self._on_vspan_slider_changed(self._trace_offset)
            self._plot_item.getViewBox().enableAutoRange()  # reset zoom
        else:
            self._refresh(traces=True, clips=True)

    def _auto_adjust_trace_offset(self) -> bool:
        """
        Helper method adjusts the current trace offset to accommodate worst-case peak-to-peak amplitude across
        channel traces. The trace offset slider position is updated, but otherwise the view is not refreshed.
        :return: True if trace offset was adjusted; else False.
        """
        max_amp = 0
        for ch_idx in self.data_manager.channel_indices:
            seg = self.data_manager.channel_trace(ch_idx)
            if seg:
                max_amp = max(max_amp, seg.amplitude_in_microvolts)
        ofs_auto = self._trace_offset if max_amp == 0 else max(self._MIN_TRACE_OFFSET, min(int(max_amp),
                                                                                           self._MAX_TRACE_OFFSET))
        if ofs_auto != self._trace_offset:
            self._vspan_slider.setSliderPosition(ofs_auto)
            self._trace_offset = ofs_auto
            return True
        return False

    def on_neuron_metrics_updated(self, uid: str) -> None:
        """
        When the metrics of a neural unit are updated, this handler refreshes the spike clips for that unit IF it is
        in the current neuron display list.
        """
        ch_indices = self.data_manager.channel_indices
        for k, unit in enumerate(self.data_manager.neurons_with_display_focus):
            if unit.uid == uid:
                spike_clip_pdi = self._spike_clip_pdis[k]
                seg: Optional[ChannelTraceSegment] = None
                offset = 0
                if unit.primary_channel in ch_indices:
                    seg = self.data_manager.channel_trace(unit.primary_channel)
                    offset = self._trace_offset * ch_indices.index(unit.primary_channel)
                if seg is None:
                    if not (spike_clip_pdi.xData is None):
                        spike_clip_pdi.setData(x=[], y=[])
                else:
                    x, y = self._prepare_clip_data(unit, seg, offset)
                    spike_clip_pdi.setData(x=x, y=y)
                break

    def on_focus_neurons_changed(self, channels_changed: True) -> None:
        """
        When the neuron display list changes, the set of displayable channels could change -- in which case this view
        is reset. If the displayable channel set is unchanged, then only the spike clip data is refreshed for **all**
        slots in the display list. For unused slots, the clip data will be empty.
        """
        if channels_changed:
            self._reset()
        else:
            self._refresh(clips=True)

    def on_channel_trace_segment_start_changed(self) -> None:
        """
        When the channel trace segment start time changes, "flat line" all channel trace segment data items and empty
        all spike clip data items, since the new channel trace segments will not be immediately available. Also update
        the digital readouts to reflect the new segment starting and ending times.
        """
        self._refresh(traces=True, clips=True)

        t0 = self.data_manager.channel_trace_seg_start
        self._t0_slider.setSliderPosition(t0)
        rng = self._plot_item.getViewBox().viewRange()
        self._t0_readout.setText(ChannelView.digital_readout(t0 + rng[0][0]))
        self._t1_readout.setText(ChannelView.digital_readout(t0 + rng[0][1]))

    def _prepare_clip_data(self, u: Neuron, seg: ChannelTraceSegment, offset: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Produce Numpy arrays (X, Y) defining a plot that traces over the specified analog trace segment for each 10-ms
        spike clip of the specified neuron that fully or partially overlaps the time interval covered by the trace.
            If no spikes overlap the trace segment's time interval, empty arrays are returned for both X and Y. Else,
        each array will include a single NaN value separating one spike clip from the next -- so that only the spike
        clips are rendered by the :class:`pyqtgraph.PlotDataItem` to which (X, Y) arrays are assigned.

        :param u: The neural unit.
        :param seg: The analog channel trace segment.
        :param offset: The vertical offset at which the trace segment is rendered in the view.
        :return: A tuple (x, y) holding the 1D Numpy arrays specifying the spike clip data for the specified neuron.
        """
        x = np.array([])
        y = np.array([])
        if u is None or seg is None:
            return x, y
        spk_indices = np.intersect1d(np.argwhere(u.spike_times < seg.t0 + 1.001),
                                     np.argwhere(u.spike_times > seg.t0 - 0.009))
        spk_times = [u.spike_times[k] for k in spk_indices]
        for t in spk_times:
            clip_start, clip_end = max(seg.t0, t - 0.001) - seg.t0, min(seg.t0 + 1, t + 0.009) - seg.t0
            ticks = int(self.data_manager.channel_samples_per_sec * (clip_end - clip_start))
            clip_start_ticks = int(self.data_manager.channel_samples_per_sec * clip_start)
            x = np.concatenate((x, np.linspace(clip_start, clip_end, num=ticks), np.array([clip_end])))
            y = np.concatenate((y, offset + seg.trace_in_microvolts[clip_start_ticks:clip_start_ticks+ticks],
                                np.array([np.NaN])))
        return x, y

    def _on_t0_slider_moved(self, t0: int) -> None:
        """
        Handler updates the elapsed time readouts while the user is dragging the slider button. The trace segment start
        time is not changed until the slider button is released -- see :func:`on_t0_slider_value_changed()`.

        :param t0: The current slider position = the elapsed recording time in seconds.
        """
        rng = self._plot_item.getViewBox().viewRange()
        self._t0_readout.setText(ChannelView.digital_readout(t0 + rng[0][0]))
        self._t1_readout.setText(ChannelView.digital_readout(t0 + rng[0][1]))

    @Slot(int)
    def on_t0_slider_value_changed(self, t0: int) -> None:
        """
        Handler called when the slider controlling the channel trace segment start time is released. The data manager is
        informed of the change in the segment start time, which will trigger a background task to retrieve the
        channel trace segments for all analog source channels.
        :param t0: The new slider position = the elapsed recording time in seconds.
        """
        self.data_manager.set_channel_trace_seg_start(t0)

    @Slot(object, object)
    def on_x_range_changed(self, _src: object, rng: Tuple[float, float]) -> None:
        """
        Handler updates the labels reflecting the elapsed recording times at the start and end of the plot's X-axis.
        These change as the user scales and pans the plot content.

        :param _src: The plot view box for which the X-axis range changed. Unused.
        :param rng: The new X-axis range (xMin, xMax).
        """
        t0 = self.data_manager.channel_trace_seg_start
        t1 = t0 + 1
        if isinstance(rng, tuple) and len(rng) == 2 and isinstance(rng[0], float):
            t0 = self.data_manager.channel_trace_seg_start + rng[0]
            t1 = self.data_manager.channel_trace_seg_start + rng[1]
        self._t0_readout.setText(ChannelView.digital_readout(t0))
        self._t1_readout.setText(ChannelView.digital_readout(t1))

    @staticmethod
    def digital_readout(t: float, with_msecs: bool = True) -> str:
        t = max(0.0, t)
        minutes = int(t / 60)
        seconds = int(t - 60 * int(t/60))
        msecs = int(1000 * (t - int(t)))
        return f"{minutes:02d}:{seconds:02d}.{msecs:03d}" if with_msecs else f"{minutes:02d}:{seconds:02d}"

    @Slot(int)
    def _on_vspan_slider_changed(self, ofs: int) -> None:
        """
        Handler called when the slider controlling the separation between channel traces is released. It updates the
        corresponding readout label, lays out all traces (and any unit clips if the neural unit display list is not
        empty), and fixes the Y-axis limits of the view IAW the new trace offset.

        Since the user may have zoomed in on the view, an effort is made to adjust the Y-axis visible range so that
        the channel traces don't shift up/down as a result of the change in separation

        :param ofs: The new slider position = the offset between consecutive channel traces in microvolts.
        """
        # compute current visible Y range as a fraction of the previous limits
        n_traces = len(self.data_manager.channel_indices)
        y_rng = self._plot_item.getViewBox().viewRange()[1]
        y_span = 2 * self._trace_offset + (n_traces - 1) * self._trace_offset
        y_min_frac = (y_rng[0] + self._trace_offset) / y_span
        y_max_frac = (y_rng[1] + self._trace_offset) / y_span

        self._trace_offset = ofs
        self._vspan_readout.setText(f"+/-{int(ofs/2)} \u00b5v")

        # refresh current traces and spike clips IAW change in vertical range
        self._refresh(traces=True, clips=True, vspan_changed=True)

        # adjust visible Y range so that the same traces are in view (in case user has zoomed in on view)
        y_span = 2 * self._trace_offset + (n_traces - 1) * self._trace_offset
        y_min = y_span * y_min_frac - self._trace_offset
        y_max = y_span * y_max_frac - self._trace_offset
        self._plot_item.getViewBox().setYRange(y_min, y_max)
