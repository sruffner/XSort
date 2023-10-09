from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QSlider, QVBoxLayout, QLabel, QHBoxLayout, QFrame
import pyqtgraph as pg

from xsort.data.analyzer import Analyzer
from xsort.data.neuron import Neuron, ChannelTraceSegment
from xsort.views.baseview import BaseView


_DEFAULT_TRACE_OFFSET: int = 200
""" Default vertical separation between channel trace segments, in microvolts. """


class ChannelView(BaseView):
    """
    This view displays a 1-sec trace for each of the analog wideband and narrowband channels available from the Omniplex
    EPhys recording. For each neuron currently selected for display in XSort, the view superimposes -- on the trace for
    the neuron's 'primary channel' -- 10-ms clips indicating the occurrence of spikes on that neuron.

        A slider at the bottom of the view lets the user choose any 1-sec segment over the entire EPhys recording. The
    companion readouts reflect the elapsed recording time -- in the format MM:SS.mmm (minutes, seconds, milliseconds) --
    at the start and end of the currently visible portion of the traces. Using the mouse scroll wheel, the user can
    zoom in on the plotted traces both horizontally in time and vertically in voltage. The plot's x- and y-axis range
    limits are configured so the user can zoom in ony 100ms portion of the 1-second segment, and on any 2 adjacent
    channel traces.

        Our strategy is to pre-create a dictionary of :class:`pyqtgraph.PlotDataItem` representing the channel trace
    segments (:class:`ChannelTraceSegment`), indexed by the analog source channel index. Similarly, we pre-create a
    list of data items that render the 10-ms spike clips for neurons in the display focus list. All of the data items
    will have empty data sets initially, and it is only the data sets that get updated when channel trace segments or
    neural metrics are retrieved from XSort's internal cache, or when the neuron focus list changes in any way.
    """

    spike_clip_pen_width: int = 1
    """ 
    Width of pen used to render spike clips on top of channel traces. IMPORTANT - Any value other than 1
    dramatically worsened rendering times!
    """

    def __init__(self, data_manager: Analyzer) -> None:
        super().__init__('Channels', None, data_manager)

        self._plot_widget = pg.PlotWidget()
        """ This plot widget fills the entire view (minus border) and contains all plotted channel traces. """
        self._plot_item: pg.PlotItem = self._plot_widget.getPlotItem()
        """ The graphics item within the plot widget that manages the plotting. """
        self._message_label: pg.LabelItem = pg.LabelItem("", size="24pt", color="#808080")
        """ This label is displayed centrally when channel data is not available. """
        self._trace_offset: int = _DEFAULT_TRACE_OFFSET
        """ Vertical separation between channel trace segments in the view, in microvolts. """
        self._trace_start: int = 0
        """ Sample index, relative to start of recording, for the first sample in each trace segment. """
        self._trace_segments: Dict[int, pg.PlotDataItem] = dict()
        """ 
        The rendered trace segments, indexed by channel index. Each segment is assigned a name reflecting the
        Omniplex analog channel label -- 'WB<N>' or SPKC<N>' 
        """
        self._unit_spike_clips: List[pg.PlotDataItem] = list()
        """ 
        The plot data item at pos N renders the spike clips for the N-th neuron in the list of neurons with the display
        focus. The 10ms clips **[T-1 .. T_9]** are plotted on top of the trace segment for the unit's primary analog 
        source channel at each spike occurrence time T that fallw within that trace segment. The clip color matches
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
        self._t0_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._t0_slider.setTracking(False)
        self._t0_slider.valueChanged.connect(self.on_t0_slider_value_changed)
        self._t0_slider.sliderMoved.connect(self._on_t0_slider_moved)

        self._reset()

        main_layout = QVBoxLayout()
        main_layout.addWidget(self._plot_widget)
        control_line = QHBoxLayout()
        control_line.addWidget(self._t0_readout)
        control_line.addWidget(self._t0_slider)
        control_line.addWidget(self._t1_readout)
        main_layout.addLayout(control_line)
        self.view_container.setLayout(main_layout)

    def _reset(self) -> None:
        """
        Reset and reinitialize this view. All plot data items representing existing channel trace segments, as well as
        spike clips for each slot in the neuron display list, are removed and then repopulated. If there are no analog
        channels available, the Y-axis is hidden and a static centered message label is shown indicating that there is
        no data. Otherwise, a "flat line" trace is drawn in [0, 1] for each available analog channel, and the Y-axis
        tick labels reflect the channel indices. The traces are separated vertically IAW the view's current trace
        offset. The plot data items for the displayed neuron spike clips will all be empty initially.
            The slider and label widgets that set and display the elapsed start time for the trace segments are reset
        for a start time of 0 and configured IAW the analog channel duration (if known).
        """
        self._plot_item.clear()  # this does not remove _message_label, which is in the internal viewbox
        self._trace_segments.clear()
        self._unit_spike_clips.clear()
        if len(self.data_manager.channel_indices) == 0:
            self._plot_item.hideAxis('left')
            self._message_label.setText("No channels available")
        else:
            self._message_label.setText("")
            self._plot_item.showAxis('left')
            ticks = []
            offset = 0
            for idx in self.data_manager.channel_indices:
                self._trace_segments[idx] = self._plot_item.plot(x=[0, 1], y=[offset, offset],
                                                                 name=self.data_manager.channel_label(idx))
                ticks.append((offset, str(idx)))
                offset += self._trace_offset
            self._plot_item.getAxis('left').setTicks([ticks])

            self._plot_item.getViewBox().setLimits(
                xMin=0, xMax=1, minXRange=0.1, maxXRange=1, yMin=-self._trace_offset, yMax=offset,
                minYRange=2*self._trace_offset, maxYRange=max(self._trace_offset, offset+self._trace_offset))

            # initialize list of plot data items that render spike clips for neurons in display list. Configured so
            # that NaN introduces a gap in the trace, and configured with a semi-transparent version of the color
            # assigned to each slot in the display list.
            for k in range(Analyzer.MAX_NUM_FOCUS_NEURONS):
                color = QColor.fromString(Analyzer.FOCUS_NEURON_COLORS[k])
                color.setAlpha(128)
                pen = pg.mkPen(color, width=self.spike_clip_pen_width)
                self._unit_spike_clips.append(
                    self._plot_item.plot(x=[], y=[], name=f"spike-clips-{k}", pen=pen, connect='finite'))

        self._t0_readout.setText(ChannelView._digital_readout(0))
        self._t1_readout.setText(ChannelView._digital_readout(1))
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

    def on_working_directory_changed(self) -> None:
        """
        When the working directory changes, the number of analog data channels should be known, but analog data for
        each channel will not be ready, nor any neural unit metrics. The view is reset, preparing 'empty' data items
        for each channel trace segment and the spike clips for each slot in the neuron display list. The actual data
        will be inserted at a later time, as channel trace segments and neuron metrics are retrieved from XSort's
        internal cache.
        """
        self._reset()

    def on_channel_trace_segment_updated(self, idx: int) -> None:
        """
        When a channel trace segment is retrieved, this handler sets the X,Y data for the corresponding
        :class:`pyqtgraph.PlotDataItem` that renders the trace segment. If the specified channel is the so-called
        primary channel for any neuron in the current display list, the spike clips for that neuron are also updated.
        :param idx: Index of analog source channel for which the trace segment was retreived.
        """
        offset = 0
        for k, pdi in self._trace_segments.items():
            if k == idx:
                seg = self.data_manager.channel_trace(idx)
                if seg is None:
                    pdi.setData(x=[0, 1], y=[offset, offset])
                else:
                    pdi.setData(x=np.linspace(start=0, stop=seg.duration, num=seg.length),
                                y=offset + seg.trace_in_microvolts)
                break
            else:
                offset += self._trace_offset

        displayed = self.data_manager.neurons_with_display_focus
        for k, pdi in enumerate(self._unit_spike_clips):
            unit: Optional[Neuron] = displayed[k] if k < len(displayed) else None
            ch_idx = -1
            offset = 0
            if unit and isinstance(unit.primary_channel, int):
                try:
                    ch_idx = unit.primary_channel
                    offset = self._trace_offset * self.data_manager.channel_indices.index(ch_idx)
                except Exception:
                    continue
            if ch_idx == idx:
                seg = self.data_manager.channel_trace(ch_idx)
                if seg is None:
                    pdi.setData(x=[], y=[])
                else:
                    x, y = self._prepare_clip_data(unit, seg, offset)
                    pdi.setData(x=x, y=y)

    def on_neuron_metrics_updated(self, unit_label: str) -> None:
        """
        When the metrics of a neural unit are updated, this handler refreshes the spike clips for that unit IF it is
        in the current neuron display list.
        :param unit_label: Label identifying the neural unit for which metrics were updated.
        """
        displayed = self.data_manager.neurons_with_display_focus
        for k, pdi in enumerate(self._unit_spike_clips):
            if (k < len(displayed)) and (displayed[k].label == unit_label):
                u = displayed[k]
                offset = self._trace_offset * self.data_manager.channel_indices.index(u.primary_channel)
                seg = self.data_manager.channel_trace(u.primary_channel)
                if seg is None:
                    pdi.setData(x=[], y=[])
                else:
                    x, y = self._prepare_clip_data(u, seg, offset)
                    pdi.setData(x=x, y=y)
                break

    def on_focus_neurons_changed(self) -> None:
        """
        When the neuron display list changes, this handler refreshes the spike clip data for **all** slots in the
        display list. For unused slots, the clip data will be empty.
        """
        displayed = self.data_manager.neurons_with_display_focus
        for k, pdi in enumerate(self._unit_spike_clips):
            x = []
            y = []
            if k < len(displayed):
                u = displayed[k]
                offset = self._trace_offset * self.data_manager.channel_indices.index(u.primary_channel)
                seg = self.data_manager.channel_trace(u.primary_channel)
                if seg is not None:
                    x, y = self._prepare_clip_data(u, seg, offset)
            pdi.setData(x=x, y=y)

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
        self._t0_readout.setText(ChannelView._digital_readout(t0 + rng[0][0]))
        self._t1_readout.setText(ChannelView._digital_readout(t0 + rng[0][1]))

    @Slot(int)
    def on_t0_slider_value_changed(self, t0: int) -> None:
        """
        Handler updates the entire view when the trace segment start time is updated. First, the data manager is
        informed of the change in the segment start time, which will trigger a background task to retrieve the
        channel trace segments for all analog source channels. It then "flat lines" all channel trace segment data
        items and empties all spike clip data items, since the new channel trace segments will not be immediately
        available.
        :param t0: The new slider position = the elapsed recording time in seconds.
        """
        if self.data_manager.set_channel_trace_seg_start(t0):
            offset = 0
            for k, pdi in self._trace_segments.items():
                pdi.setData(x=[0, 1], y=[offset, offset])
                offset += self._trace_offset
            for pdi in self._unit_spike_clips:
                pdi.setData(x=[], y=[])
        rng = self._plot_item.getViewBox().viewRange()
        self._t0_readout.setText(ChannelView._digital_readout(t0+rng[0][0]))
        self._t1_readout.setText(ChannelView._digital_readout(t0+rng[0][1]))

    # noinspection PyUnusedLocal
    @Slot(object, object)
    def on_x_range_changed(self, src: object, rng: Tuple[float, float]) -> None:
        """
        Handler updates the labels reflecting the elapsed recording times at the start and end of the plot's X-axis.
        These change as the user scales and pans the plot content.

        :param src: The plot view box for which the X-axis range changed.
        :param rng: The new X-axis range (xMin, xMax).
        """
        t0 = self.data_manager.channel_trace_seg_start
        t1 = t0 + 1
        if isinstance(rng, tuple) and len(rng) == 2 and isinstance(rng[0], float):
            t0 = self.data_manager.channel_trace_seg_start + rng[0]
            t1 = self.data_manager.channel_trace_seg_start + rng[1]
        self._t0_readout.setText(ChannelView._digital_readout(t0))
        self._t1_readout.setText(ChannelView._digital_readout(t1))

    @staticmethod
    def _digital_readout(t: float) -> str:
        minutes = int(t / 60)
        seconds = int(t - 60 * int(t/60))
        msecs = int(1000 * (t - int(t)))
        return f"{minutes:02d}:{seconds:02d}.{msecs:03d}"
