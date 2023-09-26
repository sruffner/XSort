from typing import Dict

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QSlider, QVBoxLayout, QLabel, QHBoxLayout, QFrame
import pyqtgraph as pg

from xsort.data.analyzer import Analyzer
from xsort.views.baseview import BaseView


_DEFAULT_TRACE_OFFSET: int = 200
""" Default vertical separation between channel trace segments, in microvolts. """


class ChannelView(BaseView):
    """ TODO: UNDER DEV """

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
        self._t0_slider = QSlider(orientation=Qt.Orientation.Horizontal)
        """ Slider controls the elapsed start time for the channel trace segments (segment duration fixed at 1s). """
        self._t0_readout = QLabel()
        """ Readout displays current elapsed start time for the channel trace segments, in MM:SS format. """

        # one-time only configuration of the plot: disable the context menu, don't draw y-axis line; position message
        # label centered over the plot (to indicate when no channel data is available); x-axis is hidden; use a scale
        # bar instead
        # initially: plot is empty with no axes and message label indicating that no channel data is available
        self._plot_item.setMenuEnabled(enableMenu=False)
        self._plot_item.getAxis('left').setPen(pg.mkPen(None))
        self._plot_item.getAxis('left').setStyle(tickTextOffset=5)
        self._plot_item.hideAxis('bottom')
        self._message_label.setParentItem(self._plot_item)
        self._message_label.anchor(itemPos=(0.5, 0.5), parentPos=(0.5, 0.5))
        scale_bar = pg.ScaleBar(size=0.1, suffix='s')
        scale_bar.setParentItem(self._plot_item.getViewBox())
        scale_bar.anchor(itemPos=(1, 1), parentPos=(1, 1), offset=(-20, -10))

        self._t0_readout.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        self._t0_readout.setFrameStyle(QFrame.Shadow.Sunken | QFrame.Shape.Panel)
        self._t0_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._t0_slider.valueChanged.connect(self.on_t0_slider_value_changed)
        self._t0_slider.sliderReleased.connect(self.on_t0_slider_released)
        self._reset()

        main_layout = QVBoxLayout()
        main_layout.addWidget(self._plot_widget)
        control_line = QHBoxLayout()
        control_line.addWidget(self._t0_readout)
        control_line.addWidget(self._t0_slider)
        main_layout.addLayout(control_line)
        self.view_container.setLayout(main_layout)

    def _reset(self) -> None:
        """
        Reset and reinitialize this view. All plot data items representing existing channel trace segments are removed
        and then repopulated. If there are no analog channels available, the Y-axis is hidden and a static centered
        message label is shown indicating that there is no data. Otherwise, a "flat line" trace is drawn in [0, 1] for
        each available analog channel, and the Y-axis tick labels reflect the channel indices. The traces are separated
        vertically IAW the view's current trace offset.

        The slider and label widgets that set and display the elapsed start time for the trace segments are reset for
        reset for a start time of 0 and configured IAW the analog channel duration (if known).
        """
        self._plot_item.clear()  # this does not remove _message_label, which is in the internal viewbox
        self._trace_segments.clear()
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

        self._t0_readout.setText("00:00")
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
        each channels will not be ready. Here we initialize the plot to show all available channels as "flat lines"
        spanning one second.
        """
        self._reset()

    def on_channel_trace_segment_updated(self, idx: int) -> None:
        offset = 0
        for k, pdi in self._trace_segments.items():
            if k == idx:
                seg = self.data_manager.channel_trace(idx)
                if seg is None:
                    pdi.setData(x=[0, 1], y=[offset, offset])
                else:
                    pdi.setData(x=np.linspace(start=seg.t0, stop=seg.t0+seg.duration, num=seg.length),
                                y=offset + seg.trace_in_microvolts)
                break
            else:
                offset += self._trace_offset

    @Slot(int)
    def on_t0_slider_value_changed(self, value: int) -> None:
        minutes = int(value/60)
        seconds = value % 60
        self._t0_readout.setText(f"{minutes:02d}:{seconds:02d}")

    @Slot()
    def on_t0_slider_released(self) -> None:
        t0 = self._t0_slider.sliderPosition()
        if self.data_manager.set_channel_trace_seg_start(t0):
            offset = 0
            for k, pdi in self._trace_segments.items():
                seg = self.data_manager.channel_trace(k)
                if seg is None:
                    pdi.setData(x=[0, 1], y=[offset, offset])
                else:
                    pdi.setData(x=np.linspace(start=seg.t0, stop=seg.t0 + seg.duration, num=seg.length),
                                y=offset + seg.trace_in_microvolts)
                offset += self._trace_offset
