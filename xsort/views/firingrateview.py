from typing import List, Optional

from PySide6.QtCore import Qt, Slot, QSettings, QEvent, QObject
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QVBoxLayout, QComboBox, QHBoxLayout, QCheckBox
import pyqtgraph as pg
import numpy as np

from xsort.data.analyzer import Analyzer
from xsort.data.neuron import Neuron
from xsort.views.baseview import BaseView
from xsort.views.channelview import ChannelView


class FiringRateView(BaseView):
    """
    This simple view plots the variation in firing rate of each unit in the current display/focus list over the
    course of the electrophysiological recording.
        As with many of the views rendering plots for neurons in the display list, a plot item is pre-created to render
    the firing rate histogram for the unit in each possible "slot" in the display list. Initially, these plot items
    contain empty (X,Y) data arrays. The view is "refreshed" by updating those data arrays IAW the current state --
    including the selected bin size, whether or not the histogram is normalized, and what neurons currently occupy the
    display list.
        The firing rate histogram is computed on the fly via :method:`firing_rate_histogram()` in :class:`Neuron`. The
    computation is fast, so it does not impact GUI responsiveness.
    """

    _BIN_SIZE_CHOICES: List[int] = [20, 30, 60, 90, 120, 300]
    """ Choice list for firing rate histogram bin size, in seconds. """
    _DEF_BIN_SIZE: int = 60
    """ Default bin size for firing rate histograms in seconds. """
    _BIN_SIZE_STEP: int = 20
    """ Bin step size in seconds. """
    _PEN_WIDTH: int = 3
    """ Width of pen used to draw the firing rate histograms. """

    def __init__(self, data_manager: Analyzer) -> None:
        super().__init__('Firing Rate', None, data_manager)

        self._plot_widget = pg.PlotWidget()
        """ The firing rate histograms for all neurons in the current display list are rendered in this widget. """
        self._plot_item: pg.PlotItem = self._plot_widget.getPlotItem()
        """ The graphics item that manages plotting of the firing rate histograms. """
        self._histograms: List[pg.PlotDataItem] = list()
        """ 
        The plot data item at pos N renders the firing histogram for the N-th neuron in the display focus list. The 
        trace color matches the color assigned to each neuron in the display list.
        """
        self._bin_size_combo = QComboBox()
        """ Combo box selects the bin size for the firing rate histograms, in seconds. """
        self._normalized_cb = QCheckBox("Normalized")
        """ If checked, firing rate histograms are normalized by dividing by the unit's overall mean firing rate. """
        self._vline = pg.InfiniteLine(pos=-1000, angle=90, movable=False, pen=None, label="pos",
                                      labelOpts=dict(position=0.05))
        """ Vertical line follows mouse cursor when it is inside the plot view box. Initially placed outside box. """
        self._markers: List[pg.TargetItem] = list()
        """ 
        Labelled markers, one per histogram trace, displaying Y value where vertical line cursor intersects histogram.
        """

        # one-time configuration
        self._plot_item.setMenuEnabled(enableMenu=False)
        self._plot_item.getViewBox().setMouseEnabled(x=False, y=False)
        self._plot_item.hideButtons()
        self._plot_item.hideAxis('left')
        self._plot_item.hideAxis('bottom')

        # pre-create the plot data items for the maximum number of units in the display focus list
        for i in range(Analyzer.MAX_NUM_FOCUS_NEURONS):
            color = QColor.fromString(Analyzer.FOCUS_NEURON_COLORS[i])
            pen = pg.mkPen(color=color, width=self._PEN_WIDTH)
            self._histograms.append(self._plot_item.plot(x=[], y=[], pen=pen, stepMode='left'))

        # a vertical line follows mouse as it moves over the plot window, while a marker on each displayed histogram
        # is drawn at the intersection of histogram and vertical line and indicates the y-value at that intersection
        # Each marker and its label are rendered in the display color assigned to the corresponding histogram; the
        # label background is black so we can see the label on top of the histogram trace.
        self._plot_item.addItem(self._vline, ignoreBounds=True)
        for i in range(Analyzer.MAX_NUM_FOCUS_NEURONS):
            color = Analyzer.FOCUS_NEURON_COLORS[i]
            # careful: if label string is empty, the internal TargetLabel will be None, which causes problems when
            # we update the label!
            marker = pg.TargetItem(symbol='o', brush=color, pen=color, movable=False, label=" ",
                                   labelOpts=dict(color=color, fill='k', border=color))
            self._markers.append(marker)
            self._plot_item.addItem(marker, ignoreBounds=True)
        self._plot_item.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # need to detect when mouse enters/leaves plot widget so we can hide the vertical line cursor
        self._plot_widget.installEventFilter(self)

        # set up the combo box that selects bin size (note we have to convert to/from str
        self._bin_size_combo.addItems([str(k) for k in self._BIN_SIZE_CHOICES])
        self._bin_size_combo.setCurrentText(str(self._DEF_BIN_SIZE))
        self._bin_size_combo.currentTextChanged.connect(self._on_bin_size_changed)

        self._normalized_cb.stateChanged.connect(self._on_normalization_changed)

        label = QLabel("Bin size (s):")
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self._plot_widget)
        control_line = QHBoxLayout()
        control_line.addWidget(self._normalized_cb)
        control_line.addStretch(1)
        control_line.addWidget(label)
        control_line.addWidget(self._bin_size_combo)
        main_layout.addLayout(control_line)

        self.view_container.setLayout(main_layout)

    def _refresh(self) -> None:
        """
        Refresh the firing rate histogram data rendered in this view in response to a change in the set of displayed
        neurons, or a change in the histogram options -- bin size and normalization.
            This method simply updates the data item corresponding to the firing rate histogram for the neuron in each
        'slot' in the display list. When a display slot is unused, the corresponding data items contain empty arrays
        and thus renders nothing.
            The firing rate histograms are computed on the fly in the :class:`Neuron` instance. However, this
        computation is fast and should not impact GUI responiveness.
        """
        displayed = self.data_manager.neurons_with_display_focus
        bin_size = int(self._bin_size_combo.currentText())
        normalized = self._normalized_cb.isChecked()
        for k in range(Analyzer.MAX_NUM_FOCUS_NEURONS):
            unit: Optional[Neuron] = displayed[k] if k < len(displayed) else None
            if unit is None:
                self._histograms[k].setData(x=[], y=[])
            else:
                fr_hist = unit.firing_rate_histogram(
                    bin_size, self.data_manager.channel_recording_duration_seconds, normalized)
                n = len(fr_hist)
                self._histograms[k].setData(
                    x=np.linspace(start=bin_size/2, stop=bin_size*n, num=n), y=fr_hist)

    def on_working_directory_changed(self) -> None:
        self._refresh()

    def on_focus_neurons_changed(self) -> None:
        self._refresh()

    @Slot(str)
    def _on_bin_size_changed(self, _: str) -> None:
        """ Handler refreshes the plotted firing rate histograms whenever the user changes the bin size. """
        self._refresh()

    @Slot(Qt.CheckState)
    def _on_normalization_changed(self, _: Qt.CheckState) -> None:
        """ Handler refreshes the plotted firing rate histograms whenever user toggles normalilzation on/off. """
        self._refresh()

    @Slot(QEvent)
    def _on_mouse_moved(self, evt: QEvent) -> None:
        """
        Whenever the mouse moves over the plot window, update the position of the vertical line "time cursor" and its
        label indicating the elapsed time in the format "MM:SS". In addition, update the position and readout label for
        the target marker placed at the intersection of the time cursor and each visible firing rate histogram.
        :param evt: The mouse movement event.
        """
        normalized = self._normalized_cb.isChecked()
        pos = evt
        if self._plot_item.sceneBoundingRect().contains(pos):
            loc = self._plot_item.getViewBox().mapSceneToView(pos)
            self._vline.setPos(loc.x())
            self._vline.label.setFormat(ChannelView.digital_readout(loc.x(), with_msecs=False))
            marker: pg.TargetItem
            for i, marker in enumerate(self._markers):
                y = self._get_y_for_x(x=loc.x(), which=i)
                if y is None:
                    marker.setPos(-10000, 0)
                    marker.label().setFormat(" ")
                else:
                    marker.setPos(loc.x(), y)
                    marker.label().setFormat(f"y={y:.2f}" if normalized else f"y={int(y)}")

    def _get_y_for_x(self, x: float, which: int) -> Optional[float]:
        """
         Helper method finds the (approximate) Y coordinate value on a currently displayed firing rate histogram at
         a specified elapsed time.

        :param x: The elapsed time in the plot, in seconds.
        :param which: Ordinal position in the neuron display list selects the plot data item corresponding to the
            firing rate histogram for the unit. The data for the histogram will be empty or None if the histogram is
            not currently displayed.
        :return: The corresponding y-coordinate on the specified firing rate histogram, or None if undefined.
        """
        bin_size = int(self._bin_size_combo.currentText())
        y = None
        if 0 <= which < len(self._histograms):
            x_data, y_data = self._histograms[which].getData()
            idx = int(x / bin_size)
            if (y_data is not None) and (0 <= idx < len(y_data)):
                y = y_data[idx]
        return y

    def save_settings(self, settings: QSettings) -> None:
        """ Overridden to preserve the firing rate histogram bin size and normalization flag. """
        settings.setValue('firing_rate_view_bin_size', self._bin_size_combo.currentText())
        settings.setValue('firing_rate_view_norm', self._normalized_cb.isChecked())

    def restore_settings(self, settings: QSettings) -> None:
        """ Overridden to restore the histogram bin size and normalization flag from user settings. """
        try:
            bin_size_str = settings.value('firing_rate_view_bin_size')
            if int(bin_size_str) in self._BIN_SIZE_CHOICES:
                self._bin_size_combo.setCurrentText(bin_size_str)
            normalized: bool = (settings.value('firing_rate_view_norm', defaultValue="false") == "true")
            self._normalized_cb.setChecked(normalized)
            self._refresh()
        except Exception:
            pass

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """
        Overridden to detect when mouse enters and leaves the single plot widget within this view. The
        vertical-line time cursor and the histogram markers along that line are turned on/off when the mouse
        enters/leaves the plot.
        """
        if watched == self._plot_widget:
            if event.type() == QEvent.Type.Enter:
                self._vline.setPen(pg.mkPen(color='w', width=2))
            elif event.type() == QEvent.Type.Leave:
                self._vline.setPen(None)
                self._vline.label.setFormat("")
                for marker in self._markers:
                    marker.setPos(-10000, 0)
                    marker.label().setFormat(" ")
        return super().eventFilter(watched, event)
