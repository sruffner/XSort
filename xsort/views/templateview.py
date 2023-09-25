from typing import Dict

import numpy as np
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QHBoxLayout, QGraphicsTextItem
import pyqtgraph as pg

from xsort.data.analyzer import Analyzer
from xsort.views.baseview import BaseView

_NUM_TEMPLATES_PER_ROW: int = 4
_H_OFFSET_MS: int = 12
""" Horizontal offset between starts of adjacent templates in the same row, in milliseconds. """
_V_OFFSET_UV: int = 150
""" Vertical offset between the rows of templates, in microvolts. """


class TemplateView(BaseView):
    """
    TODO: UNDER DEV For now we're plotting all templates in one PlotWidget and spacing them out using the knowledge
        that each template is 10ms long and assuming that 150uV is adequate vertical spacing.
    """

    def __init__(self, data_manager: Analyzer) -> None:
        super().__init__('Templates', None, data_manager)

        self._plot_widget = pg.PlotWidget()
        """ This plot widget fills the entire view (minus border) and contains all plotted spike templates. """
        self._plot_item: pg.PlotItem = self._plot_widget.getPlotItem()
        """ The graphics item within the plot widget that manages the plotting. """
        self._message_label: pg.LabelItem = pg.LabelItem("", size="24pt", color="#808080")
        """ This label is displayed centrally when template data is not available. """
        self._spike_templates: Dict[int, pg.PlotDataItem] = dict()
        """ 
        A dictionary of the plotted spike templates, keyed by index of the analog source channel for each channel.
        """
        self._unit_shown: str = ""
        """ 
        Label of the neural unit for which spike templates are currently shown. If this is an empty string or
        does not identify an existing neural unit, then no templates are drawn.
        """

        # one-time only configuration of the plot: disable the context menu; hide axes; position message
        # label centered over the plot (to indicate when no data is available); use a scale bar to indicate timescale.
        # initially: plot is empty with the message label indicating that no data is available
        self._plot_item.setMenuEnabled(enableMenu=False)
        self._plot_item.hideAxis('left')
        self._plot_item.hideAxis('bottom')
        self._plot_item.setXRange(0, _NUM_TEMPLATES_PER_ROW*_H_OFFSET_MS)
        self._message_label.setParentItem(self._plot_item)
        self._message_label.anchor(itemPos=(0.5, 0.5), parentPos=(0.5, 0.5))
        scale_bar = pg.ScaleBar(size=10, suffix='ms')
        scale_bar.setParentItem(self._plot_item.getViewBox())
        scale_bar.anchor(itemPos=(1, 1), parentPos=(1, 1), offset=(-20, -10))

        main_layout = QHBoxLayout()
        main_layout.addWidget(self._plot_widget)
        self.view_container.setLayout(main_layout)

    def _reset(self) -> None:
        """
        Reset and reinitialize this view. All plot data items rendering per-channel spike templates are removed and then
        repopulated. If there are no analog channels, a static centered message label is shown indicating that there is
        no data. If the available analog channels are known, but the computed spike templates are not ready, the spike
        templates are each represented by "flat line" traces, and the centered message label remains. The traces are
        distributed across the view box from bottom to top and left to right, with 4 templates per row, in ascending
        order by source channel index.
        """
        self._plot_item.clear()  # this does not remove _message_label, which is in the internal viewbox
        self._spike_templates.clear()
        if len(self.data_manager.channel_indices) == 0:
            self._message_label.setText("No channel data available")
            self._unit_shown = ""
        else:
            self._unit_shown = ""
            self._message_label.setText("No template data available")
            row: int = 0
            col: int = 0
            for idx in self.data_manager.channel_indices:
                x = col * _H_OFFSET_MS
                y = row * _V_OFFSET_UV
                self._spike_templates[idx] = self._plot_item.plot(x=[x, x+10], y=[y, y],
                                                                  name=self.data_manager.channel_label(idx))
                # TODO: Trying out TextItem
                label = pg.TextItem(text=str(idx), anchor=(1, 1))
                ti: QGraphicsTextItem = label.textItem
                font: QFont = ti.font()
                font.setBold(True)
                label.setFont(font)
                label.setPos(x + _H_OFFSET_MS - 2, y + 5)
                self._plot_item.addItem(label)

                col = col + 1
                if col == _NUM_TEMPLATES_PER_ROW:
                    col = 0
                    row = row + 1
            num_channels = len(self.data_manager.channel_indices)
            num_rows = (int(num_channels/_NUM_TEMPLATES_PER_ROW) +
                        (0 if (num_channels % _NUM_TEMPLATES_PER_ROW == 0) else 1))
            self._plot_item.setYRange(-_V_OFFSET_UV/2, (num_rows - 0.5) * _V_OFFSET_UV)

    def on_working_directory_changed(self) -> None:
        """
        When the working directory changes, the number of analog data channels should be known, but neural unit spike
        template data will not be ready. Here we initialize the plot to show per-channel placeholder spike templates as
        "flat lines", each spanning 10 ms.
        """
        self._reset()

    def on_neuron_metrics_updated(self, unit_label: str) -> None:
        """
        If the unit metrics are updated for a neural unit currently displayed in this view, update the rendered spike
        templates for the unit across all available analog channels.

        :param unit_label: Label identifying the updated neural unit.
        """
        if not (self._unit_shown == unit_label):
            return
        self._message_label.setText("")
        for unit in self.data_manager.neurons:
            if unit.label == self._unit_shown:
                row: int = 0
                col: int = 0
                for idx in self.data_manager.channel_indices:
                    t0 = col * _H_OFFSET_MS
                    y0 = row * _V_OFFSET_UV
                    pdi = self._spike_templates[idx]
                    template = unit.get_template_for_channel(idx)
                    if template is None:
                        pdi.setData(x=[t0, t0+10], y=[y0, y0],)
                    else:
                        pdi.setData(x=np.linspace(start=t0, stop=t0 + 10, num=len(template)),
                                    y=y0 + template)
                    col = col + 1
                    if col == _NUM_TEMPLATES_PER_ROW:
                        col = 0
                        row = row + 1
                break

    def on_focus_neuron_changed(self, unit_label: str) -> None:
        """
        Whenever the neural unit with the display focus changes, update the plot to show the spike templates for that
        unit across all available analog channels.
        :param unit_label: Label identifying the neural unit with the focus.
        """
        if unit_label == self._unit_shown:
            return
        self._unit_shown = unit_label
        for unit in self.data_manager.neurons:
            if unit.label == self._unit_shown:
                row: int = 0
                col: int = 0
                for idx in self.data_manager.channel_indices:
                    t0 = col * _H_OFFSET_MS
                    y0 = row * _V_OFFSET_UV
                    pdi = self._spike_templates[idx]
                    template = unit.get_template_for_channel(idx)
                    if template is None:
                        pdi.setData(x=[t0, t0+10], y=[y0, y0],)
                    else:
                        pdi.setData(x=np.linspace(start=t0, stop=t0 + 10, num=len(template)),
                                    y=y0 + template)
                    col = col + 1
                    if col == _NUM_TEMPLATES_PER_ROW:
                        col = 0
                        row = row + 1
                break
