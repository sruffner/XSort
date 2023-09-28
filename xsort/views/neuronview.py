from typing import List, Any, Optional

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QTableView, QHeaderView, QHBoxLayout, QSizePolicy

from xsort.data.analyzer import Analyzer
from xsort.data.neuron import Neuron
from xsort.views.baseview import BaseView


class _NeuronTableModel(QAbstractTableModel):
    """
    Table model for the list of neural units exposed by the data manager object, :class:`Analyzer`. It is merely a
    wrapper around that list, and supports sorting the table on any of its columns: the neuron label, its primary
    channel, total # of spikes on the neuron, mean firing rate in Hz, SNR on primary channel, peak spike template
    amplitude (typically on the primary channel), and the observed percentage of interspike intervals less than 1ms.
        Each row in the table corresponds to one neuron. Any neuron that is currently in the neuron display focus list
    are highlighted by setting the background color for that row to the RGB color assigned to the neuron's position
    within that focus list.
    """

    _header_labels: List[str] = ['UID', 'Channel', '#Spikes', 'Rate (Hz)', 'SNR', 'Amp(\u00b5V)', '%ISI<1']
    """ Column header labels. """

    def __init__(self, data_manager: Analyzer):
        """ Construct an initally empty neurons table model. """
        QAbstractTableModel.__init__(self)
        self._data_manager = data_manager
        """ The data manager provides access to the list of neurons underlying this table model. """
        self._sort_col = 0
        """ The column on which the table model is currently sorted. """
        self._reversed = False
        """ True if rows are sorted in descending order, else ascending order. """
        self._sorted_indices: List[int] = [i for i in range(len(data_manager.neurons))]
        """ Maps table row index to index of corresponding neuron IAW the current sort column and sort order. """
        self.reload_table_data()

    def reload_table_data(self):
        self._resort()
        self.layoutChanged.emit()

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._data_manager.neurons)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self._header_labels)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:
        if role != Qt.DisplayRole:
            return None
        if (orientation == Qt.Horizontal) and (0 <= section < self.columnCount()):
            return self._header_labels[section]
        else:
            return None

    def data(self, index, role: int = Qt.DisplayRole) -> Any:
        r = index.row()
        c = index.column()
        if (0 <= r < self.rowCount()) and (0 <= c <= self.columnCount()):
            if role == Qt.DisplayRole:
                idx = self._sorted_indices[r]
                return _NeuronTableModel._to_string(self._data_manager.neurons[idx], c)
            elif (role == Qt.BackgroundRole) or (role == Qt.ForegroundRole):
                u = self._data_manager.neurons[self._sorted_indices[r]].label
                color_str = self._data_manager.display_color_for_neuron(u)
                bkg_color = QColor(Qt.white) if color_str is None else QColor.fromString(color_str)
                if role == Qt.BackgroundRole:
                    return bkg_color
                else:
                    return None if color_str is None else QColor(Qt.black if bkg_color.lightness() < 140 else Qt.white)
            elif role == Qt.TextAlignmentRole:
                return Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        return None

    @staticmethod
    def _to_string(u: Neuron, col: int):
        switcher = {
            0: u.label,
            1: '' if u.primary_channel is None else str(u.primary_channel),
            2: str(u.num_spikes),
            3: f"{u.mean_firing_rate_hz:.2f}",
            4: f"{u.snr:.2f}" if isinstance(u.snr, float) else "",
            5: f"{u.amplitude:.1f}" if isinstance(u.amplitude, float) else "",
            6: f"{(100.0 * u.fraction_of_isi_violations):.2f}"
        }
        return switcher.get(col, '')

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:
        rev = (order == Qt.SortOrder.DescendingOrder)
        col = max(0, min(column, self.columnCount()-1))
        if (self._sort_col != col) or (self._reversed != rev):
            self._sort_col = col
            self._reversed = rev
            self.reload_table_data()

    def _resort(self) -> None:
        self._sorted_indices.clear()
        u = self._data_manager.neurons
        num = len(u)
        if num > 1:
            switcher = {
                0: sorted(range(num), key=lambda k: k, reverse=self._reversed),
                1: sorted(range(num), key=lambda k: -1 if (u[k].primary_channel is None) else u[k].primary_channel,
                          reverse=self._reversed),
                2: sorted(range(num), key=lambda k: u[k].num_spikes, reverse=self._reversed),
                3: sorted(range(num), key=lambda k: u[k].mean_firing_rate_hz, reverse=self._reversed),
                4: sorted(range(num), key=lambda k: 0 if (u[k].snr is None) else u[k].snr, reverse=self._reversed),
                5: sorted(range(num), key=lambda k: 0 if (u[k].amplitude is None) else u[k].amplitude,
                          reverse=self._reversed),
                6: sorted(range(num), key=lambda k: u[k].fraction_of_isi_violations, reverse=self._reversed)
            }
            self._sorted_indices = switcher.get(self._sort_col)

    def unit_label_for_row(self, row: int) -> Optional[str]:
        """
        The unit label for the specified row in the table.
        :param row: Row index
        :return: Corresponding neural unit label, or None if row index is invalid.
        """
        if 0 <= row < self.rowCount():
            idx = self._sorted_indices[row]
            return self._data_manager.neurons[idx].label
        return None


class NeuronView(BaseView):
    """
    A tabular view of the list of neurons exposed by the data manager object, :class:`Analyzer`. Each row in the
    table represents one neural unit, with the unit label and various numerical statistics shown in the columns. The
    table may be sorted on any column, in ascending or descending order.
        The user selects a neuron for the display focus by clicking on it, and removes the focus by clicking it again.
    The display focus list may contain up to MAX_NUM_FOCUS_NEURONS, and a unique color is assigned to each slot in that
    list. Most other views in XSort display data for neurons in the current display list, using the assigned colors.
    """

    def __init__(self, data_manager: Analyzer) -> None:
        super().__init__('Neurons', None, data_manager)
        self._table_view = QTableView()
        """ Table view displaying the neuron table (read-only). """
        self._model = _NeuronTableModel(data_manager)
        """ Neuron table model (essentially wraps a table model around the data manager's neuron list). """

        self._table_view.setModel(self._model)
        self._table_view.setSortingEnabled(True)
        self._model.sort(0)   # to ensure table view is initially sorted by column 0 in ascending order
        self._table_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table_view.setSelectionMode(QTableView.SelectionMode.NoSelection)
        self._table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table_view.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table_view.verticalHeader().setVisible(False)
        size = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        size.setHorizontalStretch(1)
        self._table_view.setSizePolicy(size)

        self._table_view.clicked.connect(self.on_item_clicked)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self._table_view)
        self.view_container.setLayout(main_layout)

    def on_working_directory_changed(self) -> None:
        self._model.reload_table_data()

    def on_neuron_metrics_updated(self, unit_label: str) -> None:
        self._model.reload_table_data()

    def on_focus_neurons_changed(self) -> None:
        self._model.reload_table_data()

    def on_item_clicked(self, index: QModelIndex) -> None:
        u = self._model.unit_label_for_row(index.row())
        if not (u is None):
            self.data_manager.update_neurons_with_display_focus(u)
