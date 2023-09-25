from typing import List, Any, Optional

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, QItemSelection, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QTableView, QHeaderView, QHBoxLayout, QSizePolicy

from xsort.data.analyzer import Analyzer
from xsort.data.neuron import Neuron
from xsort.views.baseview import BaseView


class NeuronTableModel(QAbstractTableModel):
    """ TODO: UNDER DEV """

    _header_labels: List[str] = ['UID', 'Channel', '#Spikes', 'Rate (Hz)', 'SNR', 'Amp(\u00b5V)', '%ISI<1']

    def __init__(self):
        """ Construct an initally empty neurons table model. """
        QAbstractTableModel.__init__(self)
        self._data: List[List[str]] = list()
        self.load_table_data(list())

    def load_table_data(self, neurons: List[Neuron]):
        self._data.clear()
        for u in neurons:
            row = [u.label, '' if u.primary_channel is None else u.primary_channel, str(u.num_spikes),
                   f"{u.mean_firing_rate_hz:.2f}", f"{u.snr:.2f}" if isinstance(u.snr, float) else "",
                   f"{u.amplitude:.1f}" if isinstance(u.amplitude, float) else "",
                   f"{(100.0 * u.fraction_of_isi_violations):.2f}"]
            self._data.append(row)

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._data)

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
                return self._data[r][c]
            elif role == Qt.BackgroundRole:
                return QColor(Qt.white)
            elif role == Qt.TextAlignmentRole:
                return Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        return None

    def unit_label_for_row(self, row: int) -> Optional[str]:
        """
        The unit label for the specified row in the table.
        :param row: Row index
        :return: Corresponding neural unit label, or None if row index is invalid.
        """
        return self._data[row][0] if (0 <= row < self.rowCount()) else None


class NeuronView(BaseView):
    """ TODO: UNDER DEV """

    # noinspection PyUnresolvedReferences
    def __init__(self, data_manager: Analyzer) -> None:
        super().__init__('Neurons', None, data_manager)
        self._table_view = QTableView()
        """ Table view displaying the neuron table (read-only). """
        self._model = NeuronTableModel()

        self._table_view.setModel(self._model)
        self._table_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table_view.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self._table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table_view.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table_view.verticalHeader().setVisible(False)
        size = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        size.setHorizontalStretch(1)
        self._table_view.setSizePolicy(size)

        self._table_view.selectionModel().selectionChanged.connect(self.on_selection_changed)
        main_layout = QHBoxLayout()
        main_layout.addWidget(self._table_view)
        self.view_container.setLayout(main_layout)

    def on_working_directory_changed(self) -> None:
        self._reload_neuron_table_contents()

    def on_neuron_metrics_updated(self, unit_label: str) -> None:
        self._reload_neuron_table_contents()

    def on_focus_neuron_changed(self, unit_label: str) -> None:
        for row in range(self._model.rowCount()):
            if unit_label == self._model.unit_label_for_row(row):
                self._table_view.selectRow(row)
                break

    def _reload_neuron_table_contents(self) -> None:
        """ Reloads the entire neuron table from scratch. """
        self._model.load_table_data(self.data_manager.neurons)
        self._model.layoutChanged.emit()

    @Slot(QItemSelection, QItemSelection)
    def on_selection_changed(self, selected: QItemSelection, _: QItemSelection) -> None:
        if isinstance(selected, QItemSelection):
            unit_selected = self._model.unit_label_for_row(selected.first().bottom())
            self.selected_neuron_changed.emit(unit_selected)
