from typing import List, Any, Optional

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, QPoint, QSettings
from PySide6.QtGui import QColor, QAction
from PySide6.QtWidgets import QTableView, QHeaderView, QHBoxLayout, QSizePolicy, QMenu

from xsort.data.analyzer import Analyzer
from xsort.data.neuron import Neuron
from xsort.views.baseview import BaseView


class _NeuronTableModel(QAbstractTableModel):
    """
    Table model for the list of neural units exposed by the data manager object, :class:`Analyzer`. It is merely a
    wrapper around that list, and supports sorting the table on any of its columns: the neuron label, its primary
    channel, total # of spikes on the neuron, mean firing rate in Hz, SNR on primary channel, peak spike template
    amplitude (typically on the primary channel), the observed percentage of interspike intervals less than 1ms, and
    the similarity metric. Note that the similarity metric is a comparison to the first unit in the focus list -- the
    so-called "primary neuron". If the focus list is empty, the metric is undefined for all units.
        Each row in the table corresponds to one neuron. Any neuron that is currently in the neuron display focus list
    are highlighted by setting the background color for that row to the RGB color assigned to the neuron's position
    within that focus list.
    """

    _header_labels: List[str] = ['UID', 'Label', 'Channel', '#Spikes', 'Rate (Hz)',
                                 'SNR', 'Amp(\u00b5V)', '%ISI<1', 'Similarity']
    """ Column header labels. """

    LABEL_COL_IDX = 1
    """ Index of the 'Label' column -- unit labels may be edited. """

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

    def hideable_columns(self) -> List[int]:
        """ The column indices of all columns that can be optionally hidden hidden. """
        return [i for i in range(1, len(self._header_labels))]

    def reload_table_data(self):
        self._resort()
        self.layoutChanged.emit()

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._data_manager.neurons)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self._header_labels)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if (orientation == Qt.Horizontal) and (0 <= section < self.columnCount()):
            return self._header_labels[section]
        else:
            return None

    def flags(self, index):
        f = super().flags(index)
        if index.column() == self.LABEL_COL_IDX:
            f = f | Qt.ItemFlag.ItemIsEditable
        return f

    def setData(self, index, value, role=...):
        r, c = index.row(), index.column()
        if ((0 <= r < self.rowCount()) and (index.column() == self.LABEL_COL_IDX) and
                (role == Qt.ItemDataRole.EditRole)):
            idx = self._sorted_indices[index.row()]
            return self._data_manager.edit_unit_label(idx, str(value))
        else:
            return super().setData(index, value, role)

    def data(self, index, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        r = index.row()
        c = index.column()
        if (0 <= r < self.rowCount()) and (0 <= c <= self.columnCount()):
            if role in [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole]:
                idx = self._sorted_indices[r]
                return self._to_string(self._data_manager.neurons[idx], c)
            elif (role == Qt.ItemDataRole.BackgroundRole) or (role == Qt.ItemDataRole.ForegroundRole):
                u = self._data_manager.neurons[self._sorted_indices[r]].uid
                color_str = self._data_manager.display_color_for_neuron(u)
                bkg_color = None if color_str is None else QColor.fromString(color_str)
                if role == Qt.BackgroundRole:
                    return bkg_color
                else:
                    return None if color_str is None else QColor(Qt.black if bkg_color.lightness() < 140 else Qt.white)
            elif role == Qt.ItemDataRole.TextAlignmentRole:
                return Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        return None

    def _to_string(self, u: Neuron, col: int):
        primary = self._data_manager.primary_neuron
        switcher = {
            0: u.uid,
            1: u.label,
            2: '' if u.primary_channel is None else str(u.primary_channel),
            3: str(u.num_spikes),
            4: f"{u.mean_firing_rate_hz:.2f}",
            5: f"{u.snr:.2f}" if isinstance(u.snr, float) else "",
            6: f"{u.amplitude:.1f}" if isinstance(u.amplitude, float) else "",
            7: f"{(100.0 * u.fraction_of_isi_violations):.2f}",
            8: "" if (primary is None) else ("---" if (primary.uid == u.uid) else f"{u.similarity_to(primary):.2f}")
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
        primary = self._data_manager.primary_neuron
        num = len(u)
        if num > 1:
            switcher = {
                0: sorted(range(num), key=lambda k: Neuron.dissect_uid(u[k].uid), reverse=self._reversed),
                1: sorted(range(num), key=lambda k: u[k].label, reverse=self._reversed),
                2: sorted(range(num), key=lambda k: -1 if (u[k].primary_channel is None) else u[k].primary_channel,
                          reverse=self._reversed),
                3: sorted(range(num), key=lambda k: u[k].num_spikes, reverse=self._reversed),
                4: sorted(range(num), key=lambda k: u[k].mean_firing_rate_hz, reverse=self._reversed),
                5: sorted(range(num), key=lambda k: 0 if (u[k].snr is None) else u[k].snr, reverse=self._reversed),
                6: sorted(range(num), key=lambda k: 0 if (u[k].amplitude is None) else u[k].amplitude,
                          reverse=self._reversed),
                7: sorted(range(num), key=lambda k: u[k].fraction_of_isi_violations, reverse=self._reversed),
                8: sorted(range(num), key=lambda k: k if (primary is None) else u[k].similarity_to(primary),
                          reverse=self._reversed)
            }
            self._sorted_indices = switcher.get(self._sort_col)
        else:
            self._sorted_indices.append(0)   # only one unit -- nothing to sort!

    def unit_uid_for_row(self, row: int) -> Optional[str]:
        """
        The UID for the neural unit displayed in the specified row in the table.
        :param row: Row index
        :return: UID of the corresponding neural unit, or None if row index is invalid.
        """
        if 0 <= row < self.rowCount():
            idx = self._sorted_indices[row]
            return self._data_manager.neurons[idx].uid
        return None

    def find_uid_of_neighboring_unit(self, uid: str) -> Optional[str]:
        """
        Find the UID of the neural unit occupying the table row immediately below the specified unit. If the specified
        unit is at the bottom of the table, then return the UID of the unit in the row above.
        :param uid: UID of a neural unit
        :return: UID of the unit's neighbor IAW the current sort order, as described. Returns None if the table is
            empty or contains only one unit, or if the specified unit is not in the table.
        """
        if self.rowCount() < 2:
            return None
        for row in range(self.rowCount()):
            idx = self._sorted_indices[row]
            if self._data_manager.neurons[idx].uid == uid:
                row = (row - 1) if (row == self.rowCount() - 1) else (row + 1)
                idx = self._sorted_indices[row]
                return self._data_manager.neurons[idx].uid
        return None


class NeuronView(BaseView):
    """
    A tabular view of the list of neurons exposed by the data manager object, :class:`Analyzer`. Each row in the
    table represents one neural unit, with the unit label and various numerical statistics shown in the columns. The
    table may be sorted on any column, in ascending or descending order. The visibility of selected columns may be
    toggled via a context menu raised when the user clicks anywhere on the column header.
        The user selects a neuron for the display focus by clicking on it, and removes the focus by clicking it again.
    The display focus list may contain up to MAX_NUM_FOCUS_NEURONS, and a unique color is assigned to each slot in that
    list. Most other views in XSort display data for neurons in the current display list, using the assigned colors.
    """

    def __init__(self, data_manager: Analyzer) -> None:
        super().__init__('Neurons', None, data_manager)
        self._table_view = QTableView()
        """ Table view displaying the neuron table (read-only). """
        self._table_context_menu = QMenu(self._table_view)
        """ Context menu for table to hide/show selected table columns. """
        self._model = _NeuronTableModel(data_manager)
        """ Neuron table model (essentially wraps a table model around the data manager's neuron list). """

        self._table_view.setModel(self._model)
        self._table_view.setSortingEnabled(True)
        self._model.sort(0)   # to ensure table view is initially sorted by column 0 in ascending order
        self._table_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._table_view.setSelectionMode(QTableView.SelectionMode.NoSelection)
        self._table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table_view.horizontalHeader().setStretchLastSection(True)
        self._table_view.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table_view.verticalHeader().setVisible(False)
        size = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        size.setHorizontalStretch(1)
        self._table_view.setSizePolicy(size)

        self._table_view.clicked.connect(self.on_item_clicked)

        # set up context menu for toggling the visibility of selected columns in the table.
        for col in self._model.hideable_columns():
            action = QAction(self._model.headerData(col, Qt.Horizontal), parent=self._table_context_menu,
                             checkable=True, checked=True,
                             triggered=lambda checked, x=col: self._toggle_table_column_visibility(x))
            self._table_context_menu.addAction(action)
        self._table_view.horizontalHeader().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table_view.horizontalHeader().customContextMenuRequested.connect(self._on_table_right_click)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self._table_view)
        self.view_container.setLayout(main_layout)

    def uid_of_unit_below(self, uid: str) -> Optional[str]:
        """
        Get the UID of the unit in the row below (or above, if necessary) the table row that displays the unit
        specified. The order of the rows depends on the current sort order.

        :param uid: The UID of a neural unit
        :return: UID of unit displayed in the row below or above that unit, or None if not found.
        """
        return self._model.find_uid_of_neighboring_unit(uid)

    def on_working_directory_changed(self) -> None:
        self._model.reload_table_data()

    def on_neuron_metrics_updated(self, uid: str) -> None:
        self._model.reload_table_data()

    def on_focus_neurons_changed(self) -> None:
        self._model.reload_table_data()

    def on_neuron_label_updated(self, uid: str) -> None:
        self._model.reload_table_data()

    def on_item_clicked(self, index: QModelIndex) -> None:
        u = self._model.unit_uid_for_row(index.row())
        if not (u is None):
            self.data_manager.update_neurons_with_display_focus(u)

    def _on_table_right_click(self, pos: QPoint) -> None:
        """ Handler raises the context menu by which user toggles the visiblity of selected columns in the table. """
        pos = self._table_view.horizontalHeader().mapToGlobal(pos)
        pos += QPoint(5, 10)
        self._table_context_menu.move(pos)
        self._table_context_menu.show()

    def _toggle_table_column_visibility(self, col: int) -> None:
        """
        Toggle the visibility of the specified column in the neural units table. No action is taken if the specified
        column may not be hidden.
        :param col: The table column index.
        """
        if col in self._model.hideable_columns():
            hidden = self._table_view.isColumnHidden(col)
            if hidden:
                self._table_view.showColumn(col)
            else:
                self._table_view.hideColumn(col)

    def save_settings(self, settings: QSettings) -> None:
        """ Overridden to preserve which columns in the neural units table have been hidden by the user. """
        hidden = [str(i) for i in self._model.hideable_columns() if self._table_view.isColumnHidden(i)]
        settings.setValue('neuronview_hidden_cols', ','.join(hidden))

    def restore_settings(self, settings: QSettings) -> None:
        """ Overridden to restore the current histogram span from user settings. """
        try:
            # because the corresponding context menu items are initially checked, we have to fix them after hiding
            # any columns to correctly reflect the current state
            hidden: str = settings.value('neuronview_hidden_cols', '')
            hidden_column_labels: List[str] = list()
            for col_str in hidden.split(','):
                col = int(col_str)
                if col in self._model.hideable_columns():
                    self._table_view.hideColumn(col)
                    hidden_column_labels.append(self._model.headerData(col, Qt.Horizontal))
            if len(hidden_column_labels) > 0:
                for a in self._table_context_menu.actions():
                    if a.text() in hidden_column_labels:
                        a.setChecked(False)
        except Exception:
            pass
