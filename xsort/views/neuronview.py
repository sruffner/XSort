from typing import List, Any, Optional, Set, Tuple

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, QPoint, QSettings, Slot, QObject, QEvent, Signal
from PySide6.QtGui import QColor, QAction, QGuiApplication, QFontMetricsF, QKeyEvent, QHelpEvent, \
    QColorConstants
from PySide6.QtWidgets import QTableView, QHeaderView, QSizePolicy, QMenu, QStyledItemDelegate, QWidget, \
    QLineEdit, QCompleter, QCheckBox, QVBoxLayout, QLabel, QHBoxLayout

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

    _header_labels: List[str] = ['UID', 'Channel', '#Spikes', 'Rate (Hz)',
                                 'SNR', 'Amp(\u00b5V)', '%ISI<1', 'Similarity', 'Label']
    """ Column header labels. """

    _col_values_for_sizing: List[str] = ['000x', '000', '000000', '0000.00',
                                         '00.00', '00.0', '0.00', '0.00', 'Purkinje']
    """ Typical cell value for each column -- to calculate fixed column sizes. """

    UID_COL_IDX = 0
    """ Index of the 'UID' column -- the first column in the table. """
    LABEL_COL_IDX = 8
    """ Index of the 'Label' column -- unit labels may be edited. """
    SIM_COL_IDX = 7
    """ Index of the 'Similarity' column -- sorting on this column is not permitted. """
    _SIMILAR_HILITE = QColor.fromString("#80B2D8FF")
    """ Background cell color highlighting units most similar to the current primary focus unit (if any). """
    _SELECTED_HILITE = QColor(Qt.GlobalColor.darkBlue)
    """ 
    Background cell color (UID column only) highlighting units currently selected for a multi-unit relabel or 
    delete operation.
    """

    edit_selection_set_changed: Signal = Signal()
    """ Signal emitted whenever the edit selection set changes. """

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
        self._selected_uids: Set[str] = set()
        """ 
        UIDs of units in the current "edit selection set" -- chosen by the user for a multi-unit delete/relabel op.
        """
        self._last_selected_row: Optional[int] = None
        """ Row of unit last added to the edit selection set (to support contiguous range selection). """
        self._highlight_similar: bool = False
        """ 
        If True, when a primary focus unit is defined, the model always lists the 5 most similar units 
        immediately after, regardless the sort order; the similar units are also highlighted with a distinctive
        background color.
        """
        self._similar_indices: List[int] = list()
        """
        Whenever the primary unit (first neuron in display/focus list) is defined, this contains the indices of the
        <= 5 most similar units within the neuron list.
        """
        self.reload_table_data()

    @property
    def highlight_similar(self) -> bool:
        """
        True if model highlights the <=5 neural units most similar to the current primary focus unit. These units
        are highlighted with a distinctive background color and appear immediately after the primary unit, regardless
        the sort order.
        """
        return self._highlight_similar

    @highlight_similar.setter
    def highlight_similar(self, ena: bool) -> None:
        """
        Enable/disable highlighting of the <= 5 neural units most similar to the current primary focus unit.
        :param ena: True to enable, False to disable.
        """
        if ena != self._highlight_similar:
            self._highlight_similar = ena
            if self._data_manager.primary_neuron is not None:
                self.reload_table_data()

    @property
    def current_sort_column_and_order(self) -> Tuple[int, Qt.SortOrder]:
        """ The current sort column and sort order for this model. """
        return self._sort_col, Qt.SortOrder.DescendingOrder if self._reversed else Qt.SortOrder.AscendingOrder

    def hideable_columns(self) -> List[int]:
        """ The column indices of all columns that can be optionally hidden hidden. """
        return [i for i in range(1, len(self._header_labels))]

    def toggle_select_row(self, row: int) -> None:
        """
        Toggle the current selection state of the specified table row.
        :param row: The row index.
        """
        uid = self.row_to_unit(row)
        if uid is None:
            return
        if uid in self._selected_uids:
            self._selected_uids.remove(uid)
            # if the row removed is the last selected, the last selected row is undefined
            if len(self._selected_uids) == 0 or row == self._last_selected_row:
                self._last_selected_row = None
        else:
            self._selected_uids.add(uid)
            self._last_selected_row = row
        top_left = self.createIndex(row, 0)
        bot_right = self.createIndex(row, self.LABEL_COL_IDX)
        self.dataChanged.emit(top_left, bot_right, Qt.ItemDataRole.BackgroundRole)
        self.edit_selection_set_changed.emit()

    def select_contiguous_range(self, end_row: int) -> None:
        """
        Select a contiguous range of rows in the table from the last-selected row to the row specified. If the selection
        set is currently empty or the last-selected row is unknown, then the unit in the row specified is added to the
        selection set.
        :param end_row: The row at which to end the continuous-range selection. No action taken if invalid.
        """
        end_row_uid = self.row_to_unit(end_row)
        if (end_row_uid is not None) and not (end_row_uid in self._selected_uids):
            start_row = end_row
            if (len(self._selected_uids) == 0) or (self._last_selected_row is None):
                self._selected_uids.add(end_row_uid)
            else:
                start_row = self._last_selected_row
                inc = 1 if start_row < end_row else - 1
                r = start_row + inc
                while True:
                    uid = self.row_to_unit(r)
                    if uid is not None:
                        self._selected_uids.add(uid)
                    if r == end_row:
                        break
                    r = r + inc
            self._last_selected_row = end_row
            top_left = self.createIndex(min(start_row, end_row), self.UID_COL_IDX)
            bot_right = self.createIndex(max(start_row, end_row), self.UID_COL_IDX)
            self.dataChanged.emit(top_left, bot_right, Qt.ItemDataRole.BackgroundRole)
            self.edit_selection_set_changed.emit()

    def clear_current_selection(self) -> None:
        """ Clear the set of currently selected table rows, if any. """
        if len(self._selected_uids) > 0:
            self._selected_uids.clear()
            self._last_selected_row = 0
            top_left = self.createIndex(0, self.UID_COL_IDX)
            bot_right = self.createIndex(self.rowCount() - 1, self.UID_COL_IDX)
            self.dataChanged.emit(top_left, bot_right, Qt.ItemDataRole.BackgroundRole)
            self.edit_selection_set_changed.emit()

    @property
    def units_in_current_selection(self) -> Set[str]:
        """ UIDs of the neural units that currently comprise the edit selection set in the unit table. """
        return self._selected_uids.copy()

    @property
    def _unit_indices_in_current_selection(self) -> List[int]:
        out = list()
        if len(self._selected_uids) > 0:
            n_found = 0
            for idx, u in enumerate(self._data_manager.neurons):
                if u.uid in self._selected_uids:
                    out.append(idx)
                    n_found += 1
                    if n_found == len(self._selected_uids):
                        break
        return out

    def reload_table_data(self, clear_selection: bool = False):
        if clear_selection:
            self._selected_uids.clear()
            self._last_selected_row = 0
        self._resort()
        self.layoutChanged.emit()

    def on_unit_labels_changed(self, uids: List[str]) -> None:
        r_min, r_max = len(self._sorted_indices), -1
        n_found = 0
        for r in range(len(self._sorted_indices)):
            if self._data_manager.neurons[self._sorted_indices[r]].uid in uids:
                r_min, r_max = min(r_min, r), max(r_max, r)
                n_found += 1
                if n_found == len(uids):
                    break
        if n_found > 0:
            top_left = self.createIndex(r_min, self.LABEL_COL_IDX)
            bot_right = self.createIndex(r_max, self.LABEL_COL_IDX)
            self.dataChanged.emit(top_left, bot_right, [Qt.ItemDataRole.DisplayRole])

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
        """
        Overridden to support in-place editing of a neural unit label. If multiple units are selected, all of them
        qre updated with the same label.
        """
        r, c = index.row(), index.column()
        if (0 <= r < self.rowCount()) and (c == self.LABEL_COL_IDX) and (role == Qt.ItemDataRole.EditRole):
            idx = self._sorted_indices[r]
            selected_indices = self._unit_indices_in_current_selection
            if not (idx in selected_indices):
                selected_indices.append(idx)
            return self._data_manager.edit_neuron_labels(selected_indices, str(value))
        else:
            return super().setData(index, value, role)

    def data(self, index, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        r = index.row()
        c = index.column()
        if (0 <= r < self.rowCount()) and (0 <= c <= self.columnCount()):
            idx = self._sorted_indices[r]
            if role in [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole]:
                return self._to_string(self._data_manager.neurons[idx], c)
            elif (role == Qt.ItemDataRole.BackgroundRole) or (role == Qt.ItemDataRole.ForegroundRole):
                # background highlight: dark blue in UID cell if unit is "selected". Else all cells in row have default
                # background unless the unit is one of the focus units or one of the highlighted similar units.
                uid = self._data_manager.neurons[idx].uid
                if (uid in self._selected_uids) and (c == self.UID_COL_IDX):
                    bkg_color = self._SELECTED_HILITE
                else:
                    color_str = self._data_manager.display_color_for_neuron(uid)
                    bkg_color = None if color_str is None else QColor.fromString(color_str)
                    if (bkg_color is None) and (idx in self._similar_indices):
                        bkg_color = self._SIMILAR_HILITE
                if role == Qt.BackgroundRole:
                    return bkg_color
                else:
                    if bkg_color is None:
                        return None
                    else:
                        # choose white or black based on estimated luminance threshold
                        lum = bkg_color.red() * 0.299 + bkg_color.green() * 0.587 + bkg_color.blue() * 0.114
                        return QColorConstants.Black if lum > 150 else QColorConstants.White
            elif role == Qt.ItemDataRole.TextAlignmentRole:
                return Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
            elif (role == Qt.ItemDataRole.ToolTipRole) and (c == self.LABEL_COL_IDX):
                return self._to_string(self._data_manager.neurons[idx], c)
        return None

    def _to_string(self, u: Neuron, col: int):
        primary = self._data_manager.primary_neuron
        switcher = {
            0: u.uid,
            1: '' if u.primary_channel is None else str(u.primary_channel),
            2: str(u.num_spikes),
            3: f"{u.mean_firing_rate_hz:.2f}",
            4: f"{u.snr:.2f}" if isinstance(u.snr, float) else "",
            5: f"{u.amplitude:.1f}" if isinstance(u.amplitude, float) else "",
            6: f"{(100.0 * u.fraction_of_isi_violations):.2f}",
            7: "" if (primary is None) else ("---" if (primary.uid == u.uid) else f"{u.similarity_to(primary):.2f}"),
            8: u.label
        }
        return switcher.get(col, '')

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:
        """ Overrridden to implement sorting on any column except the 'Similarity' column. """
        rev = (order == Qt.SortOrder.DescendingOrder)
        col = max(0, min(column, self.columnCount()-1))
        if (col != self.SIM_COL_IDX) and ((self._sort_col != col) or (self._reversed != rev)):
            self._sort_col = col
            self._reversed = rev
            self.reload_table_data()

    def _resort(self) -> None:
        self._sorted_indices.clear()
        self._similar_indices.clear()
        u = self._data_manager.neurons
        primary = self._data_manager.primary_neuron
        num = len(u)
        if num > 1:
            switcher = {
                0: sorted(range(num), key=lambda k: Neuron.dissect_uid(u[k].uid), reverse=self._reversed),
                1: sorted(range(num), key=lambda k: -1 if (u[k].primary_channel is None) else u[k].primary_channel,
                          reverse=self._reversed),
                2: sorted(range(num), key=lambda k: u[k].num_spikes, reverse=self._reversed),
                3: sorted(range(num), key=lambda k: u[k].mean_firing_rate_hz, reverse=self._reversed),
                4: sorted(range(num), key=lambda k: 0 if (u[k].snr is None) else u[k].snr, reverse=self._reversed),
                5: sorted(range(num), key=lambda k: 0 if (u[k].amplitude is None) else u[k].amplitude,
                          reverse=self._reversed),
                6: sorted(range(num), key=lambda k: u[k].fraction_of_isi_violations, reverse=self._reversed),
                7: sorted(range(num), key=lambda k: k if (primary is None) else u[k].similarity_to(primary),
                          reverse=self._reversed),
                8: sorted(range(num), key=lambda k: u[k].label, reverse=self._reversed)
            }
            self._sorted_indices = switcher.get(self._sort_col)

            # when primary focus neuron defined, we always list the 5 most similar units immediately after it,
            # regardless the sort order
            if self._highlight_similar and (primary is not None):
                similar = sorted(range(num), key=lambda k: u[k].similarity_to(primary), reverse=True)
                primary_idx = similar[0]
                self._similar_indices = similar[1:6]
                for idx in self._similar_indices:
                    self._sorted_indices.remove(idx)
                primary_pos = self._sorted_indices.index(primary_idx)
                for idx in reversed(self._similar_indices):
                    self._sorted_indices.insert(primary_pos+1, idx)

        else:
            self._sorted_indices.append(0)   # only one unit -- nothing to sort!

    def row_to_unit(self, row: int) -> Optional[str]:
        """
        The UID for the neural unit displayed in the specified row in the table.
        :param row: Row index
        :return: UID of the corresponding neural unit, or None if row index is invalid.
        """
        if 0 <= row < self.rowCount():
            idx = self._sorted_indices[row]
            return self._data_manager.neurons[idx].uid
        return None

    def unit_to_row(self, uid: str) -> Optional[int]:
        """
        Find the table row displaying the specified neural unit given the current sort order.
        :param uid: UID of a neural unit in table.
        :return: The corresponding row index, or None if unit was not found.
        """
        for row in range(self.rowCount()):
            idx = self._sorted_indices[row]
            if self._data_manager.neurons[idx].uid == uid:
                return row
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
        row = self.unit_to_row(uid)
        if isinstance(row, int):
            row = (row - 1) if (row == self.rowCount() - 1) else (row + 1)
            idx = self._sorted_indices[row]
            return self._data_manager.neurons[idx].uid
        return None

    @staticmethod
    def calc_fixed_column_widths(cell_fm: QFontMetricsF, hdr_fm: QFontMetricsF) -> List[int]:
        """
        Calculate the fixed widths of each column in the table.

        For each column, the fixed width is the greater of the width of a typical cell value or the width of the
        corresponding header label. We measure width as the C*(N+4), where C is the average character width according
        to the provided font metrics and N is the number of characters in the header label or sample cell value.

        :param cell_fm: Font metrics for values in the table cells.
        :param hdr_fm: Font metrics for the the values in the table header cells.
        :return: List of fixed column widths.
        """
        out: List[int] = list()
        for hdr in _NeuronTableModel._header_labels:
            out.append(int(hdr_fm.averageCharWidth() * (len(hdr) + 4)))
        for i, val in enumerate(_NeuronTableModel._col_values_for_sizing):
            out[i] = max(out[i], int(cell_fm.averageCharWidth() * (len(val) + 2)))
        return out


class _UnitLabelColumnDelegate(QStyledItemDelegate):
    """ A table view delegate to handle in-place editing of a neural unit label, with auto-completion support. """
    def __init__(self, table: '_NeuronTableView') -> None:
        super(_UnitLabelColumnDelegate, self).__init__(table)
        self.table: _NeuronTableView = table
        """ The neuron table. Need this to access the list of suggested unit labels for autocompletion support. """

    def createEditor(self, parent, option, index) -> QWidget:
        """
        The unit label editor is a **QLineEdit** configures with an inline auto-completer populated with the
        delegate's set of suggestions.
        """
        editor = QLineEdit(parent)
        qc = QCompleter(self.table.suggested_unit_labels, editor)
        qc.setCompletionMode(QCompleter.CompletionMode.InlineCompletion)
        qc.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        editor.setCompleter(qc)
        return editor


class _NeuronTableView(QTableView):
    """
    The neural unit table view.

    This **QTableView** subclass tailors the base implementation to provide a tabular display of the neural units in
    the current working directory:
     - Installs :class:`_NeuronTableModel` as the table model for the view.
     - Normal selection behavior is disabled to support the notion of a **display list** containing at most 3 different
       units (each a row in the table). Clicking any row with no modifier key pressed clears the previous display list
       and selects the unit in that row as the sole member of the new display list, aka the **primary neuron**. With any
       modifier key held down, clicking a row adds the corresponding unit to the display list (if it contains fewer than
       three units).
     - Uses fixed column widths and a fixed row height -- which significantly improves performance when the table
       contains hundreds of rows.
     - Overrides the default gesture for triggering an in-place edit (double-clicking on the table cell). We want the
       user to be able to change any unit's label without "selecting" the corresponding row, which changes the current
       display list, triggers background tasks to calculate statistics, etc. Instead, a right-click on a table cell
       containing a unit label raises the in-place edit.
     - Customized handling of the `Up` and `Down` arrow keys so that user can change the unit selected as the primary
       neuron. In this case, if the display list contained a second or third unit, those are removed -- the arrow keys
       are intended to change the primary unit without having to always use the mouse.
     - The **Label** column may not be wide enough to show a longer unit label (up to 25 chars), so the label is shown
       in a tooltip IF the label does not fit within its table cell.
    """
    def __init__(self, data_manager: Analyzer, settings: QSettings):
        """
        Construct and configure the neural unit table view.
        :param data_manager: The data model manager object representing the current state/contents of the current XSort
            working directory.
        :param settings: The application settings object. Stores list of suggested unit labels, among other app-wide
            and view-specific user preferences.
        """
        super().__init__()
        self._data_manager = data_manager
        """ A reference to the data model manager. """
        self._settings = settings
        """ A reference to the application settings object. """
        self.model = _NeuronTableModel(data_manager)
        """ Neuron table model (essentially wraps a table model around the data manager's neuron list). """
        self._table_context_menu = QMenu(self)
        """ Context menu for table header to hide/show selected table columns. """
        self._label_col_delegate = _UnitLabelColumnDelegate(self)

        self.setModel(self.model)

        # allow sorting on any column except Similarity column. Ensure table view is initially sorted by the UID
        # column (0) in ascending order
        self.setSortingEnabled(True)
        self.model.sort(0)  # to ensure table view is initially sorted by column 0 in ascending order
        self.horizontalHeader().setSortIndicator(0, Qt.SortOrder.AscendingOrder)
        self.horizontalHeader().sortIndicatorChanged.connect(self._on_sort_indicator_changed)

        self.model.layoutChanged.connect(self._on_table_layout_change)

        self.setSelectionMode(QTableView.SelectionMode.NoSelection)
        self.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.setItemDelegateForColumn(_NeuronTableModel.LABEL_COL_IDX, self._label_col_delegate)

        # we use fixed column widths and a fixed row height to speed up performance when there are lots of rows
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.horizontalHeader().setMinimumSectionSize(30)
        col_widths = _NeuronTableModel.calc_fixed_column_widths(self.fontMetrics(),
                                                                self.horizontalHeader().fontMetrics())
        for col, w in enumerate(col_widths):
            self.horizontalHeader().resizeSection(col, w)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        min_ht = int(self.fontMetrics().height()) + 6
        self.verticalHeader().setMinimumSectionSize(min_ht)
        self.verticalHeader().setDefaultSectionSize(min_ht)
        self.verticalHeader().setVisible(False)

        size = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        size.setHorizontalStretch(1)
        self.setSizePolicy(size)

        self.clicked.connect(self._on_cell_clicked)   # to trigger update of current display list

        # set up context menu for toggling the visibility of selected columns in the table.
        for col in self.model.hideable_columns():
            action = QAction(self.model.headerData(col, Qt.Horizontal), parent=self._table_context_menu,
                             checkable=True, checked=True,
                             triggered=lambda checked, x=col: self._toggle_table_column_visibility(x))
            self._table_context_menu.addAction(action)
        self.horizontalHeader().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.horizontalHeader().customContextMenuRequested.connect(self._on_header_right_click)

        # show tooltip only for a unit label when the label text does not fit within the table cell
        self.viewport().installEventFilter(self)

    @property
    def hidden_columns(self) -> str:
        """
        Comma-separated list of column indices of any hidden columns in the neuron table view. Empty string if all
        columns are currently visible. Intended for saving column-hide state to user settings.
        """
        return ','.join([str(i) for i in self.model.hideable_columns() if self.isColumnHidden(i)])

    @hidden_columns.setter
    def hidden_columns(self, hidden: str) -> None:
        """
        Set which columns, if any, are hidden in the neuron table view.
        :param hidden: A comma-separated list of column indices indicating which columns should be hidden. Any
            invalid indices are ignored.
        """
        hidden_column_labels: List[str] = list()
        for col_str in hidden.split(','):
            try:
                col = int(col_str)
            except ValueError:
                continue
            if col in self.model.hideable_columns():
                self.hideColumn(col)
                hidden_column_labels.append(self.model.headerData(col, Qt.Horizontal))
        # make sure corresponding menu items in the table header context menu are updated accordingly
        if len(hidden_column_labels) > 0:
            for a in self._table_context_menu.actions():
                if a.text() in hidden_column_labels:
                    a.setChecked(False)
                else:
                    a.setChecked(True)

    @property
    def suggested_unit_labels(self) -> List[str]:
        """
        A list of suggested labels for any neural unit, to facilitate auto-completion when a user is editing a unit
        label in the table view. Empty list if there are no suggestions.
        """
        labels = self._settings.value('suggested_unit_labels', '').split(',')
        return sorted(labels)

    def add_suggested_unit_label(self, s: str) -> None:
        """
        Add specified string as a suggested label for a neural unit, for auto-completion while editing any unit label.
        A valid unit label must be a non-empty string with <=25 characters, no leading or trailing whitespace, and
        cannot contain a comma.
        :param s: The suggested label. Ignored if invalid or duplicates a previously suggested label.
        """
        label_set = set(self.suggested_unit_labels)
        if (len(s) > 0) and (len(s) == len(s.strip())) and Neuron.is_valid_label(s) and not (s in label_set):
            label_set.add(s)
            self._settings.setValue('suggested_unit_labels', ','.join(sorted(label_set)))

    def closeEditor(self, editor: QWidget, hint: Optional[QStyledItemDelegate.EndEditHint]) -> None:
        """
        Overridden so that a **Tab (Shift-Tab)** keypress terminating the in-place edit of a unit label will
        immediately trigger editing of the unit label in the next (previous) row.
        """
        if hint in (QStyledItemDelegate.EndEditHint.EditNextItem, QStyledItemDelegate.EndEditHint.EditPreviousItem):
            super().closeEditor(editor, QStyledItemDelegate.EndEditHint.SubmitModelCache)
            incr = 1 if hint == QStyledItemDelegate.EndEditHint.EditNextItem else -1
            index = self.currentIndex()
            next_idx = self.model.createIndex(index.row() + incr, index.column())
            if next_idx.isValid():
                self.setCurrentIndex(next_idx)
                self.edit(next_idx)
        else:
            super().closeEditor(editor, hint)

    def keyReleaseEvent(self, event: QKeyEvent):
        """
        Handles response to implement several keyboard shortcuts for the neuron table:
         - Esc or Space key: Clears the current multi-unit edit selection set, if any.
         - Up/Down arrow keys: Shifts the primary focus unit to the unit in the table row above/below the row
           containing the current primary unit (no "wrap-around"). If there is a second or third unit in the current
           focus list, they are removed. If there is no primary unit, the first unit in the table becomes the primary.
         - Ctrl(Cmd)-Up/Down: Shifts the secondary focus unit to the unit in the table row above/below the row
           containing the current secondary unit (no "wrap-around"), but skipping over the current primary unit. If
           there is no secondary unit, the unit above or below the current primary unit becomes the seconary unit. If
           there is no primary unit, the first two units become the primary and secondary units.

        :param event: The key event. Any keys not listed above are ignored (passed on to the base class).
        """
        event.accept()
        inc = 1 if event.key() == Qt.Key.Key_Down else (-1 if event.key() == Qt.Key.Key_Up else 0)
        is_ctrl = (event.modifiers() & Qt.KeyboardModifier.ControlModifier) == Qt.KeyboardModifier.ControlModifier
        if inc == 0 and (event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Escape)):
            self.model.clear_current_selection()
        elif inc != 0 and len(self._data_manager.neurons) > 0:
            # if Ctrl key depressed, shift secondary unit up/down, leaving primary unit unchanged. If a tertiary unit
            # was selected, it is deselected. If focus list empty, select units in first two rows.
            if is_ctrl:
                curr_focus = self._data_manager.neurons_with_display_focus
                if len(curr_focus) == 0:
                    primary_uid = self.model.row_to_unit(0)
                    secondary_uid = self.model.row_to_unit(1)
                elif len(curr_focus) == 1:
                    primary_uid = curr_focus[0].uid
                    secondary_uid = self.model.row_to_unit(self.model.unit_to_row(primary_uid) + inc)
                else:
                    primary_uid = curr_focus[0].uid
                    row_1 = self.model.unit_to_row(primary_uid)
                    row_2 = self.model.unit_to_row(curr_focus[1].uid) + inc
                    if row_2 == row_1:
                        row_2 = row_2 + inc
                    secondary_uid = self.model.row_to_unit(row_2)
                if isinstance(primary_uid, str):
                    self._data_manager.update_neurons_with_display_focus([primary_uid, secondary_uid])
                    row = self.model.unit_to_row(secondary_uid if isinstance(secondary_uid, str) else primary_uid)
                    self.scrollTo(self.model.createIndex(row, 0))
                return

            # if Ctrl key not depressed, shift primary unit up/down and remove any other units from focus list.
            if self._data_manager.primary_neuron is None:
                row = 0 if (len(self._data_manager.neurons) > 0) else -1
            else:
                curr_row = self.model.unit_to_row(self._data_manager.primary_neuron.uid)
                row = (curr_row + inc) if isinstance(curr_row, int) else -1

            uid = self.model.row_to_unit(row)
            if isinstance(uid, str):
                self._data_manager.update_neurons_with_display_focus(uid, True)
                self.scrollTo(self.model.createIndex(row, 0))
        else:
            event.ignore()
            super().keyReleaseEvent(event)

    @Slot()
    def _on_table_layout_change(self) -> None:
        """
        Whenever the layout of the underlying table changes, scroll if necessary to ensure the row corresponding to
        the current primary unit (if any) is visible.
        """
        if self._data_manager.primary_neuron is not None:
            row = self.model.unit_to_row(self._data_manager.primary_neuron.uid)
            if row is not None:
                self.scrollTo(self.model.createIndex(row, 0))

    @Slot(QModelIndex)
    def _on_cell_clicked(self, index: QModelIndex) -> None:
        """
        Handles response to the user left-clicking on a cell in the neuron table, which will depend on what modifier
        key is depressed:
         - Click (no modifier): Single-select the primary focus unit.
         - Ctrl(Cmd)-Click: Toggle-select the secondary/tertiary focus units.
         - Alt(Opt)-Click: Add or remove unit in clicked row to the multi-unit edit selection set for relabel/delete op.
           However, if cell is in "Label" column, initiate in-place edit of that unit label.
         - Shift-Click: Extend multi-unit edit selection set with all units between the clicked unit and the last unit
           added to the selection set.
         - Shift+Alt-Click: Clear the edit selection set.

        :param index: Locates the table cell that was clicked.
        """
        uid = self.model.row_to_unit(index.row())
        mod = QGuiApplication.keyboardModifiers()
        if isinstance(uid, str):
            if mod in [Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ControlModifier]:
                self._data_manager.update_neurons_with_display_focus(uid, mod == Qt.KeyboardModifier.NoModifier)
            elif mod == Qt.KeyboardModifier.AltModifier:
                if index.column() == _NeuronTableModel.LABEL_COL_IDX:
                    self.setCurrentIndex(index)
                    self.edit(index)
                else:
                    self.model.toggle_select_row(index.row())
            elif mod == Qt.KeyboardModifier.ShiftModifier:
                self.model.select_contiguous_range(index.row())
            elif mod == Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.AltModifier:
                self.model.clear_current_selection()

    @Slot(QPoint)
    def _on_header_right_click(self, pos: QPoint) -> None:
        """
         Handler raises the context menu by which user toggles the visiblity of selected columns in the table.
        :param pos: Mouse cursor position (in local widget coordinates).
        """
        pos = self.horizontalHeader().mapToGlobal(pos)
        pos += QPoint(5, 10)
        self._table_context_menu.move(pos)
        self._table_context_menu.show()

    @Slot(int, Qt.SortOrder)
    def _on_sort_indicator_changed(self, col: int, _: Qt.SortOrder):
        """
        Overridden to restore sort indicator whenever user attempts to sort on the 'Similarity' column, which
        is not permitted. **NOTE**: The model ignores any attempt to sort on that column, but that does not prevent
        the table header from updating the sort indicator -- so we need to fix it here.
        """
        if col == _NeuronTableModel.SIM_COL_IDX:
            col, order = self.model.current_sort_column_and_order
            self.horizontalHeader().setSortIndicator(col, order)

    def _toggle_table_column_visibility(self, col: int) -> None:
        """
        Toggle the visibility of the specified column in the neural units table. No action is taken if the specified
        column may not be hidden.
        :param col: The table column index.
        """
        if col in self.model.hideable_columns():
            hidden = self.isColumnHidden(col)
            if hidden:
                self.showColumn(col)
            else:
                self.hideColumn(col)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """
        Filters out any tooltip events on the table EXCEPT those on a table cell containing a unit label that
        does not fit within the cell.
        """
        if (event.type() == QEvent.Type.ToolTip) and isinstance(event, QHelpEvent):
            help_event: QHelpEvent = event
            index = self.indexAt(help_event.pos())
            if index.isValid() and (index.column() == _NeuronTableModel.LABEL_COL_IDX):
                label = self.model.data(index)
                label_w = self.fontMetrics().tightBoundingRect(label).width()
                cell_w = self.visualRect(index).width()
                return label_w < cell_w   # eat the event if label fits
            return True  # or if cell does not contain a unit label

        return super(_NeuronTableView, self).eventFilter(obj, event)

    def on_unit_labels_changed(self, uids: List[str]) -> None:
        """
        Whenever one or more neural units is relabeled, refresh the underlying table model and add any new labels to the
        set of "suggestions" for the auto-completion in-place editor that handles label edits.
        :param uids: List of UIDs of the affected neural units.
        """
        n_found = 0
        for u in self._data_manager.neurons:
            if u.uid in uids:
                self.add_suggested_unit_label(u.label)
                n_found += 1
                if n_found == len(uids):
                    break
        self.model.on_unit_labels_changed(uids)
        self.model.clear_current_selection()   # always clear selection after a (possibly multi-unit) relabel.

    def reload(self, clear_selection: bool = False) -> None:
        """
        Reload neural unit table to reflect a change in the current display/focus list, or a wholesale change in
        content because the working directory has changed.
        :param clear_selection: If True, clear the current selection state. Default is False. Always set this flag if
             the working directory has just changed.
        """
        self.model.reload_table_data(clear_selection)


class NeuronView(BaseView):
    """
    A tabular view of the list of neurons exposed by the data manager object, :class:`Analyzer`. Each row in the
    table represents one neural unit, with the unit UID, optional label and various numerical statistics shown in the
    columns. The table may be sorted on any column, in ascending or descending order. The visibility of any column
    except **UID*** may be toggled via a context menu raised when the user clicks anywhere on the column header.
        The user single-selects a neuron for the display focus by clicking on the corresponding row, and adds a neuron
    to the display list by clicking on its row while holding down any modifier key. Contiguous range selection is not
    supported. The display focus list may contain up to MAX_NUM_FOCUS_NEURONS (3), and a unique color is assigned to
    each slot in that list. Most other views in XSort display data for neurons in the current display list, using the
    assigned colors.
        **NOTE**: In a previous version, the QTableView was too slow to "repaint" whenever there was any layout
    change. Performance was dramatically improved by fixing the row height and the column widths.
    """

    def __init__(self, data_manager: Analyzer, settings: QSettings) -> None:
        super().__init__('Neurons', None, data_manager, settings)
        self.table_view = _NeuronTableView(data_manager, settings)
        """ Table view displaying the neuron table. """

        self._highlight_chk = QCheckBox("Highlight up to 5 units most similar to the focussed unit (blue)")
        """ Checkbox toggles highlighting of 5 units most similar to primary focus neuron. """

        self._count_label = QLabel("Count = 0")
        """ Static label indicating how many units are listed in neural units table. """
        self._count_label.setVisible(False)

        self._highlight_chk.setChecked(False)
        self._highlight_chk.stateChanged.connect(self._on_highlight_changed)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.table_view, stretch=1)
        control_line = QHBoxLayout()
        control_line.addWidget(self._count_label)
        control_line.addStretch(1)
        control_line.addWidget(self._highlight_chk, alignment=Qt.AlignmentFlag.AlignRight)
        main_layout.addLayout(control_line)
        self.view_container.setLayout(main_layout)

    def uid_of_unit_below(self, uid: str) -> Optional[str]:
        """
        Get the UID of the unit in the row below (or above, if necessary) the table row that displays the unit
        specified. The order of the rows depends on the current sort order.

        :param uid: The UID of a neural unit
        :return: UID of unit displayed in the row below or above that unit, or None if not found.
        """
        return self.table_view.model.find_uid_of_neighboring_unit(uid)

    @property
    def edit_selection_set(self) -> Set[str]:
        """ UIDs comprising the multi-unit edit selection set in this view."""
        return self.table_view.model.units_in_current_selection

    @Slot(int)
    def _on_highlight_changed(self, _: int) -> None:
        """ Handler updates the neuron table view whenever user toggles the 'highlight similar units' check box. """
        self.table_view.model.highlight_similar = self._highlight_chk.isChecked()

    def on_working_directory_changed(self) -> None:
        self.table_view.reload(clear_selection=True)
        self._update_count_readout()

    def on_neuron_metrics_updated(self, uid: str) -> None:
        self.table_view.reload()

    def on_focus_neurons_changed(self, _: bool) -> None:
        self.table_view.reload()
        self._update_count_readout()

    def on_neuron_labels_updated(self, uids: List[str]) -> None:
        self.table_view.on_unit_labels_changed(uids)

    def on_units_deleted(self) -> None:
        """ Reload the table after one or more neural units were deleted, but the current focus list was unaffected. """
        self.table_view.reload(clear_selection=True)
        self._update_count_readout()

    def _update_count_readout(self) -> None:
        n = len(self.data_manager.neurons)
        self._count_label.setText(f"Count = {n}")
        self._count_label.setVisible(n > 30)

    def save_settings(self) -> None:
        """
        Preserves which columns in the neural units table have been hidden by the user, and whether or not to
        highlight units most simlar to the current primary focus unit.
        """
        self.settings.setValue('neuronview_hidden_cols', self.table_view.hidden_columns)
        self.settings.setValue('neuronview_highlight_similar', self.table_view.model.highlight_similar)

    def restore_settings(self) -> None:
        """
        IAW user settings: (1) hide select columns in the neural units table; and (2) enable/disable highlighting
        of the neural units most similar to the current primary focus unit.
        """
        self.table_view.hidden_columns = self.settings.value('neuronview_hidden_cols', '')
        highlight: bool = (self.settings.value('neuronview_highlight_similar', defaultValue="false") == "true")
        self._highlight_chk.setChecked(highlight)  # triggers signal if state changed
