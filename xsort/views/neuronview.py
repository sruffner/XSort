from typing import List, Any, Optional, Set, Tuple

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, QPoint, QSettings, Slot, QObject, QEvent
from PySide6.QtGui import QColor, QAction, QGuiApplication, QFontMetricsF, QContextMenuEvent, QKeyEvent, QHelpEvent
from PySide6.QtWidgets import QTableView, QHeaderView, QHBoxLayout, QSizePolicy, QMenu, QStyledItemDelegate, QWidget, \
    QLineEdit, QCompleter

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

    LABEL_COL_IDX = 8
    """ Index of the 'Label' column -- unit labels may be edited. """
    SIM_COL_IDX = 7
    """ Index of the 'Similarity' column -- sorting on this column is not permitted. """

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
        Set of UIDs identifying units selected by user - NOT necessarily part of display/focus list, but to
        mediate a multi-unit delete or relabel operation. 
        """
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
        else:
            self._selected_uids.add(uid)
        top_left = self.createIndex(row, 0)
        bot_right = self.createIndex(row, self.LABEL_COL_IDX)
        self.dataChanged.emit(top_left, bot_right, Qt.ItemDataRole.BackgroundRole)

    def select_contiguous_range(self, end_row: int) -> None:
        """
        Select a contiguous range of rows in the table from the current **singly-selected** row to the row specified.
        If the number of rows currently selected is not exactly one, no action is taken.
        :param end_row: The row at which to end the continuous-range selection. No action taken if invalid.
        """
        end_row_uid = self.row_to_unit(end_row)
        if (end_row_uid is not None) and (len(self._selected_uids) == 1) and not (end_row_uid in self._selected_uids):
            start_row = self.unit_to_row(next(iter(self._selected_uids)))
            inc = 1 if start_row < end_row else -1
            r = start_row + inc
            while True:
                uid = self.row_to_unit(r)
                if uid is not None:
                    self._selected_uids.add(uid)
                if r == end_row:
                    break
                r = r + inc
            top_left = self.createIndex(min(start_row, end_row), 0)
            bot_right = self.createIndex(max(start_row, end_row), self.LABEL_COL_IDX)
            self.dataChanged.emit(top_left, bot_right, Qt.ItemDataRole.BackgroundRole)

    def clear_current_selection(self) -> None:
        """ Clear the set of currently selected table rows, if any. """
        if len(self._selected_uids) > 0:
            self._selected_uids.clear()
            top_left = self.createIndex(0, 0)
            bot_right = self.createIndex(self.rowCount() - 1, self.columnCount() - 1)
            self.dataChanged.emit(top_left, bot_right, Qt.ItemDataRole.BackgroundRole)

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
                u = self._data_manager.neurons[idx].uid
                color_str = self._data_manager.display_color_for_neuron(u)
                bkg_color = None if color_str is None else QColor.fromString(color_str)
                # when unit is "selected", the background for the UID cell is light gray if the unit is also one of
                # the focus units. If not, the entire row is light gray
                if (u in self._selected_uids) and ((bkg_color is None) or (c == 0)):
                    bkg_color = QColor(Qt.GlobalColor.lightGray)
                if role == Qt.BackgroundRole:
                    return bkg_color
                else:
                    return None if color_str is None else QColor(Qt.black if bkg_color.lightness() < 140 else Qt.white)
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
        self._model = _NeuronTableModel(data_manager)
        """ Neuron table model (essentially wraps a table model around the data manager's neuron list). """
        self._table_context_menu = QMenu(self)
        """ Context menu for table header to hide/show selected table columns. """
        self._label_col_delegate = _UnitLabelColumnDelegate(self)

        self.setModel(self._model)

        # allow sorting on any column except Similarity column. Ensure table view is initially sorted by the UID
        # column (0) in ascending order
        self.setSortingEnabled(True)
        self._model.sort(0)  # to ensure table view is initially sorted by column 0 in ascending order
        self.horizontalHeader().setSortIndicator(0, Qt.SortOrder.AscendingOrder)
        self.horizontalHeader().sortIndicatorChanged.connect(self._on_sort_indicator_changed)

        self._model.layoutChanged.connect(self._on_table_layout_change)

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

        self.clicked.connect(self._on_row_clicked)   # to trigger update of current display list

        # set up context menu for toggling the visibility of selected columns in the table.
        for col in self._model.hideable_columns():
            action = QAction(self._model.headerData(col, Qt.Horizontal), parent=self._table_context_menu,
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
        return ','.join([str(i) for i in self._model.hideable_columns() if self.isColumnHidden(i)])

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
            if col in self._model.hideable_columns():
                self.hideColumn(col)
                hidden_column_labels.append(self._model.headerData(col, Qt.Horizontal))
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
            next_idx = self._model.createIndex(index.row() + incr, index.column())
            if next_idx.isValid():
                self.setCurrentIndex(next_idx)
                self.edit(next_idx)
        else:
            super().closeEditor(editor, hint)

    def contextMenuEvent(self, event: QContextMenuEvent):
        """
        Overridden to repurpose the context menu event (right-click) to trigger an in-place edit of the label of any
        neural unit or to toggle the selection state of a unit.

        An in-place label edit is initiated if the right-click is over a table cell containing a unit label. Otherwise,
        the set of table rows selected is updated as follows:
         - Ctrl/Command key down: Clear the selection.
         - Shift key down: If a single row is currently selected, extend the selection to the row under the mouse
           (contiguous range selection).
         - No modifier key: Toggle the selection state of the row under the mouse.

        :param event: The event object -- used to obtain the index of the table cell under the mouse.
        """
        index = self.indexAt(event.pos())
        is_ctrl = (event.modifiers() & Qt.KeyboardModifier.ControlModifier) == Qt.KeyboardModifier.ControlModifier
        is_shift = (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) == Qt.KeyboardModifier.ShiftModifier
        if (index.column() == _NeuronTableModel.LABEL_COL_IDX) and not (is_ctrl or is_shift):
            self.setCurrentIndex(index)
            self.edit(index)
        elif is_ctrl:
            self._model.clear_current_selection()
        elif is_shift:
            self._model.select_contiguous_range(index.row())
        else:
            self._model.toggle_select_row(index.row())

    def keyReleaseEvent(self, event: QKeyEvent):
        """
        Handler implements a simple keyboard interface for changing the identity of the primary neural unit: pressing
        the Up/Down-arrow key shifts the primary unit focus to the unit in the table row above/below the row that
        represents the current primary unit (no "wrap-around"). If there are other units in the display list, they are
        removed.
        :param event: The key event -- only the Up and Down arrow keys are handled. Any other key is passed on to the
            base class implementation.
        """
        inc = 1 if event.key() == Qt.Key.Key_Down else (-1 if event.key() == Qt.Key.Key_Up else 0)
        is_ctrl = (event.modifiers() & Qt.KeyboardModifier.ControlModifier) == Qt.KeyboardModifier.ControlModifier
        if inc != 0 and len(self._data_manager.neurons) > 0:
            # if Ctrl key depressed, shift secondary unit up/down, leaving primary unit unchanged. If a tertiary unit
            # was selected, it is deselected. If focus list empty, select units in first two rows.
            if is_ctrl:
                curr_focus = self._data_manager.neurons_with_display_focus
                if len(curr_focus) == 0:
                    primary_uid = self._model.row_to_unit(0)
                    secondary_uid = self._model.row_to_unit(1)
                elif len(curr_focus) == 1:
                    primary_uid = curr_focus[0].uid
                    secondary_uid = self._model.row_to_unit(self._model.unit_to_row(primary_uid) + inc)
                else:
                    primary_uid = curr_focus[0].uid
                    row_1 = self._model.unit_to_row(primary_uid)
                    row_2 = self._model.unit_to_row(curr_focus[1].uid) + inc
                    if row_2 == row_1:
                        row_2 = row_2 + inc
                    secondary_uid = self._model.row_to_unit(row_2)
                if isinstance(primary_uid, str):
                    self._data_manager.update_neurons_with_display_focus([primary_uid, secondary_uid])
                    row = self._model.unit_to_row(secondary_uid if isinstance(secondary_uid, str) else primary_uid)
                    self.scrollTo(self._model.createIndex(row, 0))
                return

            # if Ctrl key not depressed, shift primary unit up/down and remove any other units from focus list.
            if self._data_manager.primary_neuron is None:
                row = 0 if (len(self._data_manager.neurons) > 0) else -1
            else:
                curr_row = self._model.unit_to_row(self._data_manager.primary_neuron.uid)
                row = (curr_row + inc) if isinstance(curr_row, int) else -1

            uid = self._model.row_to_unit(row)
            if isinstance(uid, str):
                self._data_manager.update_neurons_with_display_focus(uid, True)
                self.scrollTo(self._model.createIndex(row, 0))
        else:
            super().keyReleaseEvent(event)

    @Slot()
    def _on_table_layout_change(self) -> None:
        """
        Whenever the layout of the underlying table changes, scroll if necessary to ensure the row corresponding to
        the current primary unit (if any) is visible.
        """
        if self._data_manager.primary_neuron is not None:
            row = self._model.unit_to_row(self._data_manager.primary_neuron.uid)
            if row is not None:
                self.scrollTo(self._model.createIndex(row, 0))

    @Slot(QModelIndex)
    def _on_row_clicked(self, index: QModelIndex) -> None:
        """
        Whenever the user left-clicks on any table cell, the neural unit in the corresponding row is added to the
        current display list, clearing the previous display list UNLESS a modifier key is held down (doesn't matter
        which key -- contiguous range selection isn't allowed).
        :param index: Locates the table cell that was clicked.
        """
        uid = self._model.row_to_unit(index.row())
        clear_previous_selection = (QGuiApplication.keyboardModifiers() == Qt.KeyboardModifier.NoModifier)
        if isinstance(uid, str):
            self._data_manager.update_neurons_with_display_focus(uid, clear_previous_selection)

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
            col, order = self._model.current_sort_column_and_order
            self.horizontalHeader().setSortIndicator(col, order)

    def _toggle_table_column_visibility(self, col: int) -> None:
        """
        Toggle the visibility of the specified column in the neural units table. No action is taken if the specified
        column may not be hidden.
        :param col: The table column index.
        """
        if col in self._model.hideable_columns():
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
                label = self._model.data(index)
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
        self._model.on_unit_labels_changed(uids)
        self._model.clear_current_selection()   # always clear selection after a (possibly multi-unit) relabel.

    def reload(self, clear_selection: bool = False) -> None:
        """
        Reload neural unit table to reflect a change in the current display/focus list, or a wholesale change in
        content because the working directory has changed.
        :param clear_selection: If True, clear the current selection state. Default is False. Always set this flag if
             the working directory has just changed.
        """
        self._model.reload_table_data(clear_selection)


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
        self._table_view = _NeuronTableView(data_manager, settings)
        """ Table view displaying the neuron table. """

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
        return self._table_view.model().find_uid_of_neighboring_unit(uid)

    def on_working_directory_changed(self) -> None:
        self._table_view.reload(clear_selection=True)

    def on_neuron_metrics_updated(self, uid: str) -> None:
        self._table_view.reload()

    def on_focus_neurons_changed(self, _: bool) -> None:
        self._table_view.reload()

    def on_neuron_labels_updated(self, uids: List[str]) -> None:
        self._table_view.on_unit_labels_changed(uids)

    def save_settings(self) -> None:
        """ Preserves which columns in the neural units table have been hidden by the user. """
        self.settings.setValue('neuronview_hidden_cols', self._table_view.hidden_columns)

    def restore_settings(self) -> None:
        """ Hides select columns in the neural units table IAW user settings. """
        self._table_view.hidden_columns = self.settings.value('neuronview_hidden_cols', '')
