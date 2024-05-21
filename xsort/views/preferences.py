from typing import Set

from PySide6.QtCore import Slot, QSettings, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QDialog, QGroupBox, QGridLayout, QRadioButton, QListWidget, QLineEdit, QDialogButtonBox, \
    QPushButton, QHBoxLayout, QVBoxLayout

from xsort.data.neuron import Neuron


class PreferencesDlg(QDialog):
    """
    Customized dialog that exposes select application preferences that are persisted in the user's XSort ssttings file:
     - Internal cache deletion.
     - Set of suggested unit labels (for auto-completion support when editing a label).
    """
    def __init__(self, settings: QSettings, parent=None) -> None:
        super(PreferencesDlg, self).__init__(parent=parent)
        self._settings = settings
        """ User settings. """
        self._suggested_unit_labels: Set[str] = set()
        """ The set of suggested labels for a neural unit (for autocompletion). """

        self.setWindowTitle('XSort Settings')

        self._del_cache_never = QRadioButton("Never")
        self._del_cache_never.setToolTip("Never delete any internal cache files")
        self._del_cache_lru = QRadioButton("LRU")
        self._del_cache_lru.setToolTip("Delete a working directory's internal cache when it is removed from the most "
                                       "recently used list")
        self._del_cache_always = QRadioButton("Always")
        self._del_cache_always.setToolTip("Always delete a working directory's internal cache at application exit or "
                                          "after switching to a different directory")

        del_cache_grp = QGroupBox("Internal cache removal policy (choose one)")
        grp_layout = QGridLayout()
        grp_layout.addWidget(self._del_cache_never, 0, 0)
        grp_layout.addWidget(self._del_cache_lru, 1, 0)
        grp_layout.addWidget(self._del_cache_always, 2, 0)
        del_cache_grp.setLayout(grp_layout)

        self._labels_list = QListWidget()
        self._labels_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._labels_list.setSortingEnabled(True)
        self._labels_list.setEditTriggers(QListWidget.EditTrigger.NoEditTriggers)
        self._labels_list.itemSelectionChanged.connect(self._on_label_item_selected)
        self._labels_list.setMinimumSize(200, 300)

        self._label_edit = QLineEdit()
        self._label_edit.textChanged.connect(self._on_label_edit_text_changed)

        # hitting return in the line edit is same as clicking "Add" (but had to disable default dialog button!)
        self._label_edit.returnPressed.connect(lambda: self._on_add_unit_label(False))

        self._add_btn = QPushButton("Add")
        self._add_btn.setEnabled(False)
        self._add_btn.clicked.connect(self._on_add_unit_label)
        self._rmv_btn = QPushButton("Del")
        self._rmv_btn.setEnabled(False)
        self._rmv_btn.clicked.connect(self._on_remove_unit_label)

        h_layout = QHBoxLayout()
        h_layout.addWidget(self._label_edit, stretch=1)
        h_layout.addWidget(self._add_btn)
        h_layout.addWidget(self._rmv_btn)

        unit_labels_grp = QGroupBox("Suggested labels for a neural unit")
        grp_layout = QGridLayout()
        grp_layout.addWidget(self._labels_list, 0, 0)
        grp_layout.addLayout(h_layout, 1, 0)
        unit_labels_grp.setLayout(grp_layout)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self._save_changes_and_accept)
        btn_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(del_cache_grp)
        layout.addSpacing(5)
        layout.addWidget(unit_labels_grp, stretch=1)
        layout.addSpacing(5)
        layout.addWidget(btn_box)
        self.setLayout(layout)

    def keyPressEvent(self, evt: QKeyEvent):
        """ Overridden to prevent dialog from closing when user hits Return key. """
        if evt.key() == Qt.Key.Key_Enter or evt.key() == Qt.Key.Key_Return:
            return
        super(PreferencesDlg, self).keyPressEvent(evt)

    def exec(self) -> None:
        """ The dialog widgets are reloaded IAW current user settings prior to raising the dialog. """
        self._load_current_settings()
        super(PreferencesDlg, self).exec()

    def _load_current_settings(self) -> None:
        """ Load dialog widgets IAW the user's current preferences. """
        self._suggested_unit_labels.clear()
        labels = self._settings.value('suggested_unit_labels', '', type=str).split(',')
        self._suggested_unit_labels.clear()
        self._suggested_unit_labels.update(labels)
        self._labels_list.clear()
        self._labels_list.addItems(labels)
        self._label_edit.clear()

        policy = self._settings.value('del_cache_policy', None)
        if isinstance(policy, str):
            if policy == "Always":
                self._del_cache_always.setChecked(True)
            elif policy == "LRU":
                self._del_cache_lru.setChecked(True)
            else:
                self._del_cache_never.setChecked(True)

    @Slot()
    def _on_label_item_selected(self) -> None:
        try:
            label_text = self._labels_list.selectedItems()[0].text()
            self._label_edit.setText(label_text)
        except Exception:
            pass

    @Slot(str)
    def _on_label_edit_text_changed(self, text: str) -> None:
        exists = text in self._suggested_unit_labels
        valid = Neuron.is_valid_label(text) and (len(text.strip()) == len(text)) and (len(text) > 0)
        self._add_btn.setEnabled(valid and not exists)
        self._rmv_btn.setEnabled(exists)

    def _on_add_unit_label(self, _: bool) -> None:
        """ Add proposed label, if valid, to the list of suggested neural unit labels."""
        s = self._label_edit.text()
        if ((len(s) > 0) and (len(s.strip()) == len(s)) and Neuron.is_valid_label(s)
                and not (s in self._suggested_unit_labels)):
            self._suggested_unit_labels.add(s)
            self._labels_list.addItems([s])
            self._label_edit.clear()

    def _on_remove_unit_label(self, _: bool) -> None:
        """ Delete suggested unit label, if it is in the current list of all suggested labels. """
        s = self._label_edit.text()
        if s in self._suggested_unit_labels:
            self._suggested_unit_labels.remove(s)
            row = self._labels_list.currentRow()
            self._labels_list.clear()
            self._labels_list.addItems(sorted(self._suggested_unit_labels))
            self._label_edit.clear()
            n_rows = self._labels_list.count()
            if n_rows > 0:
                row = row if (isinstance(row, int) and (0 <= row < n_rows)) else 0
                self._labels_list.setCurrentRow(row)  # the label edit will be updated accordingly

    @Slot()
    def _save_changes_and_accept(self) -> None:
        unit_labels: Set[str] = set()
        for row in range(self._labels_list.count()):
            unit_labels.add(self._labels_list.item(row).text())

        policy = None
        if self._del_cache_never.isChecked():
            policy = "Never"
        elif self._del_cache_lru.isChecked():
            policy = "LRU"
        elif self._del_cache_always.isChecked():
            policy = "Always"

        # TODO: Testing
        self._settings.setValue('suggested_unit_labels', ','.join(sorted(unit_labels)))
        if policy is not None:
            self._settings.setValue('del_cache_policy', policy)

        self.accept()
