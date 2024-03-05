import sys
import time
from pathlib import Path
from typing import Optional, List

from PySide6.QtCore import QStandardPaths, Slot, QCoreApplication
from PySide6.QtGui import QCloseEvent, Qt
from PySide6.QtWidgets import QMainWindow, QPushButton, QApplication, QVBoxLayout, QLabel, QFileDialog, QWidget, \
    QHBoxLayout, QMessageBox, QTextEdit, QProgressDialog

from xsort.data.files import WorkingDirectory
from xsort.data.neuron import Neuron
from xsort.data.taskmanager import TaskManager
from xsort.data.taskfunc import delete_internal_cache_files


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.task_manager = TaskManager()
        self.task_manager.progress.connect(self._on_task_progress_update)
        self.task_manager.error.connect(self._on_task_error)
        self.task_manager.ready.connect(self._on_task_mgr_ready)

        self.work_dir: Optional[WorkingDirectory] = None
        self.units: List[Neuron] = list()

        self.t_cache_start: float = 0.0

        self.setWindowTitle("Testing TaskManager")

        self.select_btn = QPushButton("Select")
        self.select_btn.clicked.connect(self._on_select_directory)

        self.work_dir_label = QLabel("<None>")

        self.build_btn = QPushButton("Build Cache")
        self.build_btn.setEnabled(False)
        self.build_btn.clicked.connect(self._on_start_or_cancel_build)

        self.clear_btn = QPushButton("Clear Cache")
        self.clear_btn.setEnabled(False)
        self.clear_btn.clicked.connect(self._on_clear_cache)

        self.progress_edit = QTextEdit()
        self.progress_edit.setReadOnly(True)

        self.progress_dlg = QProgressDialog("Please wait...", "", 0, 100, self)
        """ A modal progress dialog used to block further user input when necessary. """

        # customize progress dialog: modal, no cancel, no title bar (so you can't close it)
        self.progress_dlg.setMinimumDuration(500)
        self.progress_dlg.setCancelButton(None)
        self.progress_dlg.setModal(True)
        self.progress_dlg.setAutoClose(False)
        self.progress_dlg.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        self.progress_dlg.close()  # if we don't call this, the dialog will appear shortly after app startup

        main_layout = QVBoxLayout()
        control_line = QHBoxLayout()
        control_line.addWidget(self.select_btn)
        control_line.addWidget(self.work_dir_label, stretch=1)
        control_line.addWidget(self.build_btn)
        control_line.addWidget(self.clear_btn)
        main_layout.addLayout(control_line)
        main_layout.addWidget(self.progress_edit, stretch=1)

        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)
        self.setMinimumSize(800, 600)

    def quit(self) -> None:
        res = QMessageBox.question(self, "Exit", "Are you sure you want to quit?")
        if res == QMessageBox.StandardButton.Yes:
            self.task_manager.shutdown(self.progress_dlg)
            QCoreApplication.instance().exit(0)

    def closeEvent(self, event: QCloseEvent) -> None:
        """ [QWidget override] Closing the main window quits the application -- unless the user vetoes the quit. """
        self.quit()
        event.ignore()

    def _refresh_state(self) -> None:
        valid_dir = isinstance(self.work_dir, WorkingDirectory) and self.work_dir.is_valid
        self.select_btn.setEnabled(not self.task_manager.busy)
        self.build_btn.setEnabled(valid_dir)
        self.build_btn.setText("Cancel" if self.task_manager.busy else "Build Cache")
        self.clear_btn.setEnabled(valid_dir and not self.task_manager.busy)

    @Slot(bool)
    def _on_select_directory(self, _: bool):
        if isinstance(self.work_dir, WorkingDirectory):
            parent_path = self.work_dir.path.parent
        else:
            parent_path = Path(QStandardPaths.writableLocation(QStandardPaths.HomeLocation))
        folder_str = QFileDialog.getExistingDirectory(self, "Select working directory for XSort",
                                                      str(parent_path.absolute()))
        if len(folder_str) > 0:
            emsg, work_dir = WorkingDirectory.load_working_directory(Path(folder_str), self)
            if work_dir is None:
                if len(emsg) == 0:
                    self.progress_edit.append("User cancelled.")
                else:
                    self.progress_edit.append(f"ERROR: {emsg}")
            else:
                # success
                self.work_dir = work_dir
                _, self.units = self.work_dir.load_neural_units()
                self.progress_edit.append(f"Valid working directory: {self.work_dir.num_analog_channels()} analog "
                                          f"data channels and {len(self.units)} neural units.")
                self.work_dir_label.setText(folder_str)

        self._refresh_state()

    @Slot(bool)
    def _on_clear_cache(self, _: bool) -> None:
        if (not self.task_manager.busy) and isinstance(self.work_dir, WorkingDirectory) and self.work_dir.is_valid:
            delete_internal_cache_files(self.work_dir)
            self.progress_edit.append("Removed all internal cache files from the current working directory.")

    @Slot(bool)
    def _on_start_or_cancel_build(self, _: bool) -> None:
        if not (isinstance(self.work_dir, WorkingDirectory) and self.work_dir.is_valid):
            return
        if not self.task_manager.busy:
            task_started = self.task_manager.build_internal_cache_if_necessary(
                self.work_dir, [u.uid for u in self.units])
            if not task_started:
                self.progress_edit.append("Cache build not required.")
            else:
                self.t_cache_start = time.perf_counter()
                self.progress_edit.append("Building internal cache...")
                self._refresh_state()
        else:
            self.task_manager.cancel_all_tasks(self.progress_dlg)

    @Slot(str, int)
    def _on_task_progress_update(self, msg: str, pct: int) -> None:
        self.progress_edit.append(f"{msg}: {pct}%")

    def _on_task_error(self, emsg: str) -> None:
        self.progress_edit.append(f"==> ERROR: {emsg}")

    def _on_task_mgr_ready(self) -> None:
        self.progress_edit.append(f"\nReady. Task time = {time.perf_counter() - self.t_cache_start:.3f} seconds.")
        self._refresh_state()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()
