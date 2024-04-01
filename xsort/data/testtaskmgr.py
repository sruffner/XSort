import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from PySide6.QtCore import QStandardPaths, Slot, QCoreApplication
from PySide6.QtGui import QCloseEvent, Qt, QIntValidator
from PySide6.QtWidgets import QMainWindow, QPushButton, QApplication, QVBoxLayout, QLabel, QFileDialog, QWidget, \
    QHBoxLayout, QMessageBox, QTextEdit, QProgressDialog, QLineEdit

from xsort.data.files import WorkingDirectory
from xsort.data.neuron import Neuron, DataType, ChannelTraceSegment
from xsort.data.taskmanager import TaskManager
from xsort.data.taskfunc import delete_internal_cache_files


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.task_manager = TaskManager()
        self.task_manager.progress.connect(self._on_task_progress_update)
        self.task_manager.error.connect(self._on_task_error)
        self.task_manager.ready.connect(self._on_task_mgr_ready)
        self.task_manager.data_available.connect(self._on_data_available)

        self.work_dir: Optional[WorkingDirectory] = None
        self.units: List[Neuron] = list()

        self.t_task_start: float = 0.0

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

        self.test_btn = QPushButton("Run Test")
        self.test_btn.setEnabled(False)
        self.test_btn.clicked.connect(self._on_run_test)

        self.get_traces_btn = QPushButton("Get Channel Traces")
        self.get_traces_btn.setEnabled(False)
        self.get_traces_btn.clicked.connect(self._on_get_traces)

        first_ch_label = QLabel("First:")
        self._first_ch_edit = QLineEdit()
        self._first_ch_edit.setValidator(QIntValidator(0, 999, self._first_ch_edit))
        self._first_ch_edit.setText('0')
        self._first_ch_edit.textEdited.connect(lambda _: self._refresh_state())
        num_ch_label = QLabel(" N: ")
        self._num_ch_edit = QLineEdit()
        self._num_ch_edit.setValidator(QIntValidator(1, 999, self._num_ch_edit))
        self._num_ch_edit.setText('0')
        self._num_ch_edit.textEdited.connect(lambda _: self._refresh_state())
        t0_label = QLabel("t0 (sec):")
        self._t0_edit = QLineEdit()
        self._t0_edit.setValidator(QIntValidator(0, 999, self._t0_edit))
        self._t0_edit.setText('0')
        self._t0_edit.textEdited.connect(lambda _: self._refresh_state())
        dur_label = QLabel("dur (sec):")
        self._dur_edit = QLineEdit()
        self._dur_edit.setValidator(QIntValidator(1, 999, self._dur_edit))
        self._dur_edit.setText('0')
        self._dur_edit.textEdited.connect(lambda _: self._refresh_state())

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
        control_line.addWidget(self.test_btn)
        main_layout.addLayout(control_line)

        control_line_2 = QHBoxLayout()
        control_line_2.addWidget(self.get_traces_btn)
        control_line_2.addWidget(first_ch_label)
        control_line_2.addWidget(self._first_ch_edit)
        control_line_2.addWidget(num_ch_label)
        control_line_2.addWidget(self._num_ch_edit)
        control_line_2.addWidget(t0_label)
        control_line_2.addWidget(self._t0_edit)
        control_line_2.addWidget(dur_label)
        control_line_2.addWidget(self._dur_edit)
        main_layout.addLayout(control_line_2)

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
        self.test_btn.setEnabled(valid_dir and not self.task_manager.busy)
        self.get_traces_btn.setEnabled(
            valid_dir and isinstance(self._get_args_for_get_traces(), tuple) and (not self.task_manager.busy))

    def _get_args_for_get_traces(self) -> Optional[Tuple]:
        out: Optional[Tuple] = None
        try:
            n_ch = self.work_dir.num_analog_channels()
            rec_dur = self.work_dir.analog_channel_recording_duration_seconds
            first_ch = int(self._first_ch_edit.text())
            n = int(self._num_ch_edit.text())
            t0 = int(self._t0_edit.text())
            dur = int(self._dur_edit.text())
            samples_per_sec = self.work_dir.analog_sampling_rate
            if (0 <= first_ch < n_ch) and (0 < n_ch) and (0 <= t0 < rec_dur) and (0 < dur):
                out = (first_ch, n, t0*samples_per_sec, dur*samples_per_sec)
        except Exception:
            out = None
        return out

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
                min_spks = min([u.num_spikes for u in self.units])
                max_spks = max([u.num_spikes for u in self.units])
                total_spks = sum([u.num_spikes for u in self.units])
                median_spks = np.median([u.num_spikes for u in self.units])
                self.progress_edit.append(f"Spike count: min={min_spks}, max={max_spks}, "
                                          f"median={median_spks}, total={total_spks}")
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
                self.t_task_start = time.perf_counter()
                self.progress_edit.append("Building internal cache...")
                self._refresh_state()
        else:
            self.task_manager.cancel_all_tasks(self.progress_dlg)

    @Slot(bool)
    def _on_run_test(self, _: bool) -> None:
        if (not self.task_manager.busy) and isinstance(self.work_dir, WorkingDirectory) and self.work_dir.is_valid:
            self.task_manager.run_performance_test(self.work_dir)
            self.t_task_start = time.perf_counter()
            self.progress_edit.append("Running performance test...")
            self._refresh_state()

    @Slot(bool)
    def _on_get_traces(self, _: bool) -> None:
        params = self._get_args_for_get_traces()
        if self.task_manager.busy or (params is None):
            return
        self.t_task_start = time.perf_counter()
        self.task_manager.get_channel_traces(self.work_dir, params[0], params[1], params[2], params[3])

    @Slot(str, int)
    def _on_task_progress_update(self, msg: str, pct: int) -> None:
        self.progress_edit.append(f"{msg}: {pct}%")

    @Slot(DataType, object)
    def _on_data_available(self, data_type: DataType, data: object) -> None:
        if data_type == DataType.CHANNELTRACE:
            if isinstance(data, ChannelTraceSegment):
                seg: ChannelTraceSegment = data
                self.progress_edit.append(f"Got channel trace segment for channel {seg.channel_index}, "
                                          f"dur={seg.duration:.1f}")
            else:
                self.progress_edit.append(f"Bad data object returned, should be ChannelTraceSegment")

    def _on_task_error(self, emsg: str) -> None:
        self.progress_edit.append(f"==> ERROR: {emsg}")

    def _on_task_mgr_ready(self) -> None:
        self.progress_edit.append(f"\nReady. Task time = {time.perf_counter() - self.t_task_start:.3f} seconds.")
        self._refresh_state()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()
