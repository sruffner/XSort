import pickle
import time
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple, Any, Set

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, QTimer, QPointF
from PySide6.QtCore import Qt
from PySide6.QtGui import QPolygonF
from PySide6.QtWidgets import QMainWindow, QProgressDialog, QFileDialog, QMessageBox

from xsort.data.neuron import Neuron, ChannelTraceSegment, DataType, MAX_CHANNEL_TRACES
from xsort.data.taskmanager import TaskManager
from xsort.data.files import WorkingDirectory, UserEdit


class Analyzer(QObject):
    """
    The data model manager object for XSort.
    """
    MAX_NUM_FOCUS_NEURONS: int = 3
    """ The maximum number of neural units that can be selected simultaneously for display focus. """
    FOCUS_NEURON_COLORS: List[str] = ['#0080FF', '#FF0000', '#FFFF00']
    """ Colors assigned to neurons selected for display focus, in selection order, in the format '#RRGGBB'. """

    working_directory_changed: Signal = Signal()
    """ Signals that working directory has changed. All views should be reset and refreshed accordingly. """
    progress_updated: Signal = Signal(str)
    """ 
    Signals a status update from an IO/analysis job running in the background. Arg (str): status message. If message is 
    empty, then background task has finished. 
    """
    unit_data_ready: Signal = Signal(DataType, str)
    """ 
    Signals that neural unit information or statistics are have been prepared. Args: The data object type and the UID
    of the specific unit.
    """
    channel_traces_updated: Signal = Signal()
    """
    Signals that channel trace segments have just been updated for the current displayable channel set.
    """
    focus_neurons_changed: Signal = Signal(bool)
    """
    Signals that the set of neural units currently selected for display/comparison purposes has changed in some way.
    All views should be refreshed accordingly. **NOTE**: (1) Signal is also sent whenever a unit is added or removed
    because of a user-initiated delete/merge/split/undo operation -- because the focus list is always changed as well.
    (2) The boolean argument is True if the change in the focus list causes a in change the set of displayable analog 
    channels. When the working directory stores more than N = MAX_CHANNEL_TRACES analog channels, only the N channels 
    in the neighborhood of the primary neuron's primary channel will be available for display.
    """
    channel_seg_start_changed: Signal = Signal()
    """ 
    Signals that the elapsed starting time (relative to that start of the electrophysiological recording) for all
    analog channel trace segments has just changed.
    """
    neuron_label_updated: Signal = Signal(str)
    """ Signals that a neural unit's label wwa successfully modified. Arg(str): UID of affected unit. """
    split_lasso_region_updated: Signal = Signal()
    """ 
    Signals that the lasso region defining a split of the current primary neuron has changed -- so that UI can
    enable menu action to trigger a split when possible.
    """

    def __init__(self, main_window: QMainWindow):
        super().__init__()
        self._main_window = main_window
        """ 
        The main application window. Need this in order to block the UI with a modal progress dialog whenever we need to
        prevent further user interaction while waiting for some long-running task to finish.
        """
        self._working_directory: Optional[WorkingDirectory] = None
        """ The current working directory with configuration information. """
        self._channel_segments: Dict[int, Optional[ChannelTraceSegment]] = dict()
        """ 
        Maps the index of a recorded analog channel to a one-second segment of the data stream from that channel. Since 
        the analog data source is very large, and each channel's data stream must be extracted, bandpass-filtered if
        necessary, and cached in separate files for faster lookup, the channel trace segments will not be ready upon 
        changing the working directory.
        
        In addition, when the number of recorded channels exceeds MAX_CHANNEL_TRACES, the set of channels for which
        trace segments are those MAX_CHANNEL_TRACES "near" the primary channel for the first neural unit in the current
        focus list (if not empty). If fewer than MAX_CHANNEL_TRACES channels were recorded, than all analog channel
        traces are displayed.
        """
        self._channel_seg_start: int = 0
        """ 
        Current elapsed time, in seconds relative to the start of the electrophysiological recording, at which all 
        one-second analog channel trace segments begin (once they are loaded from internal cache).
        """
        self._neurons: List[Neuron] = list()
        """ 
        List of defined neural units. When a valid working directory is set, this will contain information on the neural
        units identified in the original spiker sorter results file located in that directory.
        """
        self._focus_neurons: List[str] = list()
        """ The UIDs of the neural units currently selected for display focus, in selection order. """
        self._lasso_for_split: Optional[QPolygonF] = None
        """ 
        The lasso region in PCA space used to split the current primary focus neuron into two separate units: one unit
        includes all spikes inside the region, while the other includes all the remaining spikes. None if undefined.
        """
        self._displayed_stats: Set[DataType] = \
            {DataType.ISI, DataType.ACG, DataType.ACG_VS_RATE, DataType.CCG, DataType.PCA}
        """ 
        The set of neural unit statistics currently on display in XSort views. To minimize background computations,
        only currently displayed statistics are computed.
        """
        self._task_manager: TaskManager = TaskManager()
        """ Background task manager. """
        self._task_manager.progress.connect(self.on_task_progress)
        self._task_manager.ready.connect(self.on_task_done)
        self._task_manager.error.connect(self.on_task_failed)
        self._task_manager.data_available.connect(self.on_data_available)

        self._progress_dlg = QProgressDialog("Please wait...", "", 0, 100, self._main_window)
        """ A modal progress dialog used to block further user input when necessary. """

        # customize progress dialog: modal, no cancel, no title bar (so you can't close it)
        self._progress_dlg.setMinimumDuration(500)
        # noinspection PyTypeChecker
        self._progress_dlg.setCancelButton(None)
        self._progress_dlg.setModal(True)
        self._progress_dlg.setAutoClose(False)
        self._progress_dlg.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        self._progress_dlg.close()   # if we don't call this, the dialog will appear shortly after app startup

    @property
    def working_directory(self) -> Optional[Path]:
        """ The analyzer's current working directory. None if no valid working directory has been set. """
        return self._working_directory.path if isinstance(self._working_directory, WorkingDirectory) else None

    @property
    def is_valid_working_directory(self) -> bool:
        """ True if analyzer's working directory is set and contains the data files XSort requires. """
        return isinstance(self._working_directory, WorkingDirectory)

    @property
    def channel_indices(self) -> List[int]:
        """
        The indices of the analog channels from the multielectrode recording that are currently available for display.
        Will be empty if no valid working directory has been set. In ascending order.

        Whenever the number of recorded analog channels exceeds MAX_CHANNEL_TRACES (recording sessions could involve
        hundreds of channels), the XSort data model only caches channel trace segments for the MAX_CHANNEL_TRACES
        channels "in the neighborhood" of the primary channel for the first unit in the current focus list, aka the
        primary neuron. In this scenario, the set of channel indices available for display will change whenever the
        focus list changes, and a signal is emitted to notify the view manager whenever that happens.
        """
        indices = [k for k in self._channel_segments.keys()]
        indices.sort()
        return indices

    def channel_label(self, idx: int) -> str:
        """
        Get the channel label for the specified analog channel. If the analog source is the Omniplex system, XSort only
        exposes the wideband and narrowband analog channels with labels "WB<N>" or "SPKC<N>, respectively, where N is
        the channel number (NOT the channel index, but the ordinal position of the channel within the set of available
        wide or narrow band channels). If the analog source is a flat binary final, then the channel label is merely
        "Ch<N>, where N is the channel index (starting from 0).
        :param idx: Index of the analog channel
        :return: The channel's label. Returns an empty string if specified channel index is invalid.
        """
        return "" if self._working_directory is None else self._working_directory.label_for_analog_channel(idx)

    def channel_trace(self, idx: int) -> Optional[ChannelTraceSegment]:
        """
        Get an analog channel trace segment
        :param idx: The channel index.
        :return: The trace segment for the specified channel, or None if that channel is not available for display.
        """
        return self._channel_segments.get(idx)

    @property
    def channel_samples_per_sec(self) -> int:
        """
        Analog channel sampling rate in Hz -- same for all channels we care about. Will be 0 if current working
        directory is invalid.
        """
        return 0 if self._working_directory is None else self._working_directory.analog_sampling_rate

    @property
    def channel_recording_duration_seconds(self) -> float:
        """
        Duration of analog channel recording in seconds. This method reports the maximum observed recording duration
        (total number of samples) across the channels we care about, but typically the duration is the same for all
        channels. Will be 0 if current working directory is invalid.
        """
        return 0 if self._working_directory is None else \
            self._working_directory.analog_channel_recording_duration_seconds

    @property
    def channel_trace_seg_start(self) -> int:
        """
        The elapsed time in seconds at which the current analog channel trace excerpts begin, relative to the start
        of the Omniplex analog recording. All excerpts are one second in duration.
        """
        return self._channel_seg_start

    def set_channel_trace_seg_start(self, t0: int) -> bool:
        """
        Set the elapsed time at which the current analog channel trace excerpts begin.
        :param t0: The new start time in seconds. If this matches the current start time, no action is taken.
        :return: True if trace segment start time was changed and a background task initiated to retrieve the trace
            segments for all relevant analog channels. False if segment start value is unchanged, or specified start
            time is invalid.
        """
        if (t0 == self._channel_seg_start) or (t0 < 0) or (t0 > self.channel_recording_duration_seconds):
            return False

        self._channel_seg_start = t0
        for idx in self._channel_segments.keys():
            self._channel_segments[idx] = None

        self.channel_seg_start_changed.emit()
        start = self._channel_seg_start * self.channel_samples_per_sec

        self._task_manager.get_channel_traces(self._working_directory, set(self.channel_indices),
                                              start, self.channel_samples_per_sec)
        return True

    def update_displayed_stats(self, show: Set[DataType], hide: Set[DataType]) -> None:
        """
        Update the shown/hidden state of one of the neural unit statistics -- ISI, ACG, ACG_VS_RATE, CCG, and PCA. When
        the corresponding view is shown/hidden, XSort's view manager calls this method to notify :class:`Analyzer`.
        To save time when computing unit statistics in the background, any hidden statistics are not computed. This is
        particularly important for principal component analysis, which is used less frequently but is by far the most
        time-consuming computation.

        :param show: The set of unit statistics to show
        :param hide: The set of unit statistics to hide
        """
        stat_added = False
        for dt in hide:
            self._displayed_stats.discard(dt)
        for dt in show:
            if DataType.is_unit_stat(dt) and not (dt in self._displayed_stats):
                self._displayed_stats.add(dt)
                stat_added = True
        if stat_added and len(self._focus_neurons) > 0:
            QTimer.singleShot(100, self.compute_statistics_for_focus_list)

    @property
    def neurons(self) -> List[Neuron]:
        """
        A **shallow** copy of the current list of neurons. If the working directory is undefined or otherwise invalid,
        this will be an empty list.
        """
        return self._neurons.copy()

    @property
    def neurons_with_display_focus(self) -> List[Neuron]:
        """
        The sublist of neural units currently selected for display/comparison purposes, in display order.

        :return: The list of neurons currently selected for display/comparison purposes, **in selection order, as that
            determines display color assigned to neuron**. Could be empty. At most will contain MAX_NUM_FOCUS_NEURONS
            entries.
        """
        out: List[Neuron] = list()
        for uid in self._focus_neurons:
            u = next((n for n in self._neurons if n.uid == uid), None)
            if u:
                out.append(u)
        return out

    @property
    def primary_neuron(self) -> Optional[Neuron]:
        """
        The primary neuron is the first in the list of neurons currently selected for display/comparison purposes.
        :return: The primary neuron, as described. Returns None if the focus list is currently empty.
        """
        focus_neurons = self.neurons_with_display_focus
        return focus_neurons[0] if len(focus_neurons) > 0 else None

    def display_color_for_neuron(self, unit_label: str) -> Optional[str]:
        """
        Get the display color assigned to a neuron in the subset of neurons selected for display/comparison purposes.

        :param unit_label: Label uniquely identifying a neural unit.
        :return: None if unit label is invalid or if the corresponding neuron is NOT currently selected for display.
            Otheriwse, the assigned color as a hexadecimal RGB string: '#RRGGBB'.
        """
        try:
            return self.FOCUS_NEURON_COLORS[self._focus_neurons.index(unit_label)]
        except ValueError:
            return None

    def update_neurons_with_display_focus(self, uid: str, clear_previous: bool = True) -> None:
        """
        Update the list of neural units currently selected for display/comparison purposes:
         - If the specified unit is invalid and the clear flag is set, the display list is cleared.
         - If the specified unit is valid and the clear flag is set, the display list is updated to contain only that
           unit. Thus, setting the clear flag enforces "single selection" behavior on the list.
         - If the specified unit is valid and the clear flag is unset, the unit is removed from the display list if it
           is already there, else it is appended to the display list unless the display list is full ("toggle selection"
           behavior).

        **A signal is emitted whenever the neuron display list changes.**

        :param uid: The UID of the neural unit to be added to the selection list.
        :param clear_previous: If True, the current display list is cleared and only the specified unit is selected. If
            False, the specified unit is added to the current display list if it is not already there and fewer than
            _MAX_NUM_FOCUS_NEURONS are already selected. If it is already present, it is removed. Default is True.
        """
        uid_exists = (uid in [n.uid for n in self._neurons])
        if not uid_exists:
            if not clear_previous:
                return
            elif len(self._focus_neurons) == 0:
                return
            else:
                self._focus_neurons.clear()
        elif not clear_previous:
            if uid in self._focus_neurons:
                self._focus_neurons.remove(uid)
            elif len(self._focus_neurons) < self.MAX_NUM_FOCUS_NEURONS:
                self._focus_neurons.append(uid)
            else:
                return
        elif (len(self._focus_neurons) == 1) and (self._focus_neurons[0] == uid):
            return
        else:
            self._focus_neurons.clear()
            self._focus_neurons.append(uid)

        self._on_focus_list_changed()

    def can_save_neurons_to_file(self) -> bool:
        """
        Can the current list of neural units be saved to a results file (currently using Python pickle format)?
        This is possible only if the working directory is valid, the neural unit list is not empty, and the standard
        metrics (spike templates, best SNR, primary channel) have been computed for all units.
        :return: True if neural unit list is ready to be saved.
        """
        return (len(self._neurons) > 0) and all([isinstance(u.primary_channel, int) for u in self._neurons])

    def save_neurons_to_file(self) -> None:
        """
        Save the current state of the neural units list to a Python pickle file specified by the user.

        After raising a modal dialog to request the destination from the user, the method blocks the UI with a
        progress dialog while the neural list is saved. If there are multiple units, with several hundred thousand
        spikes each, this operation could take a noticeable amount of time to finish.

        The units are saved as a List[Dict[str, Any]], where each dictionary in the list represents one neural unit
        and contains the following key-value pairs:
        - 'uid': The UID assigned to the neural unit (str).
        - 'spikes': A 1D Numpy array holding the unit's spike timestamps in chronological order, in seconds.
        - 'primary': The integer index identifying the primary channel on which the best signal-to-noise ratio was
          observed for the unit.
        - 'snr': The signal-to-noise observed on the unit's primary channel (float).
        - 'template': A 1D Numpy array holding the unit's mean spike waveform as recorded on the primary channel. The
          waveform spans 10-ms and is in microvolts.

        """
        if not self.can_save_neurons_to_file():
            return

        self._task_manager.cancel_all_tasks(self._progress_dlg)

        # NOTE: On MacOS Ventura, the native dialog code prints a message ("[CATransaction synchronize] called within
        # transaction") to stderr, but the dialog still appears to work.
        file_name, _ = QFileDialog.getSaveFileName(
            self._main_window, "Save neural units to file", str(self._working_directory.path.absolute()),
            'Python pickle (*.pkl *.pickle)', 'Python pickle (*.pkl *.pickle)')
        if len(file_name) == 0:
            return    # user cancelled

        out_path = Path(file_name)
        self._progress_dlg.show()
        self._progress_dlg.setLabelText(f"Saving neural units to {out_path.name}....")
        self._progress_dlg.setValue(30)

        emsg = ""
        try:
            out: List[Dict[str, Any]] = list()
            for u in self._neurons:
                out.append(dict(uid=u.uid, spikes=u.spike_times, primary=u.primary_channel, snr=u.snr,
                                template=u.get_template_for_channel(u.primary_channel)))

            with open(file_name, 'wb') as f:
                pickle.dump(out, f)
        except Exception as e:
            emsg = f"An IO or other error has occurred:\n  -->{str(e)}"
        finally:
            self._progress_dlg.close()
            self._progress_dlg.setLabelText("Please wait....")

        if len(emsg) > 0:
            QMessageBox.warning(self._main_window, "File save error", emsg)

    def prepare_for_shutdown(self) -> None:
        """
        Prepare for XSort to shutdown. The analyzer saves the edit history for the current working directory and
        cancels any running background tasks. A long-running task will prevent the application from exiting. There
        STILL may be a noticeable delay before the application exits because certain tasks may take several seconds
        to respond to the cancel request.
        """
        self._task_manager.cancel_all_tasks(self._progress_dlg)
        if isinstance(self._working_directory, WorkingDirectory):
            self._working_directory.save_edit_history()

    def change_working_directory(self, p: Union[str, Path]) -> Optional[str]:
        """
        Change the analyzer's current working directory. If the specified directory exists and contains the requisite
        data files, the analyzer will launch a background task to process these files -- and any internal XSort cache
        files already present in the directory -- to prepare the information and data needed for the various XSort
        analysis views. If the candidate directory matches the current working directory, no action is taken.

        :param p: The file system path for the candidate directory.
        :return: An error description if the cancdidate directory does not exist or does not contain the expected data
            files, or if an error occurred while building internal cache files. Returns None if successful.
        """
        if isinstance(self._working_directory, WorkingDirectory):
            self._working_directory.save_edit_history()

        self._task_manager.cancel_all_tasks(self._progress_dlg)

        _p = Path(p) if isinstance(p, str) else p
        if not isinstance(_p, Path):
            return "Invalid directory path"
        elif isinstance(self._working_directory, WorkingDirectory) and (_p == self._working_directory.path):
            return None
        elif not _p.is_dir():
            return "Directory not found"

        emsg, work_dir = WorkingDirectory.load_working_directory(_p, self._main_window)
        if work_dir is None:
            return emsg if len(emsg) > 0 else "User cancelled"

        # load the current list of neural units (takes into account any edit history)
        emsg, neurons = work_dir.load_current_neural_units()
        if len(emsg) > 0:
            return emsg

        # building the internal cache can take a long time - especially when the directory has not been cached before -
        # block user input with our modal progress dialog. For this particular task, progress messages are displayed in
        # the dialog.
        self._progress_dlg.show()
        self._progress_dlg.setValue(0)
        self._progress_dlg.setLabelText("Building internal cache files as needed...")

        # spawn task to build internal cache if necessary and/or retrieve the data we require: first second's worth
        # of each relevant analog channel trace, and metrics for all neural units
        self._task_manager.build_internal_cache(work_dir)

        # block UI while waing on BUILDCACHE task. The task progress handler will update the progress label and bar
        # irregularly, but we need to call QProgressDialog.setValue() regularly - so we do that here while never
        # hitting the maximum. NOTE that BUILDCACHE task does not deliver any data.
        channel_traces: Dict[int, ChannelTraceSegment] = dict()
        try:
            while self._task_manager.busy:
                time.sleep(0.2)
                next_value = self._progress_dlg.value() + 1
                if next_value > 99:
                    next_value = 0
                self._progress_dlg.setValue(next_value)

            # if an error occurs during the BUILDCACHE task, the progress dialog is closed and a modal dialog box
            # will have reported the error. Abort.
            if self._progress_dlg.isHidden():
                raise Exception("Unable to build internal cache!")

            self._progress_dlg.setValue(95)
            # retrieve first second's worth of samples on each of the first MAX_CHANNEL_TRACES analog channels
            self._progress_dlg.setLabelText("Loading initial channel traces...")
            for k in range(min(16, work_dir.num_analog_channels())):
                channel_traces[k] = work_dir.retrieve_cached_channel_trace(k, 0, work_dir.analog_sampling_rate)

            # load full unit metrics from cache files - inject metrics to preserve any label changes from applying the
            # edit history!
            self._progress_dlg.setLabelText("Loading neural unit metrics from cache...")
            for u in neurons:
                unit = work_dir.load_neural_unit_from_cache(u.uid)
                if unit is None or unit.primary_channel is None:
                    raise Exception(f"Missing or incomplete metrics cache file for unit {u.uid}")
                templates = {ch_idx: unit.get_template_for_channel(ch_idx) for ch_idx in unit.template_channel_indices}
                u.update_metrics(unit.primary_channel, unit.snr, templates)

        except Exception as e:
            return f"An error occurred: {str(e)}"
        finally:
            self._progress_dlg.close()
            self._progress_dlg.setLabelText("Please wait...")

        # success
        self._working_directory = work_dir
        self._channel_segments.clear()
        self._channel_segments = {k: v for k, v in channel_traces.items()}
        self._channel_seg_start = 0
        self._neurons.clear()
        self._neurons = neurons
        self._focus_neurons.clear()

        # signal views
        self.working_directory_changed.emit()

        return None

    def edit_unit_label(self, idx: int, label: str) -> bool:
        """
        Edit the label assigned to a neural unit.
        :param idx: Ordinal position of unit in the neural unit list.
        :param label: The new label. Any leading or trailing whitespace is removed. No action taken if this matches the
            current label
        :return: True if successful; False if specified unit not found or the specified label is invalid.
        """
        label = label.strip()
        try:
            u = self._neurons[idx]
            if u.label != label:
                prev_label = u.label
                u.label = label
                self._working_directory.on_unit_relabeled(u.uid, prev_label, u.label)
                self.neuron_label_updated.emit(u.uid)
            return True
        except Exception:
            pass
        return False

    def can_delete_primary_neuron(self) -> bool:
        """
        Can the unit currently selected as the 'primary neuron' be deleted? Deletion is permitted ONLY if: (1) there are
        no other units in the current display focus list; and (2) the unit's internal cache file has already been
        generated. The latter requirement facilitates quickly and reliably undoing a previous delete operation by simply
        reloading the unit from that cache file.
        :return: True only if the above requirements are met.
        """
        primary: Optional[Neuron] = self.primary_neuron
        return (len(self._focus_neurons) == 1) and self._working_directory.unit_cache_file_exists(primary.uid)

    def delete_primary_neuron(self, uid: Optional[str] = None) -> None:
        """
        Delete the current 'primary neuron', ie, the first unit in the current display focus list.

        Only one unit may be deleted at a time, so no action is taken if the display focus list is empty or contains
        more than one unit. Deleting a unit does not remove the associated unit cache file (if present) from the current
        working directory.

        :param uid: The UID of a remaining unit that should become the 'primary neuron' after the deletion. If None or
            invalid, the display focus list will be empty after the deletion.
        """
        if not self.can_delete_primary_neuron():
            return
        deleted_idx = next((i for i in range(len(self._neurons))
                            if self._neurons[i].uid == self._focus_neurons[0]), None)
        if deleted_idx is None:
            return

        # cancel all background tasks before altering the neural unit list
        self._task_manager.cancel_all_tasks(self._progress_dlg)

        u = self._neurons.pop(deleted_idx)
        if uid in [n.uid for n in self._neurons]:
            self._focus_neurons[0] = uid
        else:
            self._focus_neurons.clear()
        self._working_directory.on_unit_deleted(u.uid)
        self._on_focus_list_changed()

    def can_merge_focus_neurons(self) -> bool:
        """
        Can the units currently selected for display/focus purposes be merged into a single unit. The "merge" is
        permitted only if: (1) exactly TWO units are selected for display/focus; and (2) the internal cache file for
        each unit has already been generated. The latter requirement facilitates quickly and reliably undoing the merge
        operation -- by simply reloading the two merged units from their respective cache files.

        :return: True only if the above requirements are met.
        """
        return (len(self._focus_neurons) == 2) and all([self._working_directory.unit_cache_file_exists(uid)
                                                        for uid in self._focus_neurons])

    def merge_focus_neurons(self) -> None:
        """
        Merge the neural units in the current focus list into a single unit. The newly derived unit is appended to the
        list of derived units, while the two component units are removed. The new unit also becomes the sole occupant
        of the focus list.

            A background task is immediately spawned to cache the spike times and other metrics (per-channel template
        waveforms, SNR, etc) for the merged unit, then compute the various statistics (ISI/ACG/CCG/PCA) for units in
        the focus list.

            The unit cache files corresponding to the units that were merged remain in the XSort working directory so
        that they may be quickly "recovered" if the merge operation is later undone.
        """
        if not self.can_merge_focus_neurons():
            return
        units = self.neurons_with_display_focus
        merged_unit = Neuron.merge(units[0], units[1], self._working_directory.find_next_assignable_unit_index())
        if not self._working_directory.save_neural_unit_to_cache(merged_unit):
            QMessageBox.warning(self._main_window, "Merge failed", "Unable to save merged unit to internal cache file!")
            return

        # cancel all background tasks before altering the neural unit list
        self._task_manager.cancel_all_tasks(self._progress_dlg)

        self._focus_neurons.clear()
        for u in units:
            self._neurons.remove(u)  # works bc neurons_with_display_focus contains units from this list
        self._neurons.append(merged_unit)
        self._focus_neurons.append(merged_unit.uid)
        self._working_directory.on_units_merged(units[0].uid, units[1].uid, merged_unit.uid)
        self._on_focus_list_changed()

    def set_lasso_region_for_split(self, lasso_region: Optional[QPolygonF]) -> None:
        """
        Set or clear the lasso region that must be defined in the PCA space of the current primary neural unit in
        order to split it into two disjoint units (one including the spikes that project inside the region, and the
        other including the spike that don't). A signal is emitted so that the view manager can refresh the "split"
        action accordingly.
            The lasso region may only be set when a single unit occupies the display/focus list, and that unit's
        metrics have been computed and cached in the XSort working directory.
        :param lasso_region: The polygon defining the lasso region, in the same coordinates as the primary unit's
            cached PCA projection. If None or not closed, the lasso region is set to None (undefined).
        """
        if self._lasso_for_split == lasso_region:
            return
        if ((len(self._focus_neurons) == 1) and
                self._working_directory.unit_cache_file_exists(self._focus_neurons[0])):
            if (lasso_region is None) or not lasso_region.isClosed():
                self._lasso_for_split = None
            else:
                self._lasso_for_split = lasso_region
            self.split_lasso_region_updated.emit()

    def can_split_primary_neuron(self) -> bool:
        """
        Can the unit currently selected as the 'primary neuron' be split into two separate units? Splitting is permitted
        ONLY if: (1) there are no other units in the current display focus list; (2) the unit's internal cache file has
        already been generated (to quickly and reliably undo the split by reloading the unit from the cache); and (3)
        a closed "split region" has been defined in the unit's PCA projection space. All spikes inside the region are
        assigned to one new unit, while the remaining spikes are assigned to a second unit.
        :return: True only if the above requirements are met.
        """
        focussed = self.neurons_with_display_focus
        return ((len(focussed) == 1) and (focussed[0].num_spikes > 2) and
                self._working_directory.unit_cache_file_exists(focussed[0].uid) and
                focussed[0].is_pca_projection_cached and (self._lasso_for_split is not None))

    def split_primary_neuron(self) -> None:
        """
        Split the single neural unit in the current focus list into two distinct units: all of the unit's spikes that
        project inside the current "split region" in PCA space form the spike train for one unit, while the rest form
        the spike train for the other. The newly derived units are appended to the neuron list, while the split unit
        is removed. The two new units also get the display focus.

            Splitting a unit with 100K spikes or more can take a noticeable amount of time, so a modal progress dialog
        will block user input during this operation.

            A background task is immediately spawned to cache the spike times and other metrics (per-channel template
        waveforms, SNR, etc) for the two new units, then compute the various statistics (ISI/ACG/CCG/PCA) for them, as
        they now occupy the focus list.

            The unit cache file corresponding to the unit that was split remains in the XSort working directory so
        that it may be quickly "recovered" if the split operation is later undone.
        """
        if not self.can_split_primary_neuron():
            return

        # cancel all background tasks before altering the neural unit list
        self._task_manager.cancel_all_tasks(self._progress_dlg)

        # do the split. This involves finding which spikes project inside the lasso region in PCA space, which can take
        # a few seconds if there 100K spikes or more. So we block user input with the modal progress dialog
        self._progress_dlg.show()
        self._progress_dlg.setValue(0)

        unit = self.neurons_with_display_focus[0]
        split_units: List[Neuron] = list()

        try:
            proj = unit.cached_pca_projection()
            inside_indices: List[int] = list()
            outside_indices: List[int] = list()
            pt = QPointF(0, 0)
            for i in range(len(proj)):
                pt.setX(proj[i, 0])
                pt.setY(proj[i, 1])
                if self._lasso_for_split.containsPoint(pt, Qt.FillRule.WindingFill):
                    inside_indices.append(i)
                else:
                    outside_indices.append(i)
                pct = int(95.0 * i / len(proj))
                self._progress_dlg.setValue(pct)

            if len(inside_indices) == 0 or len(outside_indices) == 0:
                return

            idx = self._working_directory.find_next_assignable_unit_index()
            inside_spikes = np.take(unit.spike_times, inside_indices)
            inside_spikes.sort()
            split_units.append(Neuron(idx, inside_spikes, suffix='x'))
            self._progress_dlg.setValue(97)
            outside_spikes = np.take(unit.spike_times, outside_indices)
            outside_spikes.sort()
            split_units.append(Neuron(idx + 1, outside_spikes, suffix='x'))
            self._progress_dlg.setValue(99)

            cached = True
            for u in split_units:
                if not self._working_directory.save_neural_unit_to_cache(u):
                    cached = False
            if not cached:
                for u in split_units:
                    self._working_directory.delete_unit_cache_file(u.uid)
                QMessageBox.warning(self._main_window, "Split failed",
                                    "Unable to save one or both split units to internal cache!")
                return
        finally:
            self._progress_dlg.close()

        # success! Update edit history and neuron list and put focus on the two new units
        self._focus_neurons.clear()
        self._neurons.remove(unit)
        self._neurons.extend(split_units)
        self._focus_neurons.extend([u.uid for u in split_units])
        self._working_directory.on_unit_split(unit.uid, split_units[0].uid, split_units[1].uid)
        self._on_focus_list_changed()

    def undo_last_edit(self) -> bool:
        """
        Undo the most recent user-initiated edit to the current neural unit list.
        :return: True if undo succeeds, False otherwise.
        """
        edit_rec = self._working_directory.remove_most_recent_edit() if self.is_valid_working_directory else None
        if edit_rec is None:
            return False

        # cancel all background tasks before altering the neural unit list (changing the label of a unit is ok)
        if edit_rec.operation != UserEdit.LABEL:
            self._task_manager.cancel_all_tasks(self._progress_dlg)

        if edit_rec.operation == UserEdit.LABEL:
            for u in self._neurons:
                if u.uid == edit_rec.affected_uids and u.label == edit_rec.unit_label:
                    u.label = edit_rec.previous_unit_label
                    self.neuron_label_updated.emit(u.uid)
                    return True
            return False
        elif edit_rec.operation == UserEdit.DELETE:
            uid = edit_rec.affected_uids
            u = self._working_directory.load_neural_unit_from_cache(uid)
            if isinstance(u, Neuron):
                # success: make undeleted neuron the one and only neuron in the current display list
                self._neurons.append(u)
                self._focus_neurons.clear()
                self._focus_neurons.append(u.uid)
                self._on_focus_list_changed()
                return True
            return False
        elif edit_rec.operation == UserEdit.MERGE:
            merged_uid = edit_rec.result_uids
            merged_idx = next((i for i in range(len(self._neurons)) if self._neurons[i].uid == merged_uid), None)
            restored_units = list()
            for uid in edit_rec.affected_uids:
                restored_units.append(self._working_directory.load_neural_unit_from_cache(uid))
            if isinstance(merged_idx, int) and all([isinstance(u, Neuron) for u in restored_units]):
                # success: remove the merged unit (including cache file), restore component units, and put them in the
                # focus list
                self._neurons.pop(merged_idx)
                self._working_directory.delete_unit_cache_file(merged_uid)
                self._neurons.extend(restored_units)
                self._focus_neurons.clear()
                self._focus_neurons.extend([u.uid for u in restored_units])
                self._on_focus_list_changed()
                return True
            return False
        else:   # SPLIT
            split_uids = [uid for uid in edit_rec.result_uids]
            unsplit_uid = edit_rec.affected_uids
            restored_unit = self._working_directory.load_neural_unit_from_cache(unsplit_uid)
            split_units: List[Neuron] = list()
            for u in self._neurons:
                if u.uid in split_uids:
                    split_units.append(u)
            if (len(split_units) == 2) and isinstance(restored_unit, Neuron):
                # success: remove the component units, restore the unsplit unit, and put the focus on that unit
                self._focus_neurons.clear()
                for u in split_units:
                    self._working_directory.delete_unit_cache_file(u.uid)
                    self._neurons.remove(u)
                self._neurons.append(restored_unit)
                self._focus_neurons.append(restored_unit.uid)
                self._on_focus_list_changed()
                return True
            return False

    def can_undo_all_edits(self) -> bool:
        """
        Can the edit history be wiped out for the current XSort working directory? This operation is always
        possible, unless working directory is invalid or has an empty edit history.
        :return True if current edit history is not empty.
        """
        return self.is_valid_working_directory and self._working_directory.is_edited

    def undo_all_edits(self) -> None:
        """
        Undo all changes made to the contents of the current XSort woring directory, restoring it to its original
        state. No action is taken if the edit history is empty.
        """
        if not self.can_undo_all_edits():
            return

        # cancel all background tasks before altering the neural unit list
        self._task_manager.cancel_all_tasks(self._progress_dlg)

        # special case: only unit labels have been changed
        if self._working_directory.edit_history_has_only_unit_label_changes:
            for u in self._neurons:
                u.label = ''
            self._working_directory.clear_edit_history()
            self._focus_neurons.clear()
            self._on_focus_list_changed()
            return

        # reload all "original" neural units from the spike sorter results file (PKL)
        emsg, neurons = self._working_directory.load_original_neural_units()
        if len(emsg) > 0:
            return

        # go through current unit list. Determine which original units are missing, and remove derived units
        i = 0
        while i < len(self._neurons):
            u = self._neurons[i]
            if u.is_mod:
                self._neurons.pop(i)
            else:
                u.label = ''
                for unit in neurons:
                    if unit.uid == u.uid:
                        neurons.remove(unit)
                        break
                i = i + 1

        # for each "original unit" that is NOT in the current list, attempt to load its full metrics from internal
        # cache file if we can, then put it back in the unit list
        for u in neurons:
            unit = self._working_directory.load_neural_unit_from_cache(u.uid)
            self._neurons.append(unit if isinstance(unit, Neuron) else u)

        self._focus_neurons.clear()

        # clear edit history and remove any derived unit cache files
        self._working_directory.clear_edit_history()
        self._working_directory.save_edit_history()

        # signal views
        self._on_focus_list_changed()

    def _on_focus_list_changed(self) -> None:
        """
        Whenever the current display focus list changes, we need to cancel any ongoing COMPUTESTATS task before
        signalling the view manager, and then queue a new COMPUTESTATS task (unless the focus list is now empty)
        after a short delay so that the user-facing views have a chance to refresh before that CPU-intensive task
        begins.

        If the number of recorded channels is larger than what XSort will display (MAX_CHANNEL_TRACES), then the set of
        displayable channel indices is adjusted to center around the primary channel of the first neuron in the focus
        list. Thus, whenever that set changes, the trace segment cache is cleared, the view manager is notified of the
        change and a task is launched to retrieve trace segments for the new set of channel indices.
        """
        self._task_manager.cancel_compute_stats(self._progress_dlg)

        # clear out stale PCA projections
        for u in self._neurons:
            u.clear_cached_pca_projection()

        # IMPORTANT: we must update the displayable channel list before notifying the view manager that the focus list
        # changed -- eg, templates are only shown for the current set of displayable channels, which can change when
        # there are more than 16 recorded analog channels
        channels_changed = self._update_displayable_channels_if_necessary()
        self.focus_neurons_changed.emit(channels_changed)

        # if the focus list is not empty, trigger a new COMPUTESTATS task after a brief delay
        if len(self._focus_neurons) > 0:
            QTimer.singleShot(400, self.compute_statistics_for_focus_list)

    def _update_displayable_channels_if_necessary(self) -> bool:
        """
        Update the set of displayable analog channel indices if necessary. If there is a change, the previously cached
        channel trace segments are cleared, and a background task is launched to retrieve trace segments for each
        channel in the new set.

        By design, XSort limits the number of displayable analog channel traces to MAX_CHANNEL_TRACES. When the number
        of recorded channels exceeds this limit, the data model updates the displayable channel set to center it around
        the primary channel for the first unit in the current display focus list, aka, the "primary neuron".

        :return: True if the set of displayable channels has changed.
        """
        # no change if all channels are displayable, or primary neuron or its primary channel are undefined
        if ((self._working_directory.num_analog_channels() <= MAX_CHANNEL_TRACES) or (self.primary_neuron is None) or
                (self.primary_neuron.primary_channel is None)):
            return False

        current_indices = set(self.channel_indices)
        indices = set(self.primary_neuron.template_channel_indices)
        if indices == current_indices:
            return False

        self._channel_segments.clear()
        self._channel_segments = {k: None for k in indices}
        start = self._channel_seg_start * self.channel_samples_per_sec
        self._task_manager.get_channel_traces(self._working_directory, indices, start, self.channel_samples_per_sec)
        return True

    def compute_statistics_for_focus_list(self) -> None:
        """
        Prepare a request for various statistics for neural units in the focus list IAW what statistics are currently
        on display in XSort. If any requested statistics are NOT already cached in the focus units, then launch a
        background task to compute the missing statistics.

        By design, XSort only computes the statistics (ISI, ACG, ACG_VS_RATE, CCG, PCA) for those units comprising the
        current focus list, since the relevant XSort views only display statistics for those units. Other than a unit's
        PCA projection, the statistic is computed once and cached in the unit from thereon (until the working directory
        changes again). Furthermore, if the XSort view that displays a particular statistic is currently hidden, there's
        no reason to compute it. This is particularly important for the PCA projections, which take a relatively long
        time to compute. The user may hide the PCA view component until that analysis is needed.
        """
        focus_units = self.neurons_with_display_focus
        if len(focus_units) == 0:
            return
        needed_stats: List[Tuple] = list()
        for dt in self._displayed_stats:
            uids: Set[str] = set()
            if dt == DataType.CCG:
                for u in focus_units:
                    for u2 in focus_units:
                        if (u.uid != u2.uid) and not u.is_statistic_cached(dt, u2.uid):
                            uids.update({u.uid, u2.uid})
                    if len(uids) == len(focus_units):
                        break
            elif dt == DataType.PCA:
                if not all([u.is_statistic_cached(dt) for u in focus_units]):
                    uids.update({u.uid for u in focus_units})
            else:
                uids.update({u.uid for u in focus_units if not u.is_statistic_cached(dt)})
            if len(uids) > 0:
                needed_stats.append((dt,) + tuple(uids))

        if len(needed_stats) > 0:
            self._task_manager.cancel_compute_stats(self._progress_dlg)
            self._task_manager.compute_unit_stats(self._working_directory, needed_stats)

    def undo_last_edit_description(self) -> Optional[Tuple[str, str]]:
        """
        Get a short and longer description of the most recent user-initiated edit to the current neural unit list -- ie,
        the edit that will be "undone" if :method:`undo_last_edit()` is invoked.
        :return: A 2-tuple (S, L) containing the short and longer descriptions, or None if the edit history is empty.
        """
        edit_rec = self._working_directory.most_recent_edit if self.is_valid_working_directory else None
        if edit_rec is None:
            return None
        else:
            return edit_rec.short_description, edit_rec.longer_description

    @Slot(str, int)
    def on_task_progress(self, desc: str, pct: int) -> None:
        """
        This slot is the mechanism by which :class:`Analyzer` receives progress updates from a background task managed
        by the :class:`TaskManager`. It forwards a progress message to the view manager, and if the modal progress
        dialog is currently visible, it updates the dialog label and progress bar accordingly.

        :param desc: Progress message.
        :param pct: Percent complete. If this lies in [0..100], then "{pct}%" is appended to the progress message.
        """
        msg = f"{desc} - {pct}%" if (0 <= pct <= 100) else desc
        self.progress_updated.emit(msg)

        if self._progress_dlg.isVisible():
            self._progress_dlg.setLabelText(msg)
            if 0 <= pct <= 100:
                self._progress_dlg.setValue(pct)

    @Slot()
    def on_task_done(self) -> None:
        """
        This slot is the mechanism by which :class:`Analyzer` is notified that a background task has finished. The
        Analyzer notifies the view manager of this fact by emitting an empty progress message.
        """
        self.progress_updated.emit("")

    @Slot(str)
    def on_task_failed(self, emsg: str) -> None:
        """ This slot is the mechanism by which :class:`Analyzer` is notified that a background task has failed. """
        # close modal progress dialog if it is raised
        if self._progress_dlg.isVisible():
            self._progress_dlg.close()
            self._progress_dlg.setLabelText("Please wait...")

        # warn user of the error that occured
        QMessageBox.warning(self._main_window, "Background task failed", emsg)

    @Slot(DataType, object)
    def on_data_available(self, data_type: DataType, data: object) -> None:
        """
        This slot is the mechanism by which Analyzer, living in the main GUI thread, receives data objects that are
        prepared/retrieved by background tasks launched and managed by the :class:`TaskManager`. The data object
        delivered depends on the :class:`DataType`:
         - NEURON: A :class:`Neuron` object. Delivered after the unit metrics were calculated and
           saved to an internal unit cache file.
         - CHANNELTRACE: A class:`ChannelTraceSegment` object, containig a small contiguous segment of a recorded
           analog trace for a specific channel.
         - ISI: A 2-tuple (uid, A), where A is a 1D Numpy array holding the computed interspike interval histogram for
           the specified neural unit.
         - ACG: A 2-tuple (uid, A), where A is a 1D Numpy array holding the computed autocorrelogram for the unit.
         - ACG_VS_RATE: A 2-tuple (uid, (A, B)) where A is a 1D Numpy array of firing rate bin centers and B is a 2D
           Numpy array containing the computed ACB for each firing rate bin.
         - CCG: A 3-tuple (uid1, uid2, A), where A is the crosscorrelogram of units uid1 and uid2.
         - PCA: A 3-tuple (uid, K, P). The PCA projection for unit UID. PCA projections are time-consuming and delivered
           in chunks. K is the starting spike index for the chunk, and P is the 2D Numpy array of size (N,2) holding the
           PCA projection for spikes [spk_idx: spk_idx+N].

        In all cases, Analyzer caches the data object internally, then notifies the view manager. The various statistics
        are cached in the analyzer's copy of the :class:`Neuron` object.

        :param data_type: Enumeration indicating the type of data made available.
        :param data: The data retrieved. See details above.
        """
        if (data_type == DataType.NEURON) and isinstance(data, Neuron):
            # neural unit record with SNR, templates and other metrics retrieved from internal cache file
            u: Neuron = data
            for unit in self._neurons:
                if unit.uid == u.uid:
                    templates = {ch_idx: u.get_template_for_channel(ch_idx) for ch_idx in u.template_channel_indices}
                    unit.update_metrics(u.primary_channel, u.snr, templates)
                    self.unit_data_ready.emit(DataType.NEURON, unit.uid)
                    break
        elif (data_type == DataType.CHANNELTRACE) and isinstance(data, ChannelTraceSegment):
            seg: ChannelTraceSegment = data
            if seg.channel_index in self._channel_segments:
                self._channel_segments[seg.channel_index] = seg
                # we only signal the view manager once we've received all channel segments
                if not any([seg is None for seg in self._channel_segments.values()]):
                    self.channel_traces_updated.emit()
        elif DataType.is_unit_stat(data_type) and isinstance(data, tuple):
            try:
                for u in self._neurons:
                    if u.uid == data[0]:
                        u.cache_statistic(data_type, data)
                        self.unit_data_ready.emit(data_type, u.uid)
                        break
            except Exception:
                pass
