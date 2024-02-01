import time
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, QThreadPool, QTimer, QPointF
from PySide6.QtCore import Qt
from PySide6.QtGui import QPolygonF
from PySide6.QtWidgets import QMainWindow, QProgressDialog

from xsort.data import PL2
from xsort.data.edits import UserEdit
from xsort.data.neuron import Neuron, ChannelTraceSegment, DataType
from xsort.data.tasks import (Task, TaskType)
from xsort.data.files import get_required_data_files, load_spike_sorter_results, unit_cache_file_exists, \
    delete_unit_cache_file, delete_all_derived_unit_cache_files_from, save_neural_unit_to_cache, \
    load_neural_unit_from_cache


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
    data_ready: Signal = Signal(DataType, str)
    """ 
    Signals that some data that was retrieved or prepared in the background. Args: The data object type, and a 
    string identifier: the unit label for a neuron, or the integer index (as a string) of the analog channel source for
    a channel trace segment.
    """
    focus_neurons_changed: Signal = Signal()
    """
    Signals that the set of neural units currently selected for display/comparison purposes has changed in some way.
    All views should be refreshed accordingly. **NOTE**: This signal is also sent whenever a unit is added or removed
    because of a user-initiated delete/merge/split/undo operation -- because the focus list is always changed as well.
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
        self._working_directory: Optional[Path] = None
        """ The current working directory. """
        self._pl2_file: Optional[Path] = None
        """ The Omniplex mulit-channel electrode recording file (PL2). """
        self._pl2_info: Optional[Dict[str, Any]] = None
        """ Metadata extracted from the Omniplex data file. """
        self._channel_segments: Dict[int, Optional[ChannelTraceSegment]] = dict()
        """ 
        Maps the index of a recorded analog channel in the Omniplex file to a one-second segment of the data stream 
        from that channel. Only includes analog wideband and narrowband channels that were actually recorded. Since the
        Omniplex file is very large and the analog data streams must be extracted, bandpass-filtered if necessary, and
        cached in separate files for faster lookup, the channel trace segments will not be ready upon changing the
        working directory.
        """
        self._channel_seg_start: int = 0
        """ 
        Current elapsed time, in seconds relative to the start of the Omniplex recording, at which all one-second analog
        channel trace segments begin (once they are loaded from internal cache).
        """
        self._pkl_file: Optional[Path] = None
        """ The original spike sorter results file (for now, must be a Python Pickle file). """
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
        self._edit_history: List[UserEdit] = list()
        """ The edit history, a record of all user-initiated changes to the list of neural units, in chrono order. """
        self._thread_pool = QThreadPool()
        """ Managed thread pool for running slow background tasks. """

        self._background_tasks: Dict[TaskType, Optional[Task]] = {
            TaskType.BUILDCACHE: None, TaskType.COMPUTESTATS: None, TaskType.GETCHANNELS: None
        }
        """ Dictionary of running tasks, by type. """

        self._progress_dlg = QProgressDialog("Please wait...", "", 0, 100, self._main_window)
        """ A modal progress dialog used to block further user input when necessary. """

        # customize progress dialog: modal, no cancel, no title bar (so you can't close it)
        self._progress_dlg.setMinimumDuration(500)
        self._progress_dlg.setCancelButton(None)
        self._progress_dlg.setModal(True)
        self._progress_dlg.setAutoClose(False)
        self._progress_dlg.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        self._progress_dlg.close()   # if we don't call this, the dialog will appear shortly after app startup

    @property
    def working_directory(self) -> Optional[Path]:
        """ The analyzer's current working directory. None if no valid working directory has been set. """
        return self._working_directory

    @property
    def is_valid_working_directory(self) -> bool:
        """ True if analyzer's working directory is set and contains the data files XSort requires. """
        return isinstance(self._working_directory, Path)

    @property
    def channel_indices(self) -> List[int]:
        """
        The indices of the wideband ("WB") and narrowband ("SPKC") analog channels available in the Omniplex
        multielectrode recording. Will be empty if no valid working directory has been set. In ascending order.
        """
        indices = [k for k in self._channel_segments.keys()]
        indices.sort()
        return indices

    def channel_label(self, idx: int) -> str:
        """
        Get the channel label for the specified Omniplex analog channel. XSort only exposes the wideband and narrowband
        nalog channels with labels "WB<N>" or "SPKC<N>, respectively, where N is the channel number (NOT the channel
        index, but the ordinal position of the channel within the set of available wide or narrow band channels).
        :param idx: Index of an Omminplex analog channel
        :return: The channel label. Returns an empty string if specified channel index is not one of the channel
            indices returned by :method:`Analyzer.channel_indices`.
        """
        if (self._pl2_info is None) or not (idx in self._channel_segments.keys()):
            return ""
        ch_dict = self._pl2_info['analog_channels'][idx]
        src = "WB" if ch_dict['source'] == PL2.PL2_ANALOG_TYPE_WB else "SPKC"
        return f"{src}{str(ch_dict['channel'])}"

    def channel_trace(self, idx: int) -> Optional[ChannelTraceSegment]:
        """
        Get an analog channel trace segment
        :param idx: The channel index.
        :return: The trace segment for the specified channel, or None if that channel is not available.
        """
        return self._channel_segments.get(idx)

    @property
    def channel_samples_per_sec(self) -> int:
        """
        Analog channel sampling rate in Hz -- same for all channels we care about. Will be 0 if current working
        directory is invalid.
        """
        if (self._pl2_info is None) or (len(self._channel_segments) == 0):
            return 0
        else:
            idx = next(iter(self._channel_segments))
            return int(self._pl2_info['analog_channels'][idx]['samples_per_second'])

    @property
    def channel_recording_duration_seconds(self) -> float:
        """
        Duration of analog channel recording in seconds. This method reports the maximum observed recording duration
        (total number of samples) across the channels we care about, but typically the duration is the same for all
        channels. Will be 0 if current working directory is invalid.
        """
        rate = self.channel_samples_per_sec
        if rate > 0:
            dur = max([self._pl2_info['analog_channels'][idx]['num_values'] for idx in self._channel_segments.keys()])
            return dur / rate
        else:
            return 0

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
        self._launch_background_task(TaskType.GETCHANNELS)
        return True

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
         - If the specified unit is valid and the clear flag is unset, the unit is appended to the display list unless
           it is already there or the display list is full.

        **A signal is emitted whenever the neuron display list changes.**

        :param uid: The UID of the neural unit to be added to the selection list.
        :param clear_previous: If True, the current display list is cleared and only the specified unit is selected. If
            False, the specified unit is added to the current display list if it is not already there and fewer than
            _MAX_NUM_FOCUS_NEURONS are already selected. Default is True.
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
            if (len(self._focus_neurons) == self.MAX_NUM_FOCUS_NEURONS) or (uid in self._focus_neurons):
                return
            self._focus_neurons.append(uid)
        elif (len(self._focus_neurons) == 1) and (self._focus_neurons[0] == uid):
            return
        else:
            self._focus_neurons.clear()
            self._focus_neurons.append(uid)

        self._on_focus_list_changed()

    def _launch_compute_stats_task(self) -> None:
        self._launch_background_task(TaskType.COMPUTESTATS)

    def _cancel_background_task(self, t: TaskType) -> None:
        """
        Cancel a running background task.

            By design, no two instances of a particular task type can be running in the background at the same time.
        Certain tasks (building the cache files for a new XSort working directory; computing the PCA projections for
        units in the display focus list) can take a significant amount of time to complete. To keep XSort as responsive,
        as possible, all tasks are cancellable. However, in the current framework, we must wait an indeterminate
        amount of time for a task to "detect" the cancel request and stop.
            While waiting, the UI should be blocked to prevent further user interactions which could trigger additional
        background work. Thus, this method raises a modal progress dialog while waiting for the cancelled task to
        finish.
        :param t: The task type.
        """
        if not (self._background_tasks[t] is None):
            try:
                self._background_tasks[t].cancel()
                i = 0
                while self._background_tasks[t] is not None:
                    self._progress_dlg.setValue(i)
                    time.sleep(0.05)
                    # in the unlikely event it takes more than 5s for task to stop, reset progress to 0%
                    i = 0 if i == 99 else i + 1
            finally:
                self._background_tasks[t] = None
                self._progress_dlg.close()

    def _launch_background_task(self, t: TaskType) -> None:
        """
        Helper method launches ones of the XSort background tasks. If a task of that type is already running (or was
        running), it is cancelled before launching a new instance of that task. By design, only one instance of each
        type of task should run at one time.
        :param t: The type of task to launch
        """
        self._cancel_background_task(t)

        if t == TaskType.BUILDCACHE:
            task = Task(TaskType.BUILDCACHE, self._working_directory)
        elif t == TaskType.COMPUTESTATS:
            focus_list = self.neurons_with_display_focus
            if len(focus_list) == 0:
                return
            task = Task(TaskType.COMPUTESTATS, self._working_directory, units=focus_list)
        elif t == TaskType.GETCHANNELS:
            t0 = self._channel_seg_start
            task = Task(TaskType.GETCHANNELS, self._working_directory, start=t0 * self.channel_samples_per_sec,
                        count=self.channel_samples_per_sec)
        else:
            return

        task.signals.progress.connect(self.on_task_progress)
        task.signals.error.connect(self.on_task_failed)
        task.signals.data_available.connect(self.on_data_available)
        task.signals.finished.connect(self.on_task_done)
        self._background_tasks[t] = task
        self._thread_pool.start(task)

    def prepare_for_shutdown(self) -> None:
        """
        Prepare for XSort to shutdown. The analyzer saves the edit history for the current working directory and
        cancels any running background tasks. A long-running task will prevent the application from exiting. There
        STILL may be a noticeable delay before the application exits because certain tasks may take several seconds
        to respond to the cancel request.
        """
        for task_type in self._background_tasks:
            self._cancel_background_task(task_type)
        self.save_edit_history()

    def save_edit_history(self):
        """
        Save the edit history for the current XSort working directory. Be sure to call this method prior to
        application shutdown.
        """
        if isinstance(self._working_directory, Path) and self._working_directory.is_dir():
            UserEdit.save_edit_history(self._working_directory, self._edit_history)

    def change_working_directory(self, p: Union[str, Path]) -> Optional[str]:
        """
        Change the analyzer's current working directory. If the specified directory exists and contains the requisite
        data files, the analyzer will launch a background task to process these files -- and any internal XSort cache
        files already present in the directory -- to prepare the information and data needed for the various XSort
        analysis views. If the candidate directory matches the current working directory, no action is taken.

        :param p: The file system path for the candidate directory.
        :return: An error description if the cancdidate directory does not exist or does not contain the expected data
            files; else None.
        """
        self.save_edit_history()

        for task_type in self._background_tasks:
            self._cancel_background_task(task_type)

        _p = Path(p) if isinstance(p, str) else p
        if not isinstance(_p, Path):
            return "Invalid directory path"
        elif _p == self._working_directory:
            return None
        elif not _p.is_dir():
            return "Directory not found"

        # check for required data files. For now, we expect exactly one PL2 and one PKL file
        pl2_file, pkl_file, emsg = get_required_data_files(_p)
        if len(emsg) > 0:
            return emsg

        # load metadata from the PL2 file.
        pl2_info: Dict[str, Any]
        channel_map: Dict[int, Optional[ChannelTraceSegment]] = dict()
        try:
            with open(pl2_file, 'rb') as fp:
                pl2_info = PL2.load_file_information(fp)
                channel_list = pl2_info['analog_channels']
                for i in range(len(channel_list)):
                    if channel_list[i]['num_values'] > 0:
                        if channel_list[i]['source'] in [PL2.PL2_ANALOG_TYPE_WB, PL2.PL2_ANALOG_TYPE_SPKC]:
                            channel_map[i] = None
        except Exception as e:
            return f"Unable to read Ommniplex (PL2) file: {str(e)}"

        # load neural units (spike train timestamps) from the spike sorter results file (PKL)
        emsg, neurons = load_spike_sorter_results(pkl_file)
        if len(emsg) > 0:
            return f"Unable to read spike sorter results from PKL file: {emsg}"

        # load edit history file (if present)
        emsg, edit_history = UserEdit.load_edit_history(_p)
        if len(emsg) > 0:
            return f"Error reading edit history: {emsg}"

        # success
        self._working_directory = _p
        self._pl2_file = pl2_file
        self._pl2_info = pl2_info
        self._channel_segments = channel_map
        self._channel_seg_start = 0
        self._pkl_file = pkl_file
        self._neurons = neurons
        self._edit_history = edit_history
        self._focus_neurons.clear()

        # apply full edit history to bring neuron list up-to-date
        for edit_rec in self._edit_history:
            edit_rec.apply_to(self._neurons)

        # signal views
        self.working_directory_changed.emit()

        # building the internal cache can take a long time - especially when the directory has not been cached before -
        # block user input with our modal progress dialog. For this particular task, progress messages are displayed in
        # the dialog.
        self._progress_dlg.show()
        self._progress_dlg.setValue(0)

        # spawn task to build internal cache if necessary and/or retrieve the data we require: first second's worth
        # of each relevant analog channel trace, and metrics for all neural units
        self._launch_background_task(TaskType.BUILDCACHE)

        # block UI while waing on BUILDCACHE task. The task progress handle will update the progress label and bar
        # irregularly, but we need to call QProgressDialog.setValue() regularly - so we do that here while never
        # hitting the maximum.
        try:
            while self._background_tasks[TaskType.BUILDCACHE] is not None:
                time.sleep(0.05)
                next_value = self._progress_dlg.value() + 1
                if next_value > 99:
                    next_value = 0
                self._progress_dlg.setValue(next_value)
        finally:
            self._background_tasks[TaskType.BUILDCACHE] = None
            self._progress_dlg.close()
            self._progress_dlg.setLabelText("Please wait...")

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
            prev_label = u.label
            u.label = label
            edit_rec = UserEdit(op=UserEdit.LABEL, params=[u.uid, prev_label, label])
            self._edit_history.append(edit_rec)
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
        return (len(self._focus_neurons) == 1) and unit_cache_file_exists(self._working_directory, primary.uid)

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

        # thread conflict: must cancel a COMPUTESTATS task in progress bc its signals can lead to accessing the neural
        # unit list while it is being altered here!
        self._cancel_background_task(TaskType.COMPUTESTATS)

        u = self._neurons.pop(deleted_idx)
        if uid in [n.uid for n in self._neurons]:
            self._focus_neurons[0] = uid
        else:
            self._focus_neurons.clear()
        edit_rec = UserEdit(op=UserEdit.DELETE, params=[u.uid])
        self._edit_history.append(edit_rec)
        self._on_focus_list_changed()

    def can_merge_focus_neurons(self) -> bool:
        """
        Can the units currently selected for display/focus purposes be merged into a single unit. The "merge" is
        permitted only if: (1) exactly TWO units are selected for display/focus; and (2) the internal cache file for
        each unit has already been generated. The latter requirement facilitates quickly and reliably undoing the merge
        operation -- by simply reloading the two merged units from their respective cache files.

        :return: True only if the above requirements are met.
        """
        return (len(self._focus_neurons) == 2) and all([unit_cache_file_exists(self._working_directory, uid)
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
        merged_unit = Neuron.merge(units[0], units[1], self._find_next_assignable_unit_index())
        try:
            save_neural_unit_to_cache(self._working_directory, merged_unit, suppress=False)
        except Exception as e:
            print(f"Save to cache failed: {str(e)}")
            return  # TODO: Should report reason operation failed

        # thread conflict: must cancel a COMPUTESTATS task in progress bc its signals can lead to accessing the neural
        # unit list while it is being altered here!
        self._cancel_background_task(TaskType.COMPUTESTATS)

        edit_rec = UserEdit(op=UserEdit.MERGE, params=[units[0].uid, units[1].uid, merged_unit.uid])
        self._focus_neurons.clear()
        for u in units:
            self._neurons.remove(u)  # works bc neurons_with_display_focus contains units from this list
        self._neurons.append(merged_unit)
        self._focus_neurons.append(merged_unit.uid)
        self._edit_history.append(edit_rec)
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
        if (len(self._focus_neurons) == 1) and unit_cache_file_exists(self._working_directory, self._focus_neurons[0]):
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
                unit_cache_file_exists(self._working_directory, focussed[0].uid) and
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

        # thread conflict: must cancel a COMPUTESTATS task in progress bc its signals can lead to accessing the neural
        # unit list while it is being altered here!
        self._cancel_background_task(TaskType.COMPUTESTATS)

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

            idx = self._find_next_assignable_unit_index()
            inside_spikes = np.take(unit.spike_times, inside_indices)
            inside_spikes.sort()
            split_units.append(Neuron(idx, inside_spikes, suffix='x'))
            self._progress_dlg.setValue(97)
            outside_spikes = np.take(unit.spike_times, outside_indices)
            outside_spikes.sort()
            split_units.append(Neuron(idx + 1, outside_spikes, suffix='x'))
            self._progress_dlg.setValue(99)

            try:
                for u in split_units:
                    save_neural_unit_to_cache(self._working_directory, u, suppress=False)
            except Exception as e:
                print(f"Save to cache failed: {str(e)}")
                return  # TODO: Should report reason operation failed
        finally:
            self._progress_dlg.close()

        # success! Update edit history and neuron list and put focus on the two new units
        edit_rec = UserEdit(op=UserEdit.SPLIT, params=[unit.uid, split_units[0].uid, split_units[1].uid])
        self._focus_neurons.clear()
        self._neurons.remove(unit)
        self._neurons.extend(split_units)
        self._focus_neurons.extend([u.uid for u in split_units])
        self._edit_history.append(edit_rec)
        self._on_focus_list_changed()

    def _find_next_assignable_unit_index(self) -> int:
        """
        Find the next available integer that can be assigned to a new unit created by a merge or split operation. This
        method checks the edit history for the "derived" unit with the largest index N and returns N+1. If the edit
        history is empty, then it finds the largest index N among the units in the current unit list.
        :return: The next available integer index.
        """
        max_idx: int = 0
        if len(self._neurons) > 0:
            max_idx = max([u.index for u in self._neurons])
        edit_indices: List[int] = list()
        for edit_rec in self._edit_history:
            if edit_rec.operation == UserEdit.DELETE:
                edit_indices.append(Neuron.dissect_uid(edit_rec.affected_uids)[0])
            elif edit_rec.operation == UserEdit.MERGE:
                edit_indices.append(Neuron.dissect_uid(edit_rec.result_uids)[0])
            elif edit_rec.operation == UserEdit.SPLIT:
                edit_indices.append(Neuron.dissect_uid(edit_rec.result_uids[0])[0])
        if len(edit_indices) > 0:
            max_idx = max(max_idx, max([i for i in edit_indices]))

        return max_idx + 1

    def undo_last_edit(self) -> bool:
        """
        Undo the most recent user-initiated edit to the current neural unit list.
        :return: True if undo succeeds, False otherwise.
        """
        if len(self._edit_history) == 0:
            return False
        edit_rec = self._edit_history.pop()

        # thread conflict: must cancel a COMPUTESTATS task in progress bc its signals can lead to accessing the neural
        # unit list while it is being altered here!
        if edit_rec.operation != UserEdit.LABEL:
            self._cancel_background_task(TaskType.COMPUTESTATS)

        if edit_rec.operation == UserEdit.LABEL:
            for u in self._neurons:
                if u.uid == edit_rec.affected_uids and u.label == edit_rec.unit_label:
                    u.label = edit_rec.previous_unit_label
                    self.neuron_label_updated.emit(u.uid)
                    return True
            return False
        elif edit_rec.operation == UserEdit.DELETE:
            uid = edit_rec.affected_uids
            u = load_neural_unit_from_cache(self._working_directory, uid)
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
                restored_units.append(load_neural_unit_from_cache(self._working_directory, uid))
            if isinstance(merged_idx, int) and all([isinstance(u, Neuron) for u in restored_units]):
                # success: remove the merged unit (including cache file), restore component units, and put them in the
                # focus list
                self._neurons.pop(merged_idx)
                delete_unit_cache_file(self._working_directory, merged_uid)
                self._neurons.extend(restored_units)
                self._focus_neurons.clear()
                self._focus_neurons.extend([u.uid for u in restored_units])
                self._on_focus_list_changed()
                return True
            return False
        else:   # SPLIT
            split_uids = [uid for uid in edit_rec.result_uids]
            unsplit_uid = edit_rec.affected_uids
            restored_unit = load_neural_unit_from_cache(self._working_directory, unsplit_uid)
            split_units: List[Neuron] = list()
            for u in self._neurons:
                if u.uid in split_uids:
                    split_units.append(u)
            if (len(split_units) == 2) and isinstance(restored_unit, Neuron):
                # success: remove the component units, restore the unsplit unit, and put the focus on that unit
                self._focus_neurons.clear()
                for u in split_units:
                    delete_unit_cache_file(self._working_directory, u.uid)
                    self._neurons.remove(u)
                self._neurons.append(restored_unit)
                self._focus_neurons.append(restored_unit.uid)
                self._on_focus_list_changed()
                return True
            return False

    def can_undo_all_edits(self) -> bool:
        """
        Can the edit history be wiped out for the current XSort working directory? This operation is always
        possible, unless the edit history is empty.
        :return True if current edit history is not empty.
        """
        return len(self._edit_history) > 0

    def undo_all_edits(self) -> None:
        """
        Undo all changes made to the contents of the current XSort woring directory, restoring it to its original
        state. No action is taken if the edit history is empty.
        """
        if len(self._edit_history) == 0:
            return

        # thread conflict: must cancel a COMPUTESTATS task in progress bc its signals can lead to accessing the neural
        # unit list while it is being altered here!
        self._cancel_background_task(TaskType.COMPUTESTATS)

        # special case: only unit labels have been changed
        if all([rec.operation == UserEdit.LABEL for rec in self._edit_history]):
            for u in self._neurons:
                u.label = ''
            self._edit_history.clear()
            self._focus_neurons.clear()
            self._on_focus_list_changed()
            return

        # reload all "original" neural units from the spike sorter results file (PKL)
        emsg, neurons = load_spike_sorter_results(self._pkl_file)
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
            unit = load_neural_unit_from_cache(self._working_directory, u.uid)
            self._neurons.append(unit if isinstance(unit, Neuron) else u)

        self._focus_neurons.clear()

        # clear edit history and remove any derived unit cache files
        self._edit_history.clear()
        self.save_edit_history()
        delete_all_derived_unit_cache_files_from(self._working_directory)

        # signal views
        self._on_focus_list_changed()

    def _on_focus_list_changed(self) -> None:
        """
        Whenever the current display focus list changes, we need to cancel any ongoing COMPUTESTATS task before
        signalling the view manager, and then queue a new COMPUTESTATS task (unless the focus list is now empty)
        after a short delay so that the user-facing views have a chance to refresh before that CPU-intensive task
        begins.
        """
        # cancel an ongoing COMPUTESTATS task (if any) and clear out stale PCA projections
        self._cancel_background_task(TaskType.COMPUTESTATS)
        for u in self._neurons:
            u.set_cached_pca_projection(None)

        # signal the view manager and associated views
        self.focus_neurons_changed.emit()

        # if the focus list is not empty, trigger a new COMPUTESTATS task after a brief delay
        if len(self._focus_neurons) > 0:
            QTimer.singleShot(100, self._launch_compute_stats_task)

    def undo_last_edit_description(self) -> Optional[Tuple[str, str]]:
        """
        Get a short and longer description of the most recent user-initiated edit to the current neural unit list -- ie,
        the edit that will be "undone" if :method:`undo_last_edit()` is invoked.
        :return: A 2-tuple (S, L) containing the short and longer descriptions, or None if the edit history is empty.
        """
        try:
            edit_rec = self._edit_history[-1]
            return edit_rec.short_description, edit_rec.longer_description
        except IndexError:
            return None

    @Slot(str, int)
    def on_task_progress(self, desc: str, pct: int) -> None:
        """
        This slot is the mechanism by which :class:`Analyzer` receives progress updates from a task running on a
        background thread. It forwards a progress message to the view manager, and if the modal progress dialog is
        currently visible, it updates the dialog label and progress bar accordingly.

        :param desc: Progress message.
        :param pct: Percent complete. If this lies in [0..100], then "{pct}%" is appended to the progress message.
        :return:
        """
        msg = f"{desc} - {pct}%" if (0 <= pct <= 100) else desc
        self.progress_updated.emit(msg)

        if self._progress_dlg.isVisible():
            self._progress_dlg.setLabelText(msg)
            if 0 <= pct <= 100:
                self._progress_dlg.setValue(pct)

    @Slot(TaskType)
    def on_task_done(self, task_type: TaskType) -> None:
        """
        This slot is the mechanism by which :class:`Analyzer` is notified that a background task has finished. After
        discarding the task object, it signals the view manager that a background task finished.

        :param task_type: Type of task that finished.
        """
        self.progress_updated.emit("")
        self._background_tasks[task_type] = None

    @Slot(str)
    def on_task_failed(self, emsg: str) -> None:
        """ This slot is the mechanism by which :class:`Analyzer` is notified that a background task has failed. """
        print(f"Background task failed: {emsg}")
        # TODO: WHen this happens, the UI needs to raise a modal dialog? How should these errors be handled?

    @Slot(DataType, object)
    def on_data_available(self, data_type: DataType, data: object) -> None:
        """
        This slot is the mechanism by which Analyzer, living in the main GUI thread, receives data objects that are
        prepared/retrieved by a task running on a background thread. Currently, two types of data containers
        are delivered:
            - :class:`Neuron` contains metrics, including the spike train, for a specified neural unit. It also caches
        statistics computed in the background: ISI/ACG/CCG, PCA projection.
            - :class:`ChannelTraceSegment` is a small contiguous segment of a recorded analog trace.
        The retrieved data is stored in an internal member, and a signal is emitted to notify the view controller to
        refresh the GUI appropriately now that the data is immediately available for use.

        :param data_type: Enumeration indicating the type of data made available
        :param data: The data retrieved.
        """
        if (data_type == DataType.NEURON) and isinstance(data, Neuron):
            # neural unit record with SNR, templates and other metrics retrieved from internal cache file
            unit_with_metrics: Neuron = data
            found = False
            for i in range(len(self._neurons)):
                if self._neurons[i].uid == unit_with_metrics.uid:
                    # HACK: The Neuron object is created in the background with the added metrics, but it won't
                    # include the user-specified label. So here we have to restore the label
                    unit_with_metrics.label = self._neurons[i].label
                    self._neurons[i] = unit_with_metrics
                    found = True
                    break
            if found:
                self.data_ready.emit(DataType.NEURON, unit_with_metrics.uid)
        elif (data_type == DataType.CHANNELTRACE) and isinstance(data, ChannelTraceSegment):
            seg: ChannelTraceSegment = data
            if seg.channel_index in self._channel_segments:
                self._channel_segments[seg.channel_index] = seg
                self.data_ready.emit(DataType.CHANNELTRACE, str(seg.channel_index))
        elif (data_type in [DataType.ISI, DataType.ACG, DataType.ACG_VS_RATE, DataType.CCG, DataType.PCA]) and \
                isinstance(data, Neuron):
            # statistics cached in neural unit record on background thread -- NOT supplying a new Neuron instance!
            unit: Neuron = data
            self.data_ready.emit(data_type, unit.uid)
