from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Tuple

from PySide6.QtCore import QObject, Signal, Slot, QThreadPool, QTimer

from xsort.data import PL2
from xsort.data.edits import UserEdit
from xsort.data.neuron import Neuron, ChannelTraceSegment, DataType
from xsort.data.tasks import (Task, TaskType, get_required_data_files, load_spike_sorter_results,
                              unit_cache_file_exists, load_neural_unit_from_cache, delete_unit_cache_file,
                              delete_all_derived_unit_cache_files_from)


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

    def __init__(self, parent=None):
        super().__init__(parent)
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
        self._edit_history: List[UserEdit] = list()
        """ The edit history, a record of all user-initiated changes to the list of neural units, in chrono order. """
        self._thread_pool = QThreadPool()
        """ Managed thread pool for running slow background tasks. """

        self._background_tasks: Dict[TaskType, Optional[Task]] = {
            TaskType.BUILDCACHE: None, TaskType.COMPUTESTATS: None, TaskType.GETCHANNELS: None
        }

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

    def update_neurons_with_display_focus(self, uid: str) -> None:
        """
        Update the list of neural units currently selected for display/comparison purposes. The specified unit is
        removed from the selection list if it is already there; otherwise, it is appended to the end of the list unless
        _MAX_NUM_FOCUS_NEURONS are already selected. **A signal is emitted whenever the neuron display list changes.**

        :param uid: The UID of the neural unit to be added or removed from the selection list. If the UID is invalid, or
            the current display list is full and does not contain the specified unit, no action is taken.
        """
        if uid in self._focus_neurons:
            self._focus_neurons.remove(uid)
        elif ((len(self._focus_neurons) == self.MAX_NUM_FOCUS_NEURONS) or not
                (uid in [n.uid for n in self._neurons])):
            return
        else:
            self._focus_neurons.append(uid)

        self._on_focus_list_changed()

    def _launch_compute_stats_task(self) -> None:
        self._launch_background_task(TaskType.COMPUTESTATS)

    def _cancel_background_task(self, t: TaskType) -> None:
        if not (self._background_tasks[t] is None):
            self._background_tasks[t].cancel()
            self._background_tasks[t] = None

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

        # spawn task to build internal cache if necessary and/or retrieve the data we require: first second's worth
        # of each relevant analog channel trace, and metrics for all neural units
        self._launch_background_task(TaskType.BUILDCACHE)

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

    def can_delete_or_split_primary_neuron(self) -> bool:
        """
        Can the unit currently selected as the 'primary neuron' be deleted or be split into two units with disjoint
        spike trains? Deletion or splitting is permitted ONLY if: (1) there are no other units in the current display
        focus list; and (2) the unit's internal cache file has already been generated. The latter requirement
        facilitates quickly and reliably undoing a previous delete or split operation -- by simply reloading the unit
        from that cache file.

            NOTE: To perform a split, the user must select a subset of the unit's spikes to be assigned to one of the
        new units, while the remaining are assigned to the other. This requirement is enforced by the view manager.
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
        if not self.can_delete_or_split_primary_neuron():
            return
        try:
            deleted_uid = self._focus_neurons[0]
            deleted_idx = next((i for i in range(len(self._neurons)) if self._neurons[i].uid == deleted_uid), None)
            u = self._neurons.pop(deleted_idx)
            if uid in [n.uid for n in self._neurons]:
                self._focus_neurons[0] = uid
            else:
                self._focus_neurons.pop(0)
            edit_rec = UserEdit(op=UserEdit.DELETE, params=[u.uid])
            self._edit_history.append(edit_rec)
            self._on_focus_list_changed()
        except Exception:
            pass

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

        try:
            units = self.neurons_with_display_focus
            merged_unit = Neuron.merge(units[0], units[1], self._find_next_assignable_unit_index())
            edit_rec = UserEdit(op=UserEdit.MERGE, params=[units[0].uid, units[1].uid, merged_unit.uid])
            self._focus_neurons.clear()
            for u in units:
                self._neurons.remove(u)  # works bc neurons_with_display_focus contains units from this list
            self._neurons.append(merged_unit)
            self._focus_neurons.append(merged_unit.uid)
            self._edit_history.append(edit_rec)
            self._on_focus_list_changed()
        except Exception as e:
            print(f"Merge failed: {str(e)}")
            pass

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
        else:   # TODO: IMPLEMENT - SPLIT
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

        # TODO: STILL NOT RIGHT BC WE NEED TO RELOAD UNIT METRICS FROM CACHE FILES

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
            QTimer.singleShot(0, self._launch_compute_stats_task)

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
        if 0 <= pct <= 100:
            self.progress_updated.emit(f"{desc} - {pct}%")
        else:
            self.progress_updated.emit(desc)

    @Slot(TaskType)
    def on_task_done(self, task_type: TaskType) -> None:
        self.progress_updated.emit("")
        self._cancel_background_task(task_type)

    @Slot(str)
    def on_task_failed(self, emsg: str) -> None:
        print(f"Background task failed: {emsg}")
        # TODO: WHen this happens, the UI needs to raise a modal dialog? How should these errors be handled?

    @Slot(object)
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
