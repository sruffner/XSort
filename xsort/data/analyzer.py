from enum import Enum
from pathlib import Path
from typing import Union, Optional, Dict, Any, List

from PySide6.QtCore import QObject, Signal, Slot, QThreadPool, QTimer

import numpy as np

from xsort.data import PL2
from xsort.data.neuron import Neuron, ChannelTraceSegment
from xsort.data.tasks import Task, TaskType, get_required_data_files, load_spike_sorter_results


class DataType(Enum):
    """ The different types of data objects generated/retrieved by :class:`Analyzer`. """
    NEURON = 1,
    CHANNELTRACE = 2


class Analyzer(QObject):
    """
    The data model manager object for XSort.
    TODO: UNDER DEV.  Note that it subclasses QObject so we can define signals! ...Currently I'm using this object
        for inter-view communication, which should be in ViewManager: the "focus neuron list" and the elapsed time
        epoch currently displayed in the analog channel trace view. But putting these in ViewManager leads to circular
        import issue bc manaager.py imports all the view classes, which would also have to import manager.py!
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
    All views should be refreshed accordingly.
    """
    focus_neuron_stats_updated: Signal = Signal()
    """ 
    Signals that some statistics have been recomputed for one or more neural units currently selected for 
    display/comparison purposes. All views should be refreshed accordingly.
    """
    channel_seg_start_changed: Signal = Signal()
    """ 
    Signals that the elapsed starting time (relative to that start of the electrophysiological recording) for all
    analog channel trace segments has just changed.
    """

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
        """ The labels of the neural units currently selected for display focus, in selection order. """
        self._current_pca_projection: Dict[str, Optional[np.ndarray]] = dict()
        """ 
        Principal component analysis results for the neural units currently selected for display focus. The dictionary
        is emptied every time the focus selection changes, and a background task is launched to perform the analysis
        for the new selection. Each item in the dictionary is a Nx2 Numpy array keyed by the label of a unit in the
        focus list, where N is the number of spikes recorded for that unit. The array can be thought of as the 
        projection of the unit's N spikes in the MxP space of spike waveform clips (M spike clip duration, P 
        different analog channels) onto the first 2 principal components found using a subset of spikes randomly 
        sampled across all of the units in the focus list. Plotted in a 2D scatter plot, this offers a visual
        representation to verify whether or not the selected units are truly distinct.
        """
        self._thread_pool = QThreadPool()
        """ Managed thread pool for running slow background tasks. """

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

        task = Task(TaskType.GETCHANNELS, self._working_directory, start=t0*self.channel_samples_per_sec,
                    count=self.channel_samples_per_sec)
        task.signals.progress.connect(self.on_task_progress)
        task.signals.error.connect(self.on_task_failed)
        task.signals.data_retrieved.connect(self.on_data_retrieved)
        task.signals.finished.connect(self.on_task_done)
        self._thread_pool.start(task)
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
            u = next((n for n in self._neurons if n.label == uid), None)
            if u:
                out.append(u)
        return out

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

    def update_neurons_with_display_focus(self, unit_label: str) -> None:
        """
        Update the list of neural units currently selected for display/comparison purposes. The specified unit is
        removed from the selection list if it is already there; otherwise, it is appended to the end of the list unless
        _MAX_NUM_FOCUS_NEURONS are already selected. **A signal is emitted whenever the neuron display list changes.**

        :param unit_label: Unique label identifying the neural unit to be added or removed from the selection list. If
            the label is invalid, or the current display list is full and does not contain the specified unit, no action
            is taken.
        """
        need_stats = False
        if unit_label in self._focus_neurons:
            self._focus_neurons.remove(unit_label)
        elif ((len(self._focus_neurons) == self.MAX_NUM_FOCUS_NEURONS) or not
                (unit_label in [n.label for n in self._neurons])):
            return
        else:
            self._focus_neurons.append(unit_label)
            for u in self.neurons_with_display_focus:
                need_stats = (len(u.cached_isi) == 0) or (len(u.cached_acg) == 0)
                if not need_stats:
                    need_stats = (
                        any([(u.label != lbl) and (len(u.get_cached_ccg(lbl)) == 0) for lbl in self._focus_neurons]))
                if need_stats:
                    break

            # reset the PCA projections, since the set of units being compared has changed!
            self._current_pca_projection.clear()
            for unit_label in self._focus_neurons:
                self._current_pca_projection[unit_label] = None

        self.focus_neurons_changed.emit()

        # changing the focus list will trigger refreshes across all views. We don't want to launch the CPU-intensive
        # histogram computation task until AFTER those refreshes are done. Hence the delayed launch below.
        if need_stats:
            QTimer.singleShot(0, self._launch_compute_histograms_task)
            # self._launch_compute_histograms_task()

    def _launch_compute_histograms_task(self) -> None:
        focus_list = self.neurons_with_display_focus
        if len(focus_list) == 0:
            return
        task = Task(TaskType.COMPUTEHIST, self._working_directory, units=focus_list)
        task.signals.progress.connect(self.on_task_progress)
        task.signals.error.connect(self.on_task_failed)
        task.signals.data_retrieved.connect(self.on_data_retrieved)
        task.signals.finished.connect(self.on_task_done)
        self._thread_pool.start(task)

    def _launch_compute_pca_task(self) -> None:
        focus_list = self.neurons_with_display_focus
        if len(focus_list) == 0:
            return
        task = Task(TaskType.COMPUTEPCA, self._working_directory, units=focus_list)
        task.signals.progress.connect(self.on_task_progress)
        task.signals.error.connect(self.on_task_failed)
        task.signals.data_retrieved.connect(self.on_data_retrieved)
        task.signals.finished.connect(self.on_task_done)
        self._thread_pool.start(task)

    def pca_projection_for(self, unit_label: str) -> Optional[np.ndarray]:
        """
        Get the current PCA projection for a unit in the current display focus list.

            The primary purpose of XSort is to evaluate the results of initial spike sorting of multi-electrode data
        from the Omniplex MAP system. Principal component analysis (PCA) is one tool to use in that evaluation. In the
        context of the neural units selected for display/comparison, PCA randomly selects N=1000 spikes from the units
        (in proportion to each unit's share of the total number of spikes) and, for each such spike, prepares a row
        vector of L=MxP analog samples, where M is the spike clip duration in #samples and P is the number of different
        analog channels recorded. Each row vector is a 'spike multi-clip' -- the concatenation of the clip for that
        spike on each of the P analog channels. PCA finds the eigenvectors for the NxL matrix with the largest 2
        eigenvalues -- these are the first two "principal components" of the N samples of L-dimensional space containing
        the most "information" about the data. The whole point is to reduce the number of data variables from L to 2.

            To find the PCA projection for each unit in the display list, a KxL matrix is prepared containing the
        spike multi-clips for all K spikes from that unit. Multiplying this by the Lx2 principal component matrix
        yields the Kx2 matrix projection. Plotted in a 2D scatter plot, this offers a visual
        representation to verify whether or not the selected units are truly distinct.

        :param unit_label: Label uniquely identifying a neural unit.
        :return: None if unit label is invalid, if the corresponding neuron is NOT currently selected for display, OR
            if the principal component analysis is in progress in the background and the result is not yet available.
            Otherwise, returns the projection as a Nx2 Numpy array, where N is the number of spikes recorded for that
            unit and each row can be thought of as the (x,y)-coordinates locating each spike in the 2D space defined
            by the first two principal components, as described above.
        """
        if unit_label in self._focus_neurons:
            return self._current_pca_projection.get(unit_label, None)
        else:
            return None

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

        # success
        self._working_directory = _p
        self._pl2_file = pl2_file
        self._pl2_info = pl2_info
        self._channel_segments = channel_map
        self._channel_seg_start = 0
        self._pkl_file = pkl_file
        self._neurons = neurons
        self._focus_neurons.clear()

        # signal views
        self.working_directory_changed.emit()

        # spawn task to build internal cache if necessary and/or retrieve the data we require: first second's worth
        # of each relevant analog channel trace, and metrics for all neural units
        task = Task(TaskType.BUILDCACHE, self._working_directory)
        task.signals.progress.connect(self.on_task_progress)
        task.signals.error.connect(self.on_task_failed)
        task.signals.data_retrieved.connect(self.on_data_retrieved)
        task.signals.finished.connect(self.on_task_done)
        self._thread_pool.start(task)

        return None

    @Slot(str, int)
    def on_task_progress(self, desc: str, pct: int) -> None:
        self.progress_updated.emit(f"{desc} - {pct}%")

    @Slot(TaskType)
    def on_task_done(self, task_type: TaskType) -> None:
        self.progress_updated.emit("")
        if task_type == TaskType.COMPUTEHIST:
            self.focus_neuron_stats_updated.emit()
            self._launch_compute_pca_task()
        elif task_type == TaskType.BUILDCACHE:
            self._launch_compute_histograms_task()
        elif task_type == TaskType.COMPUTEPCA:
            pass

    @Slot(str)
    def on_task_failed(self, emsg: str) -> None:
        print(f"Background task failed: {emsg}")
        # TODO: WHen this happens, the UI needs to raise a modal dialog? How should these errors be handled?

    @Slot(object)
    def on_data_retrieved(self, data: object) -> None:
        """
        This slot is the mechanism by which Analyser, living in the main GUI thread, receives requested data containers
        that are prepared/retrieved by a task running on a background thread. Currently, two types of data containers
        are delivered:
            - :class:`Neuron` contains metrics, including the spike train, for a specified neural unit.
            - :class:`ChannelTraceSegment` is a small contiguous segment of a recorded analog trace.
        The retrieved data is stored in an internal member, and a signal is emitted to notify the view controller to
        refresh the GUI appropriately now that the data is immediately available for use.

        :param data: The data retrieved.
        """
        if isinstance(data, Neuron):
            unit_with_metrics: Neuron = data
            found = False
            for i in range(len(self._neurons)):
                if self._neurons[i].label == unit_with_metrics.label:
                    self._neurons[i] = unit_with_metrics
                    found = True
                    break
            if found:
                self.data_ready.emit(DataType.NEURON, unit_with_metrics.label)
        elif isinstance(data, ChannelTraceSegment):
            seg: ChannelTraceSegment = data
            if seg.channel_index in self._channel_segments:
                self._channel_segments[seg.channel_index] = seg
                self.data_ready.emit(DataType.CHANNELTRACE, str(seg.channel_index))
        elif isinstance(data, tuple):
            if (len(data) == 2) and isinstance(data[0], str) and isinstance(data[1], np.ndarray) and \
                    data[0] in self._current_pca_projection:
                self._current_pca_projection[data[0]] = data[1]
                self.focus_neuron_stats_updated.emit()
