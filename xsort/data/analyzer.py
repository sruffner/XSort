from enum import Enum
from pathlib import Path
from typing import Union, Optional, Dict, Any, List

from PySide6.QtCore import QObject, Signal, Slot, QThreadPool

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
    TODO: UNDER DEV.  Note that is subclasses QObject so we can define signals!
    """

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
    a channel trace segement.
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
        All one-second channel trace segments, once loaded, start at this sample index relative to the start of
        the Omniplex recording.
        """
        self._pkl_file: Optional[Path] = None
        """ The original spike sorter results file (for now, must be a Python Pickle file). """
        self._neurons: List[Neuron] = list()
        """ 
        List of defined neural units. When a valid working directory is set, this will contain information on the neural
        units identified in the original spiker sorter results file located in that directory.
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
        The elapsed time in seconds at which the current analog channel trace excerpts begin, relaive to the start
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
        task = Task(TaskType.GETCHANNELS, self._working_directory, start=t0, count=self.channel_samples_per_sec)
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

    @Slot()
    def on_task_done(self) -> None:
        self.progress_updated.emit("")

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
