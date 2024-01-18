import math
import pickle
import random
import struct
import time
import traceback
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple, Optional, List, IO, Any

import numpy as np
import scipy
from PySide6.QtCore import QObject, Slot, Signal, QRunnable

from xsort.data import PL2, stats
from xsort.data.edits import UserEdit
from xsort.data.neuron import Neuron, ChannelTraceSegment, DataType

CHANNEL_CACHE_FILE_PREFIX: str = '.xs.ch.'
""" Prefix for Omniplex analog channel data cache file -- followed by the Omniplex channel index. """
UNIT_CACHE_FILE_PREFIX: str = '.xs.unit.'
""" Prefix for a neural unit cache file -- followed by the unit label. """


def get_required_data_files(folder: Path) -> Tuple[Optional[Path], Optional[Path], str]:
    """
    Scan the specified folder for the data source files required by XSort: an Omniplex PL2 file containing the
    multi-channel electrode recording, and a Python pickle file containing the original spike sorter results. Enforce
    the requirement that the folder contain only ONE file of each type.

    :param folder: File system path for the current XSort working directory.
    :return: On success, returns (pl2_file, pkl_file, ""), where the first two elements are the paths of the PL2 and
    pickle files, respectively. On failure, returns (None, None, emsg) -- where the last element is a brief description
    of the error encountered.
    """
    if not isinstance(folder, Path):
        return None, None, "Invalid directory path"
    elif not folder.is_dir():
        return None, None, "Directory not found"
    pl2_file: Optional[Path] = None
    pkl_file: Optional[Path] = None
    for child in folder.iterdir():
        if child.is_file():
            ext = child.suffix.lower()
            if ext in ['.pkl', '.pickle']:
                if pkl_file is None:
                    pkl_file = child
                else:
                    return None, None, "Multiple spike sorter results files (PKL) found"
            elif ext == '.pl2':
                if pl2_file is None:
                    pl2_file = child
                else:
                    return None, None, "Multiple Omniplex files (PL2) found"
    if pl2_file is None:
        return None, None, "No Omniplex file (PL2) found in directory"
    if pkl_file is None:
        return None, None, "No spike sorter results file (PKL) found in directory"

    return pl2_file, pkl_file, ""


def load_spike_sorter_results(sorter_file: Path) -> Tuple[str, Optional[List[Neuron]]]:
    """
    Load the contents of the spike sorter results file specified. **This method strictly applies to the spike sorter
    program utilized in the Lisberger lab, which outputs the sorting algorithm's results to a Python pickle file.**

    :param sorter_file: File system path for the spike sorter results file.
    :return: On success, a tuple ("", L), where L is a list of **Neuron** objects encapsulating the neural units found
    in the file. Certain derived unit metrics -- mean spike waveforms, SNR, primary analog channel -- will be undefined.
    On failure, returns ('error description', None).
    """
    neurons: List[Neuron] = list()
    purkinje_neurons: List[Neuron] = list()  # sublist of Purkinje complex-spike neurons
    try:
        with open(sorter_file, 'rb') as f:
            res = pickle.load(f)
            ok = isinstance(res, list) and all([isinstance(k, dict) for k in res])
            if not ok:
                raise Exception("Unexpected content found")
            for i, u in enumerate(res):
                if u['type__'] == 'PurkinjeCell':
                    neurons.append(Neuron(i + 1, u['spike_indices__'] / u['sampling_rate__'], suffix='s'))
                    neurons.append(Neuron(i + 1, u['cs_spike_indices__'] / u['sampling_rate__'], suffix='c'))
                    purkinje_neurons.append(neurons[-1])
                else:
                    neurons.append(Neuron(i + 1, u['spike_indices__'] / u['sampling_rate__']))
    except Exception as e:
        return f"Unable to read spike sorter results from PKL file: {str(e)}", None

    # the spike sorter algorithm generating the PKL file copies the 'cs_spike_indices__' of a 'PurkinjeCell' type
    # into the 'spike_indices__' of a separate 'Neuron' type. Above, we split the simple and complex spike trains of
    # the sorter's 'PurkinjeCell' into two neural units. We need to remove the 'Neuron' records that duplicate any
    # 'PurkinjeCell' complex spike trains...
    removal_list: List[int] = list()
    for purkinje in purkinje_neurons:
        n: Neuron
        for i, n in enumerate(neurons):
            if (not n.is_purkinje) and purkinje.matching_spike_trains(n):
                removal_list.append(i)
                break
    for idx in sorted(removal_list, reverse=True):
        neurons.pop(idx)

    return "", neurons


def channel_cache_files_exist(folder: Path, channel_indices: List[int]) -> bool:
    """
    Scan specified folder for existing Omniplex analog channel data cache files. The cache file name has the format
    f'{CHANNEL_CACHE_FILE_PREFIX}N', where N is the channel index. The method does not validate the contents of the
    files, which are typically quite large.
    :param folder: File system path for the current XSort working directory.
    :param channel_indices: Unordered list of indices of the analog channels that should be cached.
    :return: True if a cache file is found for each channel specified; False if at least one is missing.
    """
    for i in channel_indices:
        f = Path(folder, f"{CHANNEL_CACHE_FILE_PREFIX}{str(i)}")
        if not f.is_file():
            return False
    return True


def unit_cache_file_exists(folder: Path, uid: str) -> bool:
    """
    Does an internal cache file exist for the specified neural unit in the specified working directory? The unit cache
    file name has the format f'{UNIT_CACHE_FILE_PREFIX}{label}', where {label} is a label string uniquely identifying
    the unit. The method does not validate the contents of the file.
    :param folder: File system path for the current XSort working directory.
    :param uid: A label uniquely identifying the unit.
    :return: True if the specified unit cache file is found; else False.
    """
    return Path(folder, f"{UNIT_CACHE_FILE_PREFIX}{uid}").is_file()


def delete_unit_cache_file(folder: Path, uid: str) -> None:
    """
    Delete the internal cache file for the specified neural unit, if it exists in the specified working directory.
    :param folder: File system path for the current XSort working directory.
    :param uid: A label uniquely identifying the unit.
    """
    p = Path(folder, f"{UNIT_CACHE_FILE_PREFIX}{uid}")
    p.unlink(missing_ok=True)


def delete_all_derived_unit_cache_files_from(folder: Path) -> None:
    """
    Remove all internal cache files for "derived units" (created via merge or split operations) from the specified
    working directory. Unit cache files end with the unit UID, and the UID of every derived unit ends with the letter
    'x' -- so it is a simple task to selectively delete derived unit cache files.
    :param folder: File system path for the current XSort working directory. No action taken if directory invalid.
    """
    if not (isinstance(folder, Path) and folder.is_dir()):
        return
    for child in folder.iterdir():
        if child.is_file() and child.name.endswith('x'):
            child.unlink(missing_ok=True)


def save_neural_unit_to_cache(folder: Path, unit: Neuron, suppress: bool = True) -> bool:
    """
    Save the spike train and other computed metrics for the specified unit to an internal cache file in the specified
    XSort working directory. If a cache file already exists for the unit, it will be overwritten.

    All unit cache files in the working directory start with the same prefix (UNIT_CACHE_FILE_PREFIX) and end with the
    UID of the neural unit. The unit spike times and metrics are written to the binary file as follows:
     - A 20-byte header: [best SNR (f32), primary channel index (i32), total number of spikes in unit's spike times
       array (i32), number of per-channel spike templates (i32), template length (i32)].
     - The byte sequence encoding the spike times array, as generated by np.ndarray.tobytes().
     - For each template: Source channel index (i32) followed by the byte sequence from np.ndarray.tobytes().

    Computing the best SNR (and thereby identifying the unit's "primary channel") and the per-channel mean spike
    template waveforms takes a considerable amount of time. When a derived unit is created by the user via a "merge" or
    "split" operation, it is important to cache the spike times for the new unit immediately (the spike train is not
    persisted anywhere else!); we cannot wait for the metrics to be computed. To this end, the unit cache file can come
    in either of two forms:
     - The "complete" version as described above.
     - The "incomplete" version which stores only the spike train. In this version, the header is [-1, -1, N, 0, 0],
       where N is the number of spikes.


    :param folder: File system path for the XSort working directory.
    :param unit: The neural unit object.
    :param suppress: If True, any exception (bad working directory, file IO error) is suppressed. Default is True.
    :return: True if successful, else False.
    :raises Exception: If an error occurs and exceptions are not suppressed.
    """
    incomplete = (unit.primary_channel is None)
    try:
        unit_cache_file = Path(folder, f"{UNIT_CACHE_FILE_PREFIX}{unit.uid}")
        unit_cache_file.unlink(missing_ok=True)

        with open(unit_cache_file, 'wb') as dst:
            best_snr = -1.0 if incomplete else unit.snr
            primary_channel_idx = -1 if incomplete else unit.primary_channel
            dst.write(struct.pack('<f4I', best_snr, primary_channel_idx, unit.num_spikes,
                                  unit.num_templates, unit.template_length))
            dst.write(unit.spike_times.tobytes())

            for k in unit.template_channel_indices:
                template = unit.get_template_for_channel(k)
                dst.write(struct.pack('<I', k))
                dst.write(template.tobytes())
        return True
    except Exception:
        if not suppress:
            raise
        else:
            return False


def load_neural_unit_from_cache(folder: Path, uid: str, suppress: bool = True) -> Optional[Neuron]:
    """
    Load the specified neural unit from the corresponding internal cache file in the specified working directory. The
    cache file may contain only the unit's spike train (the "incomplete" version) or the spike train along with SNR,
    primary channel index and per-channel mean spike template waveforms (the "complete" version). See
    :method:`save_neural_unit_to_cache` for file format details.

    :param folder: File system path for the XSort working directory.
    :param uid: The neural unit's UID.
    :param suppress: If True, any exception (file not found, file IO error) is suppressed. Default is True.
    :return: A **Neuron** object encapsulating the spike train and any cached metrics for the specified neural unit, or
        None if an error occurred and exceptions are suppressed.
    :raises Exception: If an error occurs and exceptions are not suppressed. However, an exception is thrown
        regardless if the unit label is invalid.
    """
    # validate unit label and extract unit index. Exception thrown if invalid
    unit_idx, unit_suffix = Neuron.dissect_uid(uid)

    try:
        unit_cache_file = Path(folder, f"{UNIT_CACHE_FILE_PREFIX}{uid}")
        if not unit_cache_file.is_file():
            raise Exception(f"Unit metrics cache file missing for neural unit {uid}")

        with open(unit_cache_file, 'rb') as fp:
            hdr = struct.unpack_from('<f4I', fp.read(struct.calcsize('<f4I')))
            if len(hdr) != 5:
                raise Exception(f"Invalid header in unit metrics cache file for neural unit {uid}")
            incomplete = hdr[0] < 0
            n_bytes = struct.calcsize("<{:d}f".format(hdr[2]))
            spike_times = np.frombuffer(fp.read(n_bytes), dtype='<f')
            template_dict: Dict[int, np.ndarray] = dict()
            if not incomplete:
                template_len = struct.calcsize("<{:d}f".format(hdr[4]))
                for i in range(hdr[3]):
                    channel_index: int = struct.unpack_from('<I', fp.read(struct.calcsize('<I')))[0]
                    template = np.frombuffer(fp.read(template_len), dtype='<f')
                    template_dict[channel_index] = template

            unit = Neuron(unit_idx, spike_times, suffix=unit_suffix)
            if not incomplete:
                unit.update_metrics(hdr[1], hdr[0], template_dict)
            return unit
    except Exception:
        if not suppress:
            raise
        else:
            return None


class TaskType(Enum):
    """ Worker task types. """
    BUILDCACHE = 1,
    """ 
    Process required data sources in XSort working directory and build internal cache of analog channel data 
    streams and neural unit metrics. 
    """
    GETCHANNELS = 2,
    """ Retrieve from internal cache all recorded analog channel traces for a specified time period [t0..t1]. """
    COMPUTESTATS = 3
    """
    Compute statistics for each neural unit in the current focus list. Currently, the following statistics are
    computed: the ISI and ACG for each unit, the CCG for each units vs the other units in the list, and the projection
    of each unit's spikes onto a 2D plane defined by principal component analysis of the spike template waveforms. All 
    statistics are cached in the unit instances.
    
        If the internal cache file is missing for any unit in the list, that unit's metrics are computed and the 
    cache file generated before proceeding with the statistics calcs. This takes care of "derived" units that are
    created when the user merges two units or splits one unit into two new units.
    """


class TaskSignals(QObject):
    """ Defines the signals available from a running worker task thread. """
    progress = Signal(str, int)
    """ 
    Signal emitted to deliver a progress message and integer completion percentage. Ignore completion percentage
    if not in [0..100].
    """
    data_available = Signal(DataType, object)
    """ 
    Signal emitted to deliver a data object to the receiver. First argument indicates the type of data retrieved
    (or computed), and the second argument is a container for the data. For the :class:`TaskType`.COMPUTESTATS task, 
    the data object is actually the :class:`Neuron` instance in which the computed statistics are cached.
    """
    error = Signal(str)
    """ Signal emitted when the worker task has failed. Argument is an error description. """
    finished = Signal(TaskType)
    """ Signal emitted when the worker task has finished, succesfully or otherwise. Argument is the task type. """


class Task(QRunnable):
    """ A background runnable that handles a specific data analysis or retrieval task. """

    def __init__(self, task_type: TaskType, working_dir: Path, **kwargs):
        """
        Initialize, but do not start, a background task runnoble.

        :param task_type: Which type of background task to execute.
        :param working_dir: The XSort working directory on which to operate.
        :param kwargs: Dictionary of keywaord argyments, varying IAW task type. For the TaskType.GETCHANNELS task,
            kwargs['start'] >= 0 and kwargs['count'] > 0 must be integers defining the starting index and size of the
            contigous channel trace segment to be extracted. For the TaskType.COMPUTESTATS task, kw_args['units'] is the
            list of :class:`Neuron` instances in the current focus list. Various statistics are cached in these
            instances as they are computed.
        """
        super().__init__()

        self._task_type = task_type
        """ The type of background task executed. """
        self._working_dir = working_dir
        """ The working directory in which required data files and internal XSort cache files are located. """
        self._start: int = kwargs.get('start', -1)
        """ For the GETCHANNELS task, the index of the first analog sample to retrieve. """
        self._count: int = kwargs.get('count', 0)
        """ 
        For the GETCHANNELS task, the number of analog samples to retrieve.
        """
        self._units: List[Neuron] = kwargs.get('units', [])
        """ 
        For the COMPUTESTATS task, this is the list of neural units that currently comprise the 'focus list' and for
        which various statistics are to be computed. **NOTE: These are the actual :class:`Neuron` objects living in 
        the GUI, and the task runner will cache the results of statistical analyses in these objects. Potentially
        not thread-safe!**
        """
        self.signals = TaskSignals()
        """ The signals emitted by this task. """
        self._info: Optional[Dict[str, Any]] = None
        """ Omniplex PL2 file information. """
        self._cancelled = False
        """ Flag set to cancel the task. """

    @Slot()
    def run(self):
        """ Perform the specified task. """
        try:
            if self._task_type == TaskType.BUILDCACHE:
                self._build_internal_cache()
            elif self._task_type == TaskType.GETCHANNELS:
                self._get_channel_traces()
            elif self._task_type == TaskType.COMPUTESTATS:
                self._compute_statistics()
            else:
                raise Exception("Unrecognized request")
        except Exception as e:
            if not self._cancelled:
                traceback.print_exception(e)   # TODO: TESTING
                self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit(self._task_type)

    def cancel(self) -> None:
        """
        Cancel this task, if possible. **NOTE that the task will not stop until the run() method detects the
        cancellation.**
        """
        self._cancelled = True

    def was_cancelled(self) -> bool:
        return self._cancelled

    def _build_internal_cache(self) -> None:
        """
        Scans working directory contents and builds any missing internal cache files. In the process, delivers cached
        data needed by the application's view controller on the GUI thread: the first second's worth of samples on each
        recorded Omniplex analog channe; complete metrics for all identified neural units.
        
        In the first phase, the task extracts all recorded analog wideband and narrowband data streams from the 
        Omniplex PL2 file in the working directory and stores each stream sequentially (int16) in a separate cache
        file named f'{CHANNEL_CACHE_FILE_PREFIX}{N}', where N is the analog channel index. For each recorded analog
        channel:
            - If the cache file already exists, it reads in the first one second's worth of analog samples and delivers
              it to any receivers via a dedicated signal.
            - If the file does not exist, the data stream is extracted and cached, then the first second of data is
              delivered. If the file exists but has the wrong size or the read fails, the file is recreated.

        In the second phase, the task loads neural unit information from the spike sorter results file (.PKL) in the
        working directory. It then applies the edit history, if any, to bring the list of neural units "up to date". It
        then uses the analog channel caches to compute the per-channel mean spike waveforms, aka 'templates', and other
        metrics for each unit, then stores the unit spike train, templates, and other metadata in a cache file named
        f'{UNIT_CACHE_FILE_PREFIX}{U}', where U is a UID uniquely identifying the unit. For each unit:
            - If the corresponding cache file already exists, it loads the unit record and delivers it to any receivers
              via a dedicated signal.
            - If the file does not exist, the unit metrics are prepared and cached, and then the unit record is 
              delivered. If the file exists but has the wrong size or the read fails, the file is recreated.
        
        When processing a new working directory for the first time, this task takes many seconds or minutes to 
        complete. Regular progress updates are signaled roughly once per second. If an error occurs, an exception is
        raised, and the exception message describes the error. When the working directory was processed previously and
        the internal cache is already present, it should take relatively little time to deliver cached channel trace
        segments and neural unit records to the GUI thread.
        """
        self.signals.progress.emit("Scanning working directory for source files and internal cache files...", 0)

        # check for required data files in working directory
        omniplex_file, sorter_file, emsg = get_required_data_files(self._working_dir)
        if len(emsg) > 0:
            raise Exception(emsg)

        # PHASE 1: Build Omniplex analog channel cache
        channel_list: List[int] = list()
        with open(omniplex_file, 'rb', buffering=65536*2) as src:
            # load Omniplex file info and get list of analog channels we care about
            self._info = PL2.load_file_information(src)
            all_channels = self._info['analog_channels']
            for i in range(len(all_channels)):
                if all_channels[i]['num_values'] > 0 and \
                        (all_channels[i]['source'] in [PL2.PL2_ANALOG_TYPE_WB, PL2.PL2_ANALOG_TYPE_SPKC]):
                    channel_list.append(i)

            if self._cancelled:
                raise Exception("Operation cancelled")

            if len(channel_list) == 0:
                raise Exception("No recorded analog channels found in Omniplex file")

            # verify that sampling rate is the same for all the analog channels
            samples_per_sec = self._info['analog_channels'][channel_list[0]]['samples_per_second']
            for idx in channel_list:
                if self._info['analog_channels'][idx]['samples_per_second'] != samples_per_sec:
                    raise Exception('Found at least one analog channel sampled at a different rate in Omniplex file!')

            # cache the (possibly filtered) data stream for each recorded analog channel, if not already cached
            for k, idx in enumerate(channel_list):
                self.signals.progress.emit("Caching/retrieving analog channel data", int(100 * k / len(channel_list)))
                channel_trace = self._retrieve_channel_trace(idx, 0, int(samples_per_sec), suppress=True)
                if channel_trace is None:
                    channel_trace = self._cache_analog_channel(idx, src)
                if self._cancelled:
                    raise Exception("Operation cancelled")
                self.signals.data_available.emit(DataType.CHANNELTRACE, channel_trace)

        # PHASE 2: Build the neural unit metrics cache
        emsg, neurons = load_spike_sorter_results(sorter_file)
        if len(emsg) > 0:
            raise Exception(emsg)
        if self._cancelled:
            raise Exception("Operation cancelled")

        # load and apply edit history to bring the neuron list "up to date"
        emsg, edit_history = UserEdit.load_edit_history(self._working_dir)
        if len(emsg) > 0:
            raise Exception(f"Error reading edit history: {emsg}")
        for edit_rec in edit_history:
            edit_rec.apply_to(neurons)

        n_units = len(neurons)
        while len(neurons) > 0:
            self.signals.progress.emit("Caching/retrieving neural unit metrics",
                                       int(100 * (n_units - len(neurons))/n_units))
            neuron = neurons.pop(0)
            updated_neuron = load_neural_unit_from_cache(self._working_dir, neuron.uid)
            if updated_neuron is None:
                self._cache_neural_unit(neuron)
                updated_neuron = neuron
            if self._cancelled:
                raise Exception("Operation cancelled")
            self.signals.data_available.emit(DataType.NEURON, updated_neuron)

    def _get_channel_traces(self) -> None:
        """
        Retrieve the specified trace segment for each wideband or narrowband analog channel recorded in the Omniplex
        file in the working directory.
            This method assumes that a cache file already exists holding the the entire data stream for each analog
        channel of interest. It reads the Omniplex PL2 file to prepare the list of relevant channel indices, then
        extracts the trace segment from each channel cache file and delivers it to any receivers via a task signal.

        :raises Exception: If unable to read Omniplex file, if a channel cache file is missing, or if a file IO
            error occurs.
        """
        self.signals.progress.emit("Retrieving trace segments from channel caches...", 0)

        # get indices of analog channels recorded from Omniplex file
        omniplex_file, _, emsg = get_required_data_files(self._working_dir)
        if len(emsg) > 0:
            raise Exception(emsg)

        channel_list: List[int] = list()
        with open(omniplex_file, 'rb', buffering=65536 * 2) as src:
            # load Omniplex file info and get list of analog channels we care about
            self._info = PL2.load_file_information(src)
            all_channels = self._info['analog_channels']
            for i in range(len(all_channels)):
                if all_channels[i]['num_values'] > 0 and \
                        (all_channels[i]['source'] in [PL2.PL2_ANALOG_TYPE_WB, PL2.PL2_ANALOG_TYPE_SPKC]):
                    channel_list.append(i)

        if self._cancelled:
            raise Exception("Operation cancelled")
        if len(channel_list) == 0:
            raise Exception("No recorded analog channels found in Omniplex file")

        # retrieve all the trace segments
        for idx in channel_list:
            segment = self._retrieve_channel_trace(idx, self._start, self._count)
            if self._cancelled:
                raise Exception("Operation cancelled")
            self.signals.data_available.emit(DataType.CHANNELTRACE, segment)

    def _retrieve_channel_trace(
            self, idx: int, start: int, count: int, suppress: bool = False) -> Optional[ChannelTraceSegment]:
        """
        Retrieve a small portion of a recorded Omniplex analog channel trace from the corresponding channel cache file
        in the working directory.

        :param idx: The analog channel index.
        :param start: Index of the first sample to retrieve.
        :param count: The number of samples to retrieve.
        :param suppress: If True, any exception (file not found, file IO error) is suppressed. Default is False.
        :return: The requested channel trace segment, or None if an error occurred and exceptions are suppressed.
        """
        try:
            cache_file = Path(self._working_dir, f"{CHANNEL_CACHE_FILE_PREFIX}{str(idx)}")
            if not cache_file.is_file():
                raise Exception(f"Channel cache file missing for analog channel {str(idx)}")

            # check file size
            ch_dict = self._info['analog_channels'][idx]
            expected_size = struct.calcsize("<{:d}h".format(ch_dict['num_values']))
            if cache_file.stat().st_size != expected_size:
                raise Exception(f"Incorrect cache file size for analog channel {str(idx)}: was "
                                f"{cache_file.stat().st_size}, expected {expected_size}")

            if (start < 0) or (count <= 0) or (start + count > ch_dict['num_values']):
                raise Exception("Invalid trace segment bounds")

            samples_per_sec = ch_dict['samples_per_second']
            to_microvolts = ch_dict['coeff_to_convert_to_units'] * 1.0e6
            offset = struct.calcsize("<{:d}h".format(start))
            n_bytes = struct.calcsize("<{:d}h".format(count))
            with open(cache_file, 'rb') as fp:
                fp.seek(offset)
                samples = np.frombuffer(fp.read(n_bytes), dtype='<h')
                return ChannelTraceSegment(idx, start, samples_per_sec, to_microvolts, samples)
        except Exception:
            if not suppress:
                raise
            else:
                return None

    def _cache_analog_channel(self, idx: int, src: IO) -> ChannelTraceSegment:
        """
        Extract the entire data stream for one Omniplex analog channel and store the raw ADC samples (16-bit signed
        integers) sequentially in a dedicated cache file within the XSort working directory. If the specified analog
        channel is wideband, the data trace is bandpass-filtered between 300-8000Hz using a second-order Butterworth
        filter prior to being cached (SciPy package).

            Since the Omniplex analog data is recorded at 40KHz for typically an hour or more, an analog channel data
        stream is quite large. Cacheing the entire stream sequentially allows faster access to smaller chunks (say 1-5
        seconds worth) of the stream.

            Reading, possibly filtering, and writing a multi-MB sequence will take a noticeable amount of time. The
        method will deliver a progress update signal roughly once per second.

        :param idx: Omniplex channel index.
        :param src: The Omniplex file object. The file must be open and is NOT closed on return.
        :returns: A data container holding the channel index and the first second's worth of samples recorded.
        :raises: IO or other exception. Raises a generic exception if task cancelled.
        """
        cache_file = Path(self._working_dir, f"{CHANNEL_CACHE_FILE_PREFIX}{str(idx)}")
        if cache_file.is_file():
            cache_file.unlink(missing_ok=True)

        channel_dict = self._info['analog_channels'][idx]
        num_blocks = len(channel_dict['block_num_items'])
        samples_per_sec: float = channel_dict['samples_per_second']
        block_idx = 0
        is_wideband = (channel_dict['source'] == PL2.PL2_ANALOG_TYPE_WB)
        channel_label = f"{'WB' if is_wideband else 'SPKC'}{channel_dict['channel']:03d}"

        # prepare bandpass filter in case analog signal is wide-band. The filter delays are initialized with zero-vector
        # initial condition and the delays are updated as each block is filtered...
        [b, a] = scipy.signal.butter(2, [2 * 300 / samples_per_sec, 2 * 8000 / samples_per_sec], btype='bandpass')
        filter_ic = scipy.signal.lfiltic(b, a, np.zeros(max(len(b), len(a)) - 1))

        first_second: np.ndarray   # This will hold first second's worth of channel samples
        with open(cache_file, 'wb', buffering=1024*1024) as dst:
            t0 = time.time()
            while block_idx < num_blocks:
                # read in next block of samples and bandpass-filter it if signal is wide-band. Filtering converts the
                # block of samples from int16 to float32, so we need to convert back!
                curr_block: np.ndarray = PL2.load_analog_channel_block_faster(src, idx, block_idx, self._info)
                if is_wideband:
                    curr_block, filter_ic = scipy.signal.lfilter(b, a, curr_block, axis=-1, zi=filter_ic)
                    curr_block = curr_block.astype(np.int16)

                # save the first second's worth of samples
                # TODO: NOTE - We're assuming here that block size > samples_per_sec!!!
                if block_idx == 0:
                    first_second = curr_block[0:int(samples_per_sec)]

                # then write that block to the cache file. NOTE that the blocks are 1D 16-bit signed integer arrays
                dst.write(curr_block.tobytes())

                # update progress roughly once per second. Also check for cancellation
                block_idx += 1
                if (time.time() - t0) > 1:
                    self.signals.progress.emit(f"Cacheing data stream for analog channel {channel_label}...",
                                               int(100.0 * block_idx / num_blocks))
                    t0 = time.time()
                if self._cancelled:
                    dst.close()
                    cache_file.unlink(missing_ok=True)
                    raise Exception("Operation cancelled")

        return ChannelTraceSegment(idx, 0, samples_per_sec,
                                   channel_dict['coeff_to_convert_to_units'] * 1.0e6, first_second)

    def _cache_neural_unit(self, unit: Neuron, allow_cancel: bool = True) -> None:
        """
        Calculate the per-channel mean spike waveforms (aka, 'templates') and other metrics for a given neural unit and
        store the waveforms, spike train and other metadata in a dedicated cache file in the XSort working directory.
            The Lisberger lab's spike sorter algorithm outputs a spike train for each identified neural unit, but it
        does not generate any other metrics. XSort needs to compute the unit's mean spike waveform, or "template", on
        each recorded Omniplex channel, measure SNR, etc.
            A unit's spike template on a given Omniplex analog source channel is calculated by averaging all of the
        10-ms clips [T-1 .. T+9] from the channel data stream surrounding each spike time T. At the same time,
        background noise is measured on that channel to calculate a signal-to-noise ratio (SNR) for that unit on that
        analog channel. The channel for which SNR is greatest is considered the "primary channel" for that unit.
            Since the recorded Omniplex analog data streams are quite long, the template and SNR computations can take
        a while. For efficiency's sake, this work is not done until after the analog data streams have been extracted
        from the Ominplex source file, bandpass-filtered if necessary, and stored sequentially in dedicated cache files
        named IAW the channel index. This method expects the channel cache files to be present, or it fails.
            Progress messages are delivered roughly once per second.

        :param unit: The neural unit record to be cached. This will be an incomplete **Neuron** object, lacking the
            spike templates, SNR, and primary channel designation. If this information is succesfully computed and
            cached, this object is updated accordingly.
        :param allow_cancel: If False, the method will ignore the task cancellation flag. Default = True.
        :raises Exception: If an error occurs. In most cases, the exception message may be used as a human-facing
            error description.
        """
        self.signals.progress.emit(f"Computing and cacheing metrics for unit {unit.uid} ...", 0)

        # get indices of recorded analog channels in ascending order
        channel_list: List[int] = list()
        all_channels = self._info['analog_channels']
        for i in range(len(all_channels)):
            if all_channels[i]['num_values'] > 0 and \
                    (all_channels[i]['source'] in [PL2.PL2_ANALOG_TYPE_WB, PL2.PL2_ANALOG_TYPE_SPKC]):
                channel_list.append(i)
        if len(channel_list) == 0:
            raise Exception("No recorded analog channels found in Omniplex file.")

        # expect the analog sample rate to be the same on all recorded channels we care about. Below we throw an
        # exception if this is not the case...
        samples_per_sec: float = all_channels[channel_list[0]]['samples_per_second']
        samples_in_template = int(samples_per_sec * 0.01)

        # compute mean spike waveform for unit on each recorded analog channel. The analog data stream is read from the
        # corresponding channel cache file, which should be present in the working directory.
        template_dict: Dict[int, np.ndarray] = dict()
        primary_channel_idx = -1
        best_snr: float = 0.0
        block_size = 256 * 1024
        for idx in channel_list:
            ch_dict = self._info['analog_channels'][idx]
            ch_file = Path(self._working_dir, f"{CHANNEL_CACHE_FILE_PREFIX}{str(idx)}")
            if not ch_file.is_file():
                raise Exception(f"Missing internal cache file for recorded data on Omniplex channel {str(idx)}.")
            if ch_dict['samples_per_second'] != samples_per_sec:
                raise Exception('Found at least one analog channel sampled at a different rate in Omniplex file!')

            template = np.zeros(samples_in_template, dtype='<f')
            to_volts: float = ch_dict['coeff_to_convert_to_units']
            t0 = time.time()
            file_size = ch_file.stat().st_size
            num_blocks = math.ceil(file_size / block_size)
            block_medians = np.zeros(num_blocks)
            with open(ch_file, 'rb', buffering=1024 * 1024) as src:
                sample_idx = 0
                spike_idx = num_clips = 0
                prev_block: Optional[np.ndarray] = None
                for i in range(num_blocks):
                    curr_block = np.frombuffer(src.read(block_size), dtype='<h')
                    block_medians[i] = np.median(np.abs(curr_block))  # for later SNR calc

                    # accumulate all spike template clips that are fully contained in the current block OR straddle the
                    # previous and current block
                    num_samples_in_block = len(curr_block)
                    while spike_idx < unit.num_spikes:
                        # clip start and end indices with respect to the current block
                        start = int((unit.spike_times[spike_idx] - 0.001) * samples_per_sec) - sample_idx
                        end = start + samples_in_template
                        if end >= num_samples_in_block:
                            break  # no more spikes fully contained in current block
                        elif start >= 0:
                            template = np.add(template, curr_block[start:end])
                            num_clips += 1
                        elif isinstance(prev_block, np.ndarray):
                            template = np.add(template, np.concatenate((prev_block[start:], curr_block[0:end])))
                            num_clips += 1
                        spike_idx += 1

                    # get ready for next block; update progress roughly once per second
                    prev_block = curr_block
                    sample_idx += num_samples_in_block
                    if (time.time() - t0) > 1:
                        channel_progress_delta = 1 / len(channel_list)
                        total_progress_frac = idx / len(channel_list) + channel_progress_delta * (i + 1) / num_blocks
                        self.signals.progress.emit(
                            f"Calculating metrics for unit {unit.uid} on Omniplex channel {str(idx)} ... ",
                            int(100.0 * total_progress_frac))
                        t0 = time.time()
                    if allow_cancel and self._cancelled:
                        raise Exception("Operation cancelled")

            # prepare mean spike waveform template and compute SNR for this channel. The unit's SNR is the highest
            # recorded per-channel SNR, and the "primary channel" is the one with the highest SNR.
            noise = np.median(block_medians) * 1.4826
            if num_clips > 0:
                template /= num_clips
            snr = (np.max(template) - np.min(template)) / (1.96 * noise)
            if snr > best_snr:
                best_snr = snr
                primary_channel_idx = idx
            template *= to_volts * 1.0e6
            template_dict[idx] = template

            if allow_cancel and self._cancelled:
                raise Exception("Operation cancelled")

        # update neural unit record in place and store in cache file
        unit.update_metrics(primary_channel_idx, best_snr, template_dict)
        save_neural_unit_to_cache(self._working_dir, unit, suppress=False)

    def _compute_statistics(self) -> None:
        """
        Compute all statistics for the list of neural units that currently comprise the focus list on the GUI:
            - Each unit's interspike interval (ISI) histogram.
            - Each unit's autocorrelogram (ACG).
            - The cross-correlogram of each unit with each of the other unit's in the list.
            - Perform principal component analysis on the spike clips across all units in the list and calculate
              the PCA projection of each unit's spike train.

        The list of neural units are supplied in the task constructor and are actually references to the
        :class:`Neuron` objects live on the main GUI thread. :class:`Neuron` instance methods compute the various
        histograms and cache them internally, but these should only be invoked on a background thread because they can
        take a significant amount of time to execute. Only histograms that have not been cached already are computed.

            If the standard metrics (spike template waveforms, SNR, etc) have not been computed yet for any unit in the
        focus list, then those metrics are computed and cached in a unit cache file -- as would be done in the
        BUILDCACHE task. This takes care of cache file generation for "derived units" that are created by a user-
        initiated merge or split operation.

            The principal component analysis can take quite a long time, depending on how many total spikes there are
        across the units in the focus list. The projections are cached in the `Neuron` instances as they are computed,
        but unlike the ISI/ACG/CCG, these only apply to the current focus list. Every time the focus list changes --
        which the user can do at any time on the GUI side -- any previously computed projections are reset.

            **IMPORTANT: Threading concerns.** The computation of ACG/CCGs is a time-consuming operation that does NOT
        involve file IO. File IO operations release the Python Global Interpreter Lock (GIL) and thus won't impact the
        main GUI thread. But heavy-compute tasks like the ACG/CCG will. Hence we sleep briefly after computing each
        ACG or CCG to release the GIL. A similar concern applies to the computations in principal component analysis,
        but some of the time-consuming computations are handled within a third-party library. If the GUI is blocked
        too severely, we may have to offload the computations into a separate process.

        :raises Exception: If an error occurs. In most cases, the exception message may be used as a human-facing
            error description.
        """
        if len(self._units) == 0:
            return

        # read in information from Omniplex file. We need the Omniplex information to perform this task...
        omniplex_file, _, emsg = get_required_data_files(self._working_dir)
        if len(emsg) > 0:
            raise Exception(emsg)
        with open(omniplex_file, 'rb', buffering=65536 * 2) as src:
            self._info = PL2.load_file_information(src)

        # if any unit is missing the standard metrics (like template waveforms), either load them from the corresponding
        # cache file (if it's there) or calculate the metrics and generate the cache. The calcs are SLOW. May not be
        # cancelled here, since we may be cacheing a derived unit!
        u: Neuron
        for i, u in enumerate(self._units):
            if u.primary_channel is None:
                updated_unit = load_neural_unit_from_cache(self._working_dir, u.uid)
                if updated_unit is None:
                    self._cache_neural_unit(u, allow_cancel=False)
                else:
                    self._units[i] = updated_unit
                self.signals.data_available.emit(DataType.NEURON, self._units[i])

        num_units = len(self._units)
        num_hists = 3 * num_units + num_units * (num_units - 1)

        self.signals.progress.emit(f"Computing histograms for {len(self._units)} units ...", 0)
        t0 = time.time()
        n = 0
        for u in self._units:
            if u.cache_isi_if_necessary():
                self.signals.data_available.emit(DataType.ISI, u)
            if self._cancelled:
                raise Exception("Operation cancelled")
            time.sleep(0.2)
            n += 1
            if u.cache_acg_if_necessary():
                self.signals.data_available.emit(DataType.ACG, u)
            if self._cancelled:
                raise Exception("Operation cancelled")
            time.sleep(0.2)
            n += 1
            if u.cache_acg_vs_rate_if_necessary():
                self.signals.data_available.emit(DataType.ACG_VS_RATE, u)
            if self._cancelled:
                raise Exception("Operation cancelled")
            time.sleep(0.2)
            n += 1

            if time.time() - t0 > 1:
                pct = int(100 * n / num_hists)
                self.signals.progress.emit(f"Computing histograms for {len(self._units)} units ...", pct)
                t0 = time.time()

            for u2 in self._units:
                if u.uid != u2.uid:
                    if u.cache_ccg_if_necessary(other_unit=u2):
                        self.signals.data_available.emit(DataType.CCG, u)
                    if self._cancelled:
                        raise Exception("Operation cancelled")
                    time.sleep(0.2)
                    n += 1
                    if time.time() - t0 > 1:
                        pct = int(100 * n / num_hists)
                        self.signals.progress.emit(f"Computing histograms for {len(self._units)} units ...", pct)
                        t0 = time.time()

        self._compute_pca_projection()

    NCLIPS_FOR_PCA: int = 5000
    """
    When only 1 unit is in the focus list, we select this many randomly selected spike multi-clips (horizontal concat
    of spike clip on each of the recorded analog channels) to calculate principal components. When more than one unit
    is selected, PC calculation uses the spike templates for all units.
    """
    SPIKE_CLIP_DUR_SEC: float = 0.002
    """ 
    Spike clip duration in seconds for purposes of principal component analysis. Cannot exceed 10ms, as that is
    the fixed length of the mean spike template waveforms used to compute the PCs.
    """

    PRE_SPIKE_SEC: float = 0.001
    """ 
    Pre-spike interval included in spike clip, in seconds, for purposes of principal component analysis. **Do NOT
    change, as this is the pre-spike interval used to compute all mean spike template waveforms.**
    """

    SAMPLES_PER_CHUNK: int = 65536
    """ Number analog samples read in one go while extracting spike clips from an analog channel cache file. """

    SPIKES_PER_BATCH: int = 20000
    """ Batch size used when projecting all spike clips for a unit to 2D space defined by 2 principal components. """

    def _compute_pca_projection(self) -> None:
        """
        Perform principal component analysis on spike waveform clips across all recorded analog channels for up to 3
        neural units. PCA provides a mechanism for detecting whether distinct neural units actually represent
        incorrectly segregated populations of spikes recorded from the same unit.

        Let N=N1+N2+N3 represent the total number of spikes recorded across all the units (we're assuming 3 units here).
        Let the spike clip size be M analog samples long and the number of analog channels recorded be P. Then every
        spike may be represented by an L=MxP vector, the concatenation of the clips for that spike across the P
        channels. The goal of PCA analysis is to reduce this L-dimensional space down to 2, which can then be easily
        visualized as a 2D scatter plot.

        The first step is to compute the principal components for the N samples in L-dimensional space. A great many of
        these clips will be mostly noise -- since, for every spike, we include the clip from every analog channel, not
        just the primary channel for a give unit. So, instead of using a random sampling of individual clips, we use the
        mean spike template waveforms computed on each channel for each unit. The per-channel spike templates -- which
        should be available already in the :class:`Neuron` instances -- are concatenated to form a KxL matrix, and the
        principal component analysis yields an Lx2 matrix in which the two columns represent the first 2 principal
        components of the data with the greatest variance and therefore most information. **However, if only 1 unit
        is included in the analysis, we revert to using a random sampling of individual clips (because we need at
        least two samples of the L=MxP space in order to compute compute covariance matrix).**

        Then, to compute the PCA projection of unit 1 onto the 2D space defined by these two PCs, we form the N1xL
        matrix representing ALL the individual spike clips for that unit, then multiply that by the Lx2 PCA matrix to
        yield the N1x2 projection. Similarly for the other units.

        Progress messages are delivered regularly as the computation proceeds, and the PCA projection for each specified
        unit is delivered as a 2D Numpy array via the :class:`TaskSignals`.data_retrieved signal. You can think of each
        row in the array as the (x,y) coordinates of each spike in the 2D space defined by the first 2 principal
        components of the analysis.

        All spike clips used in the analysis are 2ms in duration and start 1ms prior to the spike occurrence time.

        :raises Exception: If any required files are missing from the current working directory, such as the
            analog channel data cache files, or if an IO error or other unexpected failure occurs.
        """
        if not (0 < len(self._units) <= 3):
            raise Exception(f"PCA projection requires 1-3 units, not {len(self._units)}!")
        for u in self._units:
            if u.primary_channel is None:
                raise Exception(f"Mean spike templates missing for unit {u.uid}; cannot do PC analaysis.")

        self.signals.progress.emit(f"Computing PCA projections for {len(self._units)} units: "
                                   f"{','.join([u.uid for u in self._units])} ...", -1)

        # the analog channel list. All analog channel cache files should already exist.
        channel_list: List[int] = list()
        all_channels = self._info['analog_channels']
        for i in range(len(all_channels)):
            if all_channels[i]['num_values'] > 0 and \
                    (all_channels[i]['source'] in [PL2.PL2_ANALOG_TYPE_WB, PL2.PL2_ANALOG_TYPE_SPKC]):
                channel_list.append(i)
        if len(channel_list) == 0:
            raise Exception("No recorded analog channels found in Omniplex file")

        samples_per_sec: float = all_channels[channel_list[0]]['samples_per_second']

        # phase 1: compute 2 highest-variance principal components
        time.sleep(0.2)
        all_clips: np.ndarray
        if len(self._units) == 1:
            # when only one unit is selected, chose a randomly selected set of individual multi-clips to calc PC
            u: Neuron = self._units[0]
            n_clips = min(Task.NCLIPS_FOR_PCA, u.num_spikes)
            clip_dur = int(Task.SPIKE_CLIP_DUR_SEC * samples_per_sec)
            if n_clips == u.num_spikes:
                clip_starts = [int((t - Task.PRE_SPIKE_SEC) * samples_per_sec) for t in u.spike_times]
            else:
                spike_indices = sorted(random.sample(range(u.num_spikes), n_clips))
                clip_starts = [int((u.spike_times[i] - Task.PRE_SPIKE_SEC) * samples_per_sec) for i in spike_indices]

            all_clips = np.empty((n_clips, 0), dtype='<h')
            for i, ch_idx in enumerate(channel_list):
                clips = self._retrieve_channel_clips(ch_idx, clip_starts, clip_dur)
                all_clips = np.hstack((all_clips, clips))
                pct = int(100 * (i + 1) / len(channel_list))
                self.signals.progress.emit(f"Retrieving {n_clips} spike clips for unit {u.uid} "
                                           f"across {len(channel_list)} analog channels.", pct)
                if self._cancelled:
                    raise Exception("Operation cancelled")
        else:
            # when multiple units in focus list, use spike template waveform clips to form matrix for PCA: Each row is
            # the horizontal concatenation of the spike templates for a unit across all recorded analog channels. Each
            # row corresponds to a different unit.
            clip_dur = int(Task.SPIKE_CLIP_DUR_SEC * samples_per_sec)
            all_clips = np.zeros((len(self._units), clip_dur*len(channel_list)))
            for i_unit, u in enumerate(self._units):
                for i, ch_idx in enumerate(channel_list):
                    all_clips[i_unit, i*clip_dur:(i+1)*clip_dur] = u.get_template_for_channel(ch_idx)[0:clip_dur]

        self.signals.progress.emit(f"Computing principal components from spike clips/templates...", -1)
        pc_matrix = stats.compute_principal_components(all_clips)
        if self._cancelled:
            raise Exception("Operation cancelled")
        time.sleep(0.2)  # need to yield the GIL after this computation!

        # compute the projection of each unit's spikes onto the 2D space defined by the 2 principal components
        clips_in_chunk = np.zeros((Task.SPIKES_PER_BATCH, clip_dur * len(channel_list)), dtype='<h')
        for i, u in enumerate(self._units):
            self.signals.progress.emit(f"Computing PCA projection for unit {u.uid}...", 0)

            unit_prj = np.zeros((u.num_spikes, 2))
            n_spikes_so_far = 0
            while n_spikes_so_far < u.num_spikes:
                n_spikes_in_chunk = min(Task.SPIKES_PER_BATCH, u.num_spikes - n_spikes_so_far)
                clip_starts = [int((u.spike_times[i+n_spikes_so_far] - Task.PRE_SPIKE_SEC) * samples_per_sec)
                               for i in range(n_spikes_in_chunk)]
                for k, ch_idx in enumerate(channel_list):
                    clips = self._retrieve_channel_clips(ch_idx, clip_starts, clip_dur)
                    clips_in_chunk[0:n_spikes_in_chunk, k*clip_dur:(k+1)*clip_dur] = clips
                    if self._cancelled:
                        raise Exception("Operation cancelled")
                unit_prj[n_spikes_so_far:n_spikes_so_far+n_spikes_in_chunk, :] = \
                    np.matmul(clips_in_chunk[0:n_spikes_in_chunk], pc_matrix)
                if self._cancelled:
                    raise Exception("Operation cancelled")
                time.sleep(0.2)
                n_spikes_so_far += n_spikes_in_chunk

                pct = int(100 * n_spikes_so_far / u.num_spikes)
                self.signals.progress.emit(f"Computing PCA projection for unit {u.uid}...", pct)

                # cache the PCA projection so far and signal it is available so GUI can update
                if n_spikes_so_far < u.num_spikes:
                    u.set_cached_pca_projection(unit_prj[0:n_spikes_so_far, :])
                    self.signals.data_available.emit(DataType.PCA, u)

            u.set_cached_pca_projection(unit_prj)
            self.signals.data_available.emit(DataType.PCA, u)
            if self._cancelled:
                raise Exception("Operation cancelled")

    def _retrieve_channel_clips(self, ch_idx: int, clip_starts: List[int], clip_dur: int) -> np.ndarray:
        """
        Retrieve a series of short "clips" from a recorded Omniplex analog channel trace as stored in the corresponding
        channel cache file in the working directory.

        :param clip_starts: A Nx1 array containing the start index for each clip; MUST be in ascending order!
        :param clip_dur: The duration of each clip in # of analog samples, M.
        :return: An NxM Numpy array of 16-bit integers containing the clips, one per row.
        """
        cache_file = Path(self._working_dir, f"{CHANNEL_CACHE_FILE_PREFIX}{str(ch_idx)}")
        if not cache_file.is_file():
            raise Exception(f"Channel cache file missing for analog channel {str(ch_idx)}")

        # check file size
        ch_dict = self._info['analog_channels'][ch_idx]
        total_samples: int = ch_dict['num_values']
        expected_size = struct.calcsize("<{:d}h".format(total_samples))
        if cache_file.stat().st_size != expected_size:
            raise Exception(f"Incorrect cache file size for analog channel {str(ch_idx)}: was "
                            f"{cache_file.stat().st_size}, expected {expected_size}")

        n_clips = len(clip_starts)
        n_clips_so_far = 0
        out = np.zeros((len(clip_starts), clip_dur), dtype='<h')
        with (open(cache_file, 'rb') as fp):
            # special case: check for a spike clip that starts before recording began or ends after. For these, the
            # portion of the clip that wasn't sampled is set to zeros.
            if clip_starts[0] < 0:
                chunk = np.frombuffer(fp.read(struct.calcsize(f"<{clip_dur+clip_starts[0]}h")))
                out[0, -clip_starts[0]:] = chunk
                n_clips_so_far += 1
            if (clip_starts[-1] + clip_dur) > total_samples:
                fp.seek(struct.calcsize(f"<{clip_starts[-1]}h"))
                chunk = np.frombuffer(fp.read(struct.calcsize(f"<{total_samples-clip_starts[-1]}")))
                out[n_clips-1, 0:len(chunk)] = chunk
                n_clips -= 1

            while n_clips_so_far < n_clips:
                if self._cancelled:
                    raise Exception("Operation cancelled")
                chunk_start = clip_starts[n_clips_so_far]
                fp.seek(struct.calcsize(f"<{chunk_start}h"))
                n_chunk_samples = min(Task.SAMPLES_PER_CHUNK, total_samples - chunk_start)
                chunk = np.frombuffer(fp.read(struct.calcsize(f"<{n_chunk_samples}h")), dtype='<h')
                n_clips_in_chunk = 0
                while (n_clips_so_far + n_clips_in_chunk < n_clips) and \
                        (clip_starts[n_clips_so_far + n_clips_in_chunk] - chunk_start + clip_dur < n_chunk_samples):
                    n_clips_in_chunk += 1
                for k in range(n_clips_in_chunk):
                    chunk_idx = clip_starts[n_clips_so_far + k] - chunk_start
                    out[n_clips_so_far + k, :] = chunk[chunk_idx:chunk_idx+clip_dur]
                n_clips_so_far += n_clips_in_chunk

        return out
