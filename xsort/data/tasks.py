import math
import pickle
import struct
import time
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple, Optional, List, IO, Any

import numpy as np
import scipy
from PySide6.QtCore import QObject, Slot, Signal, QRunnable

from xsort.data import PL2
from xsort.data.neuron import Neuron, ChannelTraceSegment

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
    pickle files, respectively. On failure, returns (None, None, emgs) -- where the last element is a brief description
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
                    neurons.append(Neuron(i + 1, u['spike_indices__'] / u['sampling_rate__'], False))
                    neurons.append(Neuron(i + 1, u['cs_spike_indices__'] / u['sampling_rate__'], True))
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
            if (not n.is_purkinje()) and purkinje.matching_spike_trains(n):
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


def unit_cache_file_exists(folder: Path, label: str) -> bool:
    """
    Does an internal cache file exist for the specified neural unit in the specified working directory? The unit cache
    file name has the format f'{UNIT_CACHE_FILE_PREFIX}{label}', where {label} is a label string uniquely identifying
    the unit. The method does not validate the contents of the file.
    :param folder: File system path for the current XSort working directory.
    :param label: A label uniquely identifying the unit.
    :return: True if the specified unit cache file is found; else False.
    """
    return Path(folder, f"{UNIT_CACHE_FILE_PREFIX}{label}").is_file()


class TaskType(Enum):
    """ Worker task types. """
    BUILDCACHE = 1,
    """ 
    Process required data sources in XSort working directory and build internal cache of analog channel data 
    streams and neural unit metrics. 
    """
    GETCHANNELS = 2,
    """ Retrieve from internal cache all recorded analog channel traces for a specified time period [t0..t1]. """
    COMPUTEHIST = 3
    """ 
    Compute any missing histograms for a list of units: the ISI and ACG for each unit in the list, and the CCG
    for the each unit in the list vs the other units. The results are cached in the unit instances.
    """


class TaskSignals(QObject):
    """ Defines the signals available from a running worker task thread. """
    progress = Signal(str, int)
    """ Signal emitted to deliver a progress message and integer completion percentage. """
    data_retrieved = Signal(object)
    """ Signal emitted to deliver a data object to the receiver. Data type will depend on specific task. """
    error = Signal(str)
    """ Signal emitted when the worker task has failed for any reason. Argument is an error description. """
    finished = Signal(TaskType)
    """ Signal emitted when the worker task has finished, succesfully or otherwise. Argument is the task type. """


class Task(QRunnable):
    """ A background runnable that handles a specific data analysis or retrieval task. """

    def __init__(self, task_type: TaskType, working_dir: Path, **kwargs):
        """
        Initialize, but do not start, a background task runnoble
        :param task_type: Which type of background task to execute.
        :param working_dir: The XSort working directory on which to operate.
        :param kwargs: Dictionary of keywaord argyments, varying IAW task type. For the TaskType.GETUNIT task,
        kwargs['unit_label'] must be a str identifying the neural unit record to retrieve from internal cache. For the
        TaskType.GETCHANNELS task, kwargs['start'] >= 0 and kwargs['count'] > 0 must be integers defining the starting
        index and size of the contigous channel trace segment to be extracted. For the TaskType.COMPUTEHIST task,
        kw_args['units'] is the list of :class:`Neuron` instances for which ISI/ACG/CCG histograms are to be computed.

        """
        super().__init__()

        self._task_type = task_type
        """ The type of background task executed. """
        self._working_dir = working_dir
        """ The working directory in which required data files and internal XSort cache files are located. """
        self._unit_label: str = kwargs.get('unit_label', '')
        """ For the GETUNIT task only, identifies the neural unit record to retrieve from internal cache. """
        self._start: int = kwargs.get('start', -1)
        """ For the GETCHANNELS task, the index of the first analog sample to retrieve. """
        self._count: int = kwargs.get('count', 0)
        """ For the GETCHANNELS task, the number of analog samples to retrieve. """
        self._units: List[Neuron] = kwargs.get('units', [])
        """ 
        For the COMPUTEHIST task only, a list of neural units for which histograms are to be computed. **NOTE**:
        These are the actual :class:`Neuron` objects living in the GUI, and they are upated in place. This should be
        thread-safe because, once computed and cached, the histograms never change again.
        """
        self.signals = TaskSignals()
        """ The signals emitted by this task. """
        self._info: Optional[Dict[str, Any]] = None
        """ Omniplex PL2 file information. """

    @Slot()
    def run(self):
        """ Perform the specified task. """
        try:
            if self._task_type == TaskType.BUILDCACHE:
                self._build_internal_cache()
            elif self._task_type == TaskType.GETCHANNELS:
                self._get_channel_traces(self._start, self._count)
            elif self._task_type == TaskType.COMPUTEHIST:
                self._compute_histograms()
            else:
                raise Exception("Unrecognized request")
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit(self._task_type)

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
        working directory. It then uses the analog channel caches to compute the per-channel mean spike waveforms, aka
        'templates', and other metrics for each unit, then stores the unit spike train, templates, and other metadata
        in a cache file named f'{UNIT_CACHE_FILE_PREFIX}{U}', where U is a label uniquely identifying the unit. For
        each unit found in the sorter results file:
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
                self.signals.data_retrieved.emit(channel_trace)

        # PHASE 2: Build the neural unit metrics cache
        emsg, neurons = load_spike_sorter_results(sorter_file)
        if len(emsg) > 0:
            raise Exception(emsg)
        n_units = len(neurons)
        while len(neurons) > 0:
            self.signals.progress.emit("Caching/retrieving neural unit metrics",
                                       int(100 * (n_units - len(neurons))/n_units))
            neuron = neurons.pop(0)
            updated_neuron = self._retrieve_neural_unit(neuron.label, suppress=True)
            if updated_neuron is None:
                self._cache_neural_unit(neuron)
                updated_neuron = neuron
            self.signals.data_retrieved.emit(updated_neuron)

    def _get_channel_traces(self, start: int, count: int) -> None:
        """
        Retrieve the specified trace segment for each wideband or narrowband analog channel recorded in the Omniplex
        file in the working directory.
            This method assumes that a cache file already exists holding the the entire data stream for each analog
        channel of interest. It reads the Omniplex PL2 file to prepare the list of relevant channel indices, then
        extracts the trace segment from each channel cache file and delivers it to any receivers via a task signal.

        :param start: Offset to first sample in trace segment to retrieve.
        :param count: Number of samples in segment.
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

        if len(channel_list) == 0:
            raise Exception("No recorded analog channels found in Omniplex file")

        # retrieve all the trace segments
        for idx in channel_list:
            segment = self._retrieve_channel_trace(idx, start, count)
            self.signals.data_retrieved.emit(segment)

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
        :raises: IO or other exception
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

                # update progress roughly once per second
                block_idx += 1
                if (time.time() - t0) > 1:
                    self.signals.progress.emit(f"Cacheing data stream for analog channel {channel_label}...",
                                               int(100.0 * block_idx / num_blocks))
                    t0 = time.time()

        return ChannelTraceSegment(idx, 0, samples_per_sec,
                                   channel_dict['coeff_to_convert_to_units'] * 1.0e6, first_second)

    def _retrieve_neural_unit(self, unit_label: str, suppress: bool = False) -> Optional[Neuron]:
        """
        Retrieve all metrics (spike train, per-channel mean spike waveforms, metadata) for a specified neural unit
        from the corresponding cache file in the working directory.

        :param unit_label: Label uniquely identifying the neural unit.
        :param suppress: If True, any exception (file not found, file IO error) is suppressed. Default is False.
        :return: A **Neuron** object encapsulating all metrics for the specified neural unit, or None if an error
            occurred and exceptions are suppressed.
        :raises Exception: If an error occurs and exceptions are not suppressed. However, an exception is thrown
            regardless if the unit label is invalid.
        """
        # validate unit label and extract unit index. Exception thrown if invalid
        unit_idx, unit_is_complex = Neuron.dissect_unit_label(unit_label)

        try:
            unit_cache_file = Path(self._working_dir, f"{UNIT_CACHE_FILE_PREFIX}{unit_label}")
            if not unit_cache_file.is_file():
                raise Exception(f"Unit metrics cache file missing for neural unit {unit_label}")

            with open(unit_cache_file, 'rb') as fp:
                hdr = struct.unpack_from('f4I', fp.read(struct.calcsize('f4I')))
                ok = (len(hdr) == 5) and all([k >= 0 for k in hdr])
                if not ok:
                    raise Exception(f"Invalid header in unit metrics cache file for neural unit {unit_label}")
                n_bytes = struct.calcsize("<{:d}f".format(hdr[2]))
                spike_times = np.frombuffer(fp.read(n_bytes), dtype='<f')
                template_dict: Dict[int, np.ndarray] = dict()
                template_len = struct.calcsize("<{:d}f".format(hdr[4]))
                for i in range(hdr[3]):
                    channel_index: int = struct.unpack_from('<I', fp.read(struct.calcsize('<I')))[0]
                    template = np.frombuffer(fp.read(template_len), dtype='<f')
                    template_dict[channel_index] = template

                unit = Neuron(unit_idx, spike_times, unit_is_complex)
                unit.update_metrics(hdr[1], hdr[0], template_dict)
                return unit
        except Exception:
            if not suppress:
                raise
            else:
                return None

    def _cache_neural_unit(self, unit: Neuron) -> None:
        """
        Calculate the per-channel mean spike waveforms (aka, 'templates') and other metrics for a given neural unit and
        store the waveforms, spike train and other metadata in a dedicated cache file in thw XSort working directory.
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
        :raises Exception: If an error occurs. In most cases, the exception message may be used as a human-facing
            error description.
        """
        self.signals.progress.emit(f"Computing and cacheing statistics for unit {unit.label} ...", 0)

        omniplex_file, _, emsg = get_required_data_files(self._working_dir)
        if len(emsg) > 0:
            raise Exception(emsg)

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

        # the file where unit metrics will be cached
        unit_cache_file = Path(self._working_dir, f"{UNIT_CACHE_FILE_PREFIX}{unit.label}")

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
                            f"Calculating metrics for unit {unit.label} on Omniplex channel {str(idx)} ... ",
                            int(100.0 * total_progress_frac))
                        t0 = time.time()

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

        # store metrics in cache file
        self.signals.progress.emit(f"Storing metrics for unit {unit.label} to internal cache...", 99)
        unit_cache_file.unlink(missing_ok=True)
        with open(unit_cache_file, 'wb') as dst:
            dst.write(struct.pack('<f4I', best_snr, primary_channel_idx, unit.num_spikes,
                                  len(template_dict), samples_in_template))
            dst.write(unit.spike_times.tobytes())
            for k, t in template_dict.items():
                dst.write(struct.pack('<I', k))  # store source channel index before template!
                dst.write(t.tobytes())

        # update neural unit record in place
        unit.update_metrics(primary_channel_idx, best_snr, template_dict)

    def _compute_histograms(self) -> None:
        """
        Compute the ISI histogram, autocorrelograms, and cross-correlograms for a list of neural units (the
        TaskType.COMPUTEHIST task). The list of units are supplied in the task constructor. The :class:`Neuron` object
        has instance methods to compute the histograms and cache them internally, but these should only be invoked on
        a background because they can take a significant amount of time. Only histograms that have not been cached
        already are computed.
            **IMPORTANT**: This is a time-consuming computation task that does not involve file IO -- unlike the other
        tasks defined in this module. File IO operations release the Python Global Interpreter lock and thus won't
        impact the main GUI thread. But heavy-compute tasks like this one will.
        :raises Exception: If an error occurs. In most cases, the exception message may be used as a human-facing
            error description.
        """
        if len(self._units) == 0:
            return

        num_units = len(self._units)
        num_hists = 2 * num_units + num_units * (num_units - 1)

        self.signals.progress.emit(f"Computing histograms for {len(self._units)} units ...", 0)
        t0 = time.time()
        n = 0
        for u in self._units:
            u.cache_isi_if_necessary()
            time.sleep(0.2)
            n += 1
            u.cache_acg_if_necessary()
            time.sleep(0.2)
            n += 1
            if time.time() - t0 > 1:
                pct = int(100 * n / num_hists)
                self.signals.progress.emit(f"Computing histograms for {len(self._units)} units ...", pct)
                t0 = time.time()
            for u2 in self._units:
                if u.label != u2.label:
                    u.cache_ccg_if_necessary(other_unit=u2)
                    time.sleep(0.2)
                n += 1
                if time.time() - t0 > 1:
                    pct = int(100 * n / num_hists)
                    self.signals.progress.emit(f"Computing histograms for {len(self._units)} units ...", pct)
                    t0 = time.time()
