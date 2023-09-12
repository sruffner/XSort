import math
import struct
import time
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple, Optional, List, IO, Any

import numpy as np
import scipy
from PySide6.QtCore import QObject, Slot, Signal, QRunnable

from xsort.data import PL2
from xsort.data.neuron import Neuron

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


def channel_cache_files_exist(folder: Path, channel_map: Dict[str, int]) -> bool:
    """
    Scan specified folder for existing Omniplex analog channel data cache files. The cache file name has the format
    f'{CHANNEL_CACHE_FILE_PREFIX}N', where N is the channel index. The method does not validate the contents of the
    files, which are typically quite large.
    :param folder: File system path for the current XSort working directory.
    :param channel_map: The values of this map contain the indices of the analog channels that should be cached.
    :return: True if a cache file is found for each of the channel indices in **channel_map**; False if at least one
    is missing.
    """
    for i in channel_map.values():
        f = Path(folder, f"{CHANNEL_CACHE_FILE_PREFIX}{str(i)}")
        if not f.is_file():
            return False
    return True


def unit_cache_file_exists(folder: Path, label: str) -> bool:
    f"""
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
    DUMMY = 1,
    """ Temporary 10-second dummy task that reports 'progress' approximately once per second. """
    CACHECHANNELS = 2,
    """ Extract recorded channel data streams from Omniplex and cache in binary file for faster lookup. """
    CACHEUNIT = 3,
    """ Cache computed metadata and per-channel spike template waveforms for an identified neural unit. """
    GETCHANNELS = 4
    """ Get all recorded channel traces for a specified time period [t0..t1]. """


class TaskSignals(QObject):
    """ Defines the signals available from a running worker task thread. """
    finished = Signal(bool, object)
    """ 
    Signal emitted when the worker task has finished, successfully or otherwise. First argument indicates whether or
    not task completed successfully. If False, second argument is an error description (str). Otherwise, second argument
    is the returned data, which will depend on the task type and may be None.
    """
    progress = Signal(str, int)
    """ Signal emitted to deliver a progress message and integer completion percentage. """


class Task(QRunnable):
    """ A background runnable that handles a specific data analysis or retrieval task. """

    _descriptors: Dict[TaskType, str] = {
        TaskType.DUMMY: 'Test task',
        TaskType.CACHECHANNELS: 'Extracting and cacheing Omniplex recorded channels...',
        TaskType.CACHEUNIT: 'Cacheing neural unit information...',
        TaskType.GETCHANNELS: 'Retrieving excerpts of Omniplex recorded channel streams...'
    }

    def __init__(self, task_type: TaskType, working_dir: Path, unit: Optional[Neuron] = None):
        super().__init__()

        self._task_type = task_type
        """ The type of background task executed. """
        self._working_dir = working_dir
        """ The working directory in which required data files and internal XSort cache files are located. """
        self._unit = unit
        """ For the CACHEUNIT task, this is the neural unit for which statistics need to be cached. """
        self.signals = TaskSignals()
        """ The signals emitted by this task. """

    @Slot()
    def run(self):
        """ Perform the specified task. """
        result: object = None
        ok: bool = False
        try:
            if self._task_type == TaskType.DUMMY:
                ok = self._test_task()
            elif self._task_type == TaskType.CACHECHANNELS:
                ok = self._build_recorded_channels_cache()
            elif self._task_type == TaskType.CACHEUNIT:
                ok = self._build_neural_unit_cache()
                if ok:   # TODO: Working on implementation...
                    ok = False
                    raise Exception("TODO: Write unit cache file...")
            else:
                result = "Task not yet implemented!"
        except Exception as e:
            result = f"ERROR ({Task._descriptors[self._task_type]}): {str(e)}"
        finally:
            self.signals.finished.emit(ok, result)

    def _test_task(self) -> bool:
        for i in range(10):
            time.sleep(1)
            self.signals.progress.emit(Task._descriptors[self._task_type], 10 * (i + 1))
        return True

    def _build_recorded_channels_cache(self) -> bool:
        """
        Extracts all recorded analog wideband and narrowband data streams from the Omniplex PL2 file in the working
        directory and store each stream sequentially (int16) in separate cache files.

            Raises an Exception if an error occurs while reading or processing the Omniplex file or writing a channel's
        data data to the dedicated cache file. The exception message is a brief description of the error.

        :return: True if operation succeeded; else False
        """
        self.signals.progress.emit('Extracting and cacheing Omniplex recorded channels...', 0)

        omniplex_file, _, emsg = get_required_data_files(self._working_dir)
        if len(emsg) > 0:
            raise Exception(emsg)

        with open(omniplex_file, 'rb', buffering=65536*2) as src:
            info = PL2.load_file_information(src)
            channel_list: List[int] = list()   # ordered list of indices identifying the recorded channels of interest
            all_channels = info['analog_channels']
            for i in range(len(all_channels)):
                if all_channels[i]['num_values'] > 0 and \
                        (all_channels[i]['source'] in [PL2.PL2_ANALOG_TYPE_WB, PL2.PL2_ANALOG_TYPE_SPKC]):
                    channel_list.append(i)

            for idx in channel_list:
                self._cache_analog_channel_stream(idx, src, info)

        return True

    def _cache_analog_channel_stream(self, idx: int, src: IO, info: Dict[str, Any]) -> None:
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
        :param info: Omniplex file information structure, containing metadata needed to locate the channel data.
        :raises: IO or other exception
        """
        cache_file = Path(self._working_dir, f"{CHANNEL_CACHE_FILE_PREFIX}{str(idx)}")
        if cache_file.is_file():
            cache_file.unlink(missing_ok=True)

        channel_dict = info['analog_channels'][idx]
        num_blocks = len(channel_dict['block_num_items'])
        samples_per_sec: float = channel_dict['samples_per_second']
        block_idx = 0
        is_wideband = (channel_dict['source'] == PL2.PL2_ANALOG_TYPE_WB)
        channel_label = f"{'WB' if is_wideband else 'SPKC'}{channel_dict['channel']:03d}"

        # prepare bandpass filter in case analog signal is wide-band. The filter delays are initialized with zero-vector
        # initial condition and the delays are updated as each block is filtered...
        [b, a] = scipy.signal.butter(2, [2 * 300 / samples_per_sec, 2 * 8000 / samples_per_sec], btype='bandpass')
        filter_ic = scipy.signal.lfiltic(b, a, np.zeros(max(len(b), len(a)) - 1))

        with open(cache_file, 'wb', buffering=1024*1024) as dst:
            t0 = time.time()
            while block_idx < num_blocks:
                # read in next block of samples and bandpass-filter it if signal is wide-band. Filtering converts the
                # block of samples from int16 to float32, so we need to convert back!
                curr_block: np.ndarray = PL2.load_analog_channel_block_faster(src, idx, block_idx, info)
                if is_wideband:
                    curr_block, filter_ic = scipy.signal.lfilter(b, a, curr_block, axis=-1, zi=filter_ic)
                    curr_block = curr_block.astype(np.int16)

                # then write that block to the cache file. NOTE that the blocks are 1D 16-bit signed integer arrays
                dst.write(curr_block.tobytes())

                # update progress roughly once per second
                block_idx += 1
                if (time.time() - t0) > 1:
                    self.signals.progress.emit(f"Cacheing data stream for analog channel {channel_label}...",
                                               int(100.0 * block_idx / num_blocks))
                    t0 = time.time()

    # TODO: CONTINUE HERE 9/12 -- We can safely assume that all analog channels we care about are sampled at the
    #  same rate, so we do not need to store that in neural unit file. It will be implied by the length of each 10ms
    #  template, which is a parameter we should add to unit file.
    def _build_neural_unit_cache(self) -> bool:
        """
        Calculate the per-channel mean spike waveforms (aka, 'templates') and other metrics for a given neural unit and
        store the waveforms, spike train and other metadata in a dedicated cache file in thw XSort eorking directory.
            A unit's spike template on a given Omniplex analog source channel is calculated by averaging all of the
        10-ms clips [T-1 .. T+9] from the channel data stream surrounding each spike time T. At the same time,
        background noise is measured on that channel to calculate a signal-to-noise ratio (SNR) for that unit on that
        analog channel. The channel for which SNR is greatest is considered the "primary channel" for that unit.
            Since the recorded Omniplex analog data streams are quite long, the template and SNR computations can take
        a while. For efficiency's sake, this task is not run until after the analog data streams have been extracted
        from the Ominplex source file, bandpass-filtered if necessary, and stored sequentially in dedicated cache files
        named IAW the channel index. This method expects the channel cache files to be present, or it fails.
            Progress messages are delivered roughly once per second.
        :return: True if operation succeeded; else False
        """
        self.signals.progress.emit(f"Computing and cacheing statistics for unit {self._unit.label} ...", 0)

        omniplex_file, _, emsg = get_required_data_files(self._working_dir)
        if len(emsg) > 0:
            raise Exception(emsg)

        # get Omniplex file info and indices of recorded analog channels in ascending order
        info: Dict[str, Any]
        channel_list: List[int] = list()
        with open(omniplex_file, 'rb', buffering=65536*2) as src:
            info = PL2.load_file_information(src)
            all_channels = info['analog_channels']
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
        unit_cache_file = Path(self._working_dir, f"{UNIT_CACHE_FILE_PREFIX}{self._unit.label}")

        # compute mean spike waveform for unit on each recorded analog channel. The analog data stream is read from the
        # corresponding channel cache file, which should be present in the working directory.
        templates: List[np.ndarray] = list()   # populated in ascending order by index of Omniplex analog channel
        primary_channel_idx: int = -1
        best_snr: float = 0.0
        block_size = 256 * 1024
        for idx in channel_list:
            channel_id = Neuron.omniplex_channel_id(info['analog_channels'][idx]['source'] == PL2.PL2_ANALOG_TYPE_WB,
                                                    info['analog_channels'][idx]['channel'])
            ch_file = Path(self._working_dir, f"{CHANNEL_CACHE_FILE_PREFIX}{str(idx)}")
            if not ch_file.is_file():
                raise Exception(f"Missing internal cache file for recorded data on Omniplex channel {channel_id}.")
            if info['analog_channels'][idx]['samples_per_second'] != samples_per_sec:
                raise Exception('Found at least one analog channel sampled at a different rate in Omniplex file!')

            template = np.zeros(samples_in_template, dtype='<f')
            to_volts: float = info['analog_channels'][idx]['coeff_to_convert_to_units']
            t0 = time.time()
            file_size = ch_file.stat().st_size
            num_blocks = math.ceil(file_size / block_size)
            block_medians = np.zeros(num_blocks)
            with open(ch_file, 'rb', buffering=1024*1024) as src:
                sample_idx = 0
                spike_idx = num_clips = 0
                prev_block: Optional[np.ndarray] = None
                for i in range(num_blocks):
                    curr_block = np.frombuffer(src.read(block_size), dtype='<h')
                    block_medians[i] = np.median(np.abs(curr_block))   # for later SNR calc

                    # TODO: NEED TO REVISE BECAUSE Neuron.spike_times are now in seconds elapsed!!!!!
                    # accumulate all spike template clips that are fully contained in the current block OR straddle the
                    # previous and current block
                    num_samples_in_block = len(curr_block)
                    while spike_idx < self._unit.num_spikes:
                        # clip start and end indices with respect to the current block
                        start = int((self._unit.spike_times[spike_idx] - 0.001) * samples_per_sec) - sample_idx
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
                        channel_progress_delta = 1/len(channel_list)
                        total_progress_frac = idx/len(channel_list) + channel_progress_delta * (i + 1) / num_blocks
                        self.signals.progress.emit(
                            f"Calculating metrics for unit {self._unit.label} on Omniplex channel {channel_id} ... ",
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
            templates.append(template)

        # store metrics in cache file
        self.signals.progress.emit(f"Storing metrics for unit {self._unit.label} to internal cache...", 99)
        unit_cache_file.unlink(missing_ok=True)
        with open(unit_cache_file, 'wb') as dst:
            dst.write(struct.pack('<f4I', best_snr, primary_channel_idx, self._unit.num_spikes,
                                  len(templates), samples_in_template))
            dst.write(self._unit.spike_times.tobytes())
            for t in templates:
                dst.write(t.tobytes())

        # TODO: Save template, snr, and primary channel ID in the Neuron object (here the background thread is
        #  altering an object that is read on the GUI thread!)

        return True
