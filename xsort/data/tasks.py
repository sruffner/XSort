import math
import random
import time
import traceback
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, List, IO

import numpy as np
import scipy
from PySide6.QtCore import QObject, Slot, Signal, QRunnable

from xsort.data import PL2, stats
from xsort.data.edits import UserEdit
from xsort.data.files import CHANNEL_CACHE_FILE_PREFIX, WorkingDirectory
from xsort.data.neuron import Neuron, ChannelTraceSegment, DataType


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

    def __init__(self, task_type: TaskType, working_dir: WorkingDirectory, **kwargs):
        """
        Initialize, but do not start, a background task runnoble.

        :param task_type: Which type of background task to execute.
        :param working_dir: The XSort working directory on which to operate.
        :param kwargs: Dictionary of keyword argyments, varying IAW task type. For the TaskType.GETCHANNELS task,
            kwargs['start'] >= 0 and kwargs['count'] > 0 must be integers defining the starting index and size of the
            contigous channel trace segment to be extracted. For the TaskType.COMPUTESTATS task, kw_args['units'] is the
            list of :class:`Neuron` instances in the current focus list. Various statistics are cached in these
            instances as they are computed.
        """
        super().__init__()

        self._task_type = task_type
        """ The type of background task executed. """
        self._working_dir = working_dir
        """ The XSort working directory in which required source files and internal cache files are located. """
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
                traceback.print_exception(e)
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
        recorded analog data channel; complete metrics for all identified neural units.
        
        In the first phase, the task extracts all recorded analog data streams from the analog source file (either an
        Omniplex PL2 file or a flat binary file) in the working directory and stores each stream sequentially (int16) in
        a separate cache file named f'{CHANNEL_CACHE_FILE_PREFIX}{N}', where N is the analog channel index. For each
        recorded analog channel:
            - If the cache file already exists, it reads in the first one second's worth of analog samples and delivers
              it to any receivers via a dedicated signal.
            - If the file does not exist, the data stream is extracted and cached, then the first second of data is
              delivered. If the file exists but has the wrong size or the read fails, the file is recreated.

        In the second phase, the task loads neural unit information from the neural unit data source file (a Python
        pickle containing the results generated by a spike sorter on the analog data) in the working directory. It then
        applies the edit history, if any, to bring the list of neural units "up to date". It then uses the analog
        channel caches to compute the per-channel mean spike waveforms, aka 'templates', and other metrics for each
        unit, and stores the unit spike train, templates, and other metadata in a cache file named
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

        # PHASE 1: Build analog data channel cache
        channel_list = self._working_dir.analog_channel_indices
        with open(self._working_dir.analog_source, 'rb', buffering=65536*2) as src:
            # cache the (possibly filtered) data stream for each recorded analog channel, if not already cached
            for k, idx in enumerate(channel_list):
                self.signals.progress.emit("Caching/retrieving analog channel data", int(100 * k / len(channel_list)))
                channel_trace = self._working_dir.retrieve_cached_channel_trace(
                    idx, 0, self._working_dir.analog_sampling_rate, suppress=True)
                if channel_trace is None:
                    channel_trace = self._cache_analog_channel(idx, src)
                if self._cancelled:
                    raise Exception("Operation cancelled")
                self.signals.data_available.emit(DataType.CHANNELTRACE, channel_trace)

        # PHASE 2: Build the neural unit metrics cache
        emsg, neurons = self._working_dir.load_neural_units()
        if len(emsg) > 0:
            raise Exception(emsg)
        if self._cancelled:
            raise Exception("Operation cancelled")

        # load and apply edit history to bring the neuron list "up to date"
        emsg, edit_history = UserEdit.load_edit_history(self._working_dir.path)
        if len(emsg) > 0:
            raise Exception(f"Error reading edit history: {emsg}")
        for edit_rec in edit_history:
            edit_rec.apply_to(neurons)
        if self._cancelled:
            raise Exception("Operation cancelled")

        n_units = len(neurons)
        while len(neurons) > 0:
            self.signals.progress.emit("Caching/retrieving neural unit metrics",
                                       int(100 * (n_units - len(neurons))/n_units))
            neuron = neurons.pop(0)
            updated_neuron = self._working_dir.load_neural_unit_from_cache(neuron.uid)
            # if cache file missing OR incomplete (spike times only), then we need to generate it
            if (updated_neuron is None) or (updated_neuron.primary_channel is None):
                self._cache_neural_unit(neuron)
                updated_neuron = neuron
            if self._cancelled:
                raise Exception("Operation cancelled")
            self.signals.data_available.emit(DataType.NEURON, updated_neuron)

    def _get_channel_traces(self) -> None:
        """
        Retrieve the specified trace segment for each recorded analog data channel stream cached in the working
        directory.
            This method assumes that a cache file already exists holding the the entire data stream for each analog
        channel of interest.
        :raises Exception: If a channel cache file is missing, or if a file IO error occurs.
        """
        self.signals.progress.emit("Retrieving trace segments from channel caches...", 0)

        # retrieve all the trace segments
        for idx in self._working_dir.analog_channel_indices:
            segment = self._working_dir.retrieve_cached_channel_trace(idx, self._start, self._count)
            if self._cancelled:
                raise Exception("Operation cancelled")
            self.signals.data_available.emit(DataType.CHANNELTRACE, segment)

    def _cache_analog_channel(self, idx: int, src: IO) -> ChannelTraceSegment:
        """
        Extract the entire data stream for one analog data channel and store the raw ADC samples (16-bit signed ints)
        sequentially in a dedicated cache file within the XSort working directory. If the specified analog channel is
        wideband, the data trace is bandpass-filtered between 300-8000Hz using a second-order Butterworth filter prior
        to being cached (SciPy package).

            Since the multielectrode data is recorded at relatively high sampling rate (40KHz for Omniplex, eg) for
        typically an hour or more, an analog channel data stream is quite large. Cacheing the entire stream sequentially
        allows faster access to smaller chunks (say 1-5 seconds worth) of the stream.

            Reading, possibly filtering, and writing a multi-MB sequence will take a noticeable amount of time. The
        method will deliver a progress update signal roughly once per second.

        :param idx: Analog channel index.
        :param src: Open file stream to analog data source - either an Omniplex PL2 file or a flat binary file. The file
            stream must be open and is NOT closed on return.
        :returns: A data container holding the channel index and the first second's worth of samples recorded.
        :raises: IO or other exception. Raises a generic exception if task cancelled.
        """
        cache_file = Path(self._working_dir.path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(idx)}")
        if cache_file.is_file():
            cache_file.unlink(missing_ok=True)

        # This will hold first second's worth of channel samples, which we need to return
        first_second: np.ndarray
        samples_per_sec = self._working_dir.analog_sampling_rate

        channel_label = self._working_dir.label_for_analog_channel(idx)

        # prepare bandpass filter in case analog signal is wide-band. The filter delays are initialized with zero-vector
        # initial condition and the delays are updated as each block is filtered...
        [b, a] = scipy.signal.butter(2, [2 * 300 / samples_per_sec, 2 * 8000 / samples_per_sec], btype='bandpass')
        filter_ic = scipy.signal.lfiltic(b, a, np.zeros(max(len(b), len(a)) - 1))

        if self._working_dir.uses_omniplex_as_analog_source:
            info = self._working_dir.omniplex_file_info
            channel_dict = info['analog_channels'][idx]
            num_blocks = len(channel_dict['block_num_items'])
            block_idx = 0
            is_wideband = (channel_dict['source'] == PL2.PL2_ANALOG_TYPE_WB)

            with open(cache_file, 'wb', buffering=1024*1024) as dst:
                t0 = time.time()
                while block_idx < num_blocks:
                    # read in next block of samples and bandpass-filter if signal is wide-band. Filtering converts the
                    # block of samples from int16 to float32, so we need to convert back!
                    curr_block: np.ndarray = PL2.load_analog_channel_block_faster(src, idx, block_idx, info)
                    if is_wideband:
                        curr_block, filter_ic = scipy.signal.lfilter(b, a, curr_block, axis=-1, zi=filter_ic)
                        curr_block = curr_block.astype(np.int16)

                    # save the first second's worth of samples. NOTE - We assume block size > samples_per_sec!!!
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
        else:
            # how many samples are stored for each analog channel
            n_ch = self._working_dir.num_analog_channels()
            file_size = self._working_dir.analog_source.stat().st_size
            if file_size % (2 * n_ch) != 0:
                raise Exception(f'Flat binary file size inconsistent with specified number of channels ({n_ch})')
            n_samples = int(file_size / (2 * n_ch))

            interleaved = self._working_dir.is_analog_data_interleaved
            prefiltered = self._working_dir.is_analog_data_prefiltered

            with open(cache_file, 'wb', buffering=1024 * 1024) as dst:
                t0 = time.time()

                # if not interleaved, seek to start of stream for specified channel
                if not interleaved:
                    src.seek(idx * n_samples * 2)
                n_bytes_per_sample = n_ch * 2 if interleaved else 2
                num_samples_read = 0
                while num_samples_read < n_samples:
                    n_samples_to_read = min(samples_per_sec, n_samples - num_samples_read)
                    curr_block = np.frombuffer(src.read(n_samples_to_read * n_bytes_per_sample), dtype='<h')

                    # if interleaved, extract the samples for the channel we're cacheing
                    if interleaved:
                        curr_block = curr_block[idx::n_ch].copy()

                    if not prefiltered:
                        curr_block, filter_ic = scipy.signal.lfilter(b, a, curr_block, axis=-1, zi=filter_ic)
                        curr_block = curr_block.astype(np.int16)

                    # save the first second's worth of samples
                    if num_samples_read == 0:
                        first_second = curr_block

                    # write block to cache file
                    dst.write(curr_block.tobytes())

                    # update progress roughly once per second. Also check for cancellation
                    num_samples_read += n_samples_to_read
                    if (time.time() - t0) > 1:
                        self.signals.progress.emit(f"Cacheing data stream for analog channel {channel_label}...",
                                                   int(100.0 * num_samples_read / n_samples))
                        t0 = time.time()
                    if self._cancelled:
                        dst.close()
                        cache_file.unlink(missing_ok=True)
                        raise Exception("Operation cancelled")

        return ChannelTraceSegment(idx, 0, samples_per_sec, self._working_dir.analog_channel_sample_to_uv(idx),
                                   first_second)

    def _cache_neural_unit(self, unit: Neuron) -> None:
        """
        Calculate the per-channel mean spike waveforms (aka, 'templates') and other metrics for a given neural unit and
        store the waveforms, spike train and other metadata in a dedicated cache file in the XSort working directory.
            The Lisberger lab's spike sorter algorithm outputs a spike train for each identified neural unit, but it
        does not generate any other metrics. XSort needs to compute the unit's mean spike waveform, or "template", on
        each recorded analog data channel, measure SNR, etc.
            A unit's spike template on a given analog data channel is calculated by averaging all of the 10-ms clips
        [T-1 .. T+9] from the channel data stream surrounding each spike time T. At the same time, background noise is
        measured on that channel to calculate a signal-to-noise ratio (SNR) for that unit on that analog channel. The
        channel for which SNR is greatest is considered the "primary channel" for that unit.
            Since the recorded analog data streams are quite long, the template and SNR computations can take a while.
        For efficiency's sake, this work is not done until after the analog data streams have been extracted from the
        analog source file, bandpass-filtered if necessary, and stored sequentially in dedicated cache files named IAW
        the channel index. This method will fail if any channel cache file is missing. **However**, if the analog source
        is a flat binary file containing prefiltered analog data, then the per-channel cache files are not needed, and
        the analog data streams are read directly from the original binary source file.
            Progress messages are delivered roughly once per second.

        :param unit: The neural unit record to be cached. This will be an incomplete **Neuron** object, lacking the
            spike templates, SNR, and primary channel designation. If this information is succesfully computed and
            cached, this object is updated accordingly.
        :raises Exception: If an error occurs. In most cases, the exception message may be used as a human-facing
            error description.
        """
        self.signals.progress.emit(f"Computing and cacheing metrics for unit {unit.uid} ...", 0)

        channel_list = self._working_dir.analog_channel_indices

        # expect the analog sample rate to be the same on all recorded channels we care about. Below we throw an
        # exception if this is not the case...
        samples_per_sec = self._working_dir.analog_sampling_rate
        samples_in_template = int(samples_per_sec * 0.01)

        # compute mean spike waveform for unit on each recorded analog channel and determine primary channel on which
        # best SNR is observed
        template_dict: Dict[int, np.ndarray] = dict()
        primary_channel_idx = -1
        best_snr: float = 0.0

        # CASE 1: The analog data stream is read from the corresponding channel cache file, which must be present.
        if self._working_dir.need_analog_cache:
            block_size = 256 * 1024
            for idx in channel_list:
                ch_file = Path(self._working_dir.path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(idx)}")
                if not ch_file.is_file():
                    raise Exception(f"Missing internal cache file for recorded data on analog channel {str(idx)}.")

                template = np.zeros(samples_in_template, dtype='<f')
                t0 = time.time()
                file_size = ch_file.stat().st_size
                num_blocks = math.ceil(file_size / block_size)
                block_medians: List[float] = list()
                with open(ch_file, 'rb', buffering=1024 * 1024) as src:
                    sample_idx = 0
                    spike_idx = num_clips = 0
                    prev_block: Optional[np.ndarray] = None
                    for i in range(num_blocks):
                        curr_block = np.frombuffer(src.read(block_size), dtype='<h')
                        block_medians.append(np.median(np.abs(curr_block)))  # for later SNR calc

                        # accumulate all spike template clips that are fully contained in the current block OR straddle
                        # the previous and current block
                        num_samples_in_block = len(curr_block)
                        while spike_idx < unit.num_spikes:
                            # clip start and end indices with respect to the current block
                            start = int((unit.spike_times[spike_idx] - 0.001) * samples_per_sec) - sample_idx
                            end = start + samples_in_template
                            if end >= num_samples_in_block:
                                break  # no more spikes fully contained in current block
                            elif start >= 0:
                                np.add(template, curr_block[start:end], out=template)
                                num_clips += 1
                            elif isinstance(prev_block, np.ndarray):
                                np.add(template, np.concatenate((prev_block[start:], curr_block[0:end])), out=template)
                                num_clips += 1
                            spike_idx += 1

                        # get ready for next block; update progress roughly once per second
                        prev_block = curr_block
                        sample_idx += num_samples_in_block
                        if (time.time() - t0) > 1:
                            channel_progress_delta = 1 / len(channel_list)
                            total_progress_frac = idx / len(channel_list)
                            total_progress_frac += channel_progress_delta * (i + 1) / num_blocks
                            self.signals.progress.emit(
                                f"Calculating metrics for unit {unit.uid} on Omniplex channel {str(idx)} ... ",
                                int(100.0 * total_progress_frac))
                            t0 = time.time()
                        if self._cancelled:
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
                template *= self._working_dir.analog_channel_sample_to_uv(idx)
                template_dict[idx] = template

                if self._cancelled:
                    raise Exception("Operation cancelled")

        # CASE 2: Read directly from non-interleaved, prefiltered flat binary file. Channel indices are 0..N-1 and
        # are stored in that order in the file.
        elif not self._working_dir.is_analog_data_interleaved:
            n_ch = self._working_dir.num_analog_channels()
            file_size = self._working_dir.analog_source.stat().st_size
            if file_size % (2 * n_ch) != 0:
                raise Exception(f'Flat binary file size inconsistent with specified number of channels ({n_ch})')
            n_samples = int(file_size / (2 * n_ch))
            n_bytes_per_sample = 2

            # want to read up to 256KB at a time
            n_secs_per_read = 0
            while (n_secs_per_read * samples_per_sec * n_bytes_per_sample) < 256 * 1024:
                n_secs_per_read += 1

            # read the file sequentially from beginning to end. Since the channel data streams are not interleaved,
            # channel 0's data stream comes first, then channnel 1, and so on...
            t0 = time.time()
            with open(self._working_dir.analog_source, 'rb', buffering=1024 * 1024) as src:
                for idx in range(n_ch):
                    template = np.zeros(samples_in_template, dtype='<f')
                    block_medians: List[float] = list()
                    num_samples_read = 0
                    spike_idx = num_clips = 0
                    prev_block: Optional[np.ndarray] = None
                    while num_samples_read < n_samples:
                        n_samples_to_read = min(n_secs_per_read*samples_per_sec, n_samples - num_samples_read)
                        curr_block = np.frombuffer(src.read(n_samples_to_read * n_bytes_per_sample), dtype='<h')

                        block_medians.append(np.median(np.abs(curr_block)))  # for later SNR calc

                        # accumulate all spike template clips that are fully contained in the current block OR straddle
                        # the previous and current block
                        while spike_idx < unit.num_spikes:
                            # clip start and end indices with respect to the current block
                            start = int((unit.spike_times[spike_idx] - 0.001) * samples_per_sec) - num_samples_read
                            end = start + samples_in_template
                            if end >= n_samples_to_read:
                                break  # no more spikes fully contained in current block
                            elif start >= 0:
                                np.add(template, curr_block[start:end], out=template)
                                num_clips += 1
                            elif isinstance(prev_block, np.ndarray):
                                np.add(template, np.concatenate((prev_block[start:], curr_block[0:end])), out=template)
                                num_clips += 1
                            spike_idx += 1

                        # get ready for next block; update progress roughly once per second
                        prev_block = curr_block
                        num_samples_read += n_samples_to_read
                        if (time.time() - t0) > 1:
                            channel_progress_delta = 1 / n_ch
                            total_progress_frac = idx / n_ch + channel_progress_delta * (num_samples_read / n_samples)
                            self.signals.progress.emit(
                                f"Calculating metrics for unit {unit.uid} on analog channel {str(idx)} ... ",
                                int(100.0 * total_progress_frac))
                            t0 = time.time()
                        if self._cancelled:
                            raise Exception("Operation cancelled")

                    # prepare mean spike waveform template and compute SNR for channel. The unit's SNR is the highest
                    # recorded per-channel SNR, and the "primary channel" is the one with the highest SNR.
                    noise = np.median(block_medians) * 1.4826
                    if num_clips > 0:
                        template /= num_clips
                    snr = (np.max(template) - np.min(template)) / (1.96 * noise)
                    if snr > best_snr:
                        best_snr = snr
                        primary_channel_idx = idx
                    template *= self._working_dir.analog_channel_sample_to_uv(idx)
                    template_dict[idx] = template

                    if self._cancelled:
                        raise Exception("Operation cancelled")

        # CASE 3: Read directly from interleaved, prefiltered flat binary file. Because the channels are interleaved,
        # we have to accumulate the templates and noise measures for all channels as we read through the file ONCE. We
        # DON'T want to read through the file N times, where N is the number of channels.
        else:
            n_ch = self._working_dir.num_analog_channels()
            file_size = self._working_dir.analog_source.stat().st_size
            if file_size % (2 * n_ch) != 0:
                raise Exception(f'Flat binary file size inconsistent with specified number of channels ({n_ch})')
            n_samples = int(file_size / (2 * n_ch))
            n_bytes_per_sample = 2 * n_ch   # <== one "sample" in the interleaved case is one scan of the N channels!

            # because we're processing all channels simultaneously, we have to maintain various accumulators/counters
            # for all of the channels at once
            block_medians_dict: Dict[int, List[float]] = dict()
            prev_block_dict: Dict[int, Optional[np.ndarray]] = dict()
            spike_index_dict: Dict[int, int] = dict()
            num_clips_dict: Dict[int, int] = dict()
            for idx in range(n_ch):
                template_dict[idx] = np.zeros(samples_in_template, dtype='<f')
                block_medians_dict[idx] = list()
                prev_block_dict[idx] = None
                spike_index_dict[idx] = 0
                num_clips_dict[idx] = 0

            # want to read up to 256KB at a time
            n_secs_per_read = 0
            while (n_secs_per_read * samples_per_sec * n_bytes_per_sample) < 256 * 1024:
                n_secs_per_read += 1

            t0 = time.time()
            with open(self._working_dir.analog_source, 'rb', buffering=1024 * 1024) as src:
                num_samples_read = 0
                while num_samples_read < n_samples:
                    n_samples_to_read = min(n_secs_per_read*samples_per_sec, n_samples - num_samples_read)
                    curr_block = np.frombuffer(src.read(n_samples_to_read * n_bytes_per_sample), dtype='<h')
                    # curr_block has the next N seconds's worth of interleaved channel samples!

                    for idx in range(n_ch):
                        # extract the interleaved samples for the current channel index
                        block = curr_block[idx::n_ch].copy()
                        block_medians_dict[idx].append(np.median(np.abs(block)))  # for later SNR calc

                        template = template_dict[idx]
                        spike_idx = spike_index_dict[idx]
                        num_clips = num_clips_dict[idx]
                        prev_block = prev_block_dict[idx]
                        while spike_idx < unit.num_spikes:
                            # clip start and end indices with respect to the current block
                            start = int((unit.spike_times[spike_idx] - 0.001) * samples_per_sec) - num_samples_read
                            end = start + samples_in_template
                            if end >= n_samples_to_read:
                                break  # no more spikes fully contained in current block
                            elif start >= 0:
                                np.add(template, block[start:end], out=template)
                                num_clips += 1
                            elif isinstance(prev_block, np.ndarray):
                                np.add(template, np.concatenate((prev_block[start:], block[0:end])), out=template)
                                num_clips += 1
                            spike_idx += 1

                        # remember stuff needed to continue processing with next block
                        prev_block_dict[idx] = block
                        spike_index_dict[idx] = spike_idx
                        num_clips_dict[idx] = num_clips

                    # update progress roughly once per second
                    num_samples_read += n_samples_to_read
                    if (time.time() - t0) > 1:
                        self.signals.progress.emit(
                            f"Calculating metrics for unit {unit.uid} across {n_ch} interleaved channels ... ",
                            int(100.0 * num_samples_read / n_samples))
                        t0 = time.time()
                    if self._cancelled:
                        raise Exception("Operation cancelled")

            # prepare mean spike waveform template and compute SNR for each channel. The unit's SNR is the highest
            # recorded per-channel SNR, and the "primary channel" is the one with the highest SNR.
            for idx in range(n_ch):
                template = template_dict[idx]
                noise = np.median(block_medians_dict[idx]) * 1.4826
                if num_clips_dict[idx] > 0:
                    template /= num_clips_dict[idx]
                snr = (np.max(template) - np.min(template)) / (1.96 * noise)
                if snr > best_snr:
                    best_snr = snr
                    primary_channel_idx = idx
                template *= self._working_dir.analog_channel_sample_to_uv(idx)

                if self._cancelled:
                    raise Exception("Operation cancelled")

        # update neural unit record in place and store in cache file
        unit.update_metrics(primary_channel_idx, best_snr, template_dict)
        if not self._working_dir.save_neural_unit_to_cache(unit):
            raise Exception(f"Error occurred while writing unit metrics to internal cache: uid={unit.uid}")

    def _compute_statistics(self) -> None:
        """
        Compute all statistics for the list of neural units that currently comprise the focus list on the GUI:
            - Each unit's interspike interval (ISI) histogram.
            - Each unit's autocorrelogram (ACG).
            - The cross-correlogram of each unit with each of the other unit's in the list.
            - Perform principal component analysis on the spike clips across all units in the list and calculate
              the PCA projection of each unit's spike train.

        The list of neural units are supplied in the task constructor and are actually references to the
        :class:`Neuron` objects living on the main GUI thread. :class:`Neuron` instance methods compute the various
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

        # if any unit is missing the standard metrics (like template waveforms), either load them from the corresponding
        # cache file (if it's there) or calculate the metrics and generate the cache.
        u: Neuron
        for i, u in enumerate(self._units):
            if u.primary_channel is None:
                updated_unit = self._working_dir.load_neural_unit_from_cache(u.uid)
                if (updated_unit is None) or (updated_unit.primary_channel is None):
                    self._cache_neural_unit(u)
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

    SAMPLES_PER_CHUNK: int = 256 * 1024
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
                raise Exception(f"Mean spike templates missing for unit {u.uid}; cannot do PC analysis.")

        self.signals.progress.emit(f"Computing PCA projections for {len(self._units)} units: "
                                   f"{','.join([u.uid for u in self._units])} ...", -1)

        # the analog channel list. All analog channel cache files should already exist.
        channel_list = self._working_dir.analog_channel_indices
        samples_per_sec = self._working_dir.analog_sampling_rate

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
        Retrieve a series of short "clips" from a recorded analog data channel trace as stored in the corresponding
        channel cache file in the working directory.

        :param ch_idx: The analog channel index.
        :param clip_starts: A Nx1 array containing the start index for each clip; MUST be in ascending order!
        :param clip_dur: The duration of each clip in # of analog samples, M.
        :return: An NxM Numpy array of 16-bit integers containing the clips, one per row.
        :raises Exception if an error occurs (channel cache file missing or invalid, IO error).
        """
        n_clips = len(clip_starts)
        n_clips_so_far = 0
        out = np.zeros((len(clip_starts), clip_dur), dtype='<h')

        # When the original analog source is a prefiltered flat binary file -- either interleaved or not --, there is
        # no reason to create per-channel cache files. Thus, the clips must be extracted from 3 possible source file
        # formats, two of which are non-interleaved.
        src_path: Path
        interleaved = False
        num_samples: int
        n_ch = self._working_dir.num_analog_channels()
        n_bytes_per_sample = 2
        ch_offset = 0   # byte offset to start of specified channel's stream (non-interleaved flat binary file only)
        n_samples_per_chunk_read = Task.SAMPLES_PER_CHUNK

        if self._working_dir.need_analog_cache:
            src_path = Path(self._working_dir.path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(ch_idx)}")
            if not src_path.is_file():
                raise Exception(f"Channel cache file missing for analog channel {str(ch_idx)}")
            num_samples = int(src_path.stat().st_size / n_bytes_per_sample)
        else:
            src_path = self._working_dir.analog_source
            if not src_path.is_file():
                raise Exception(f"Original flat binary analog source file is missing!")
            file_size = self._working_dir.analog_source.stat().st_size
            if file_size % (2 * n_ch) != 0:
                raise Exception(f'Flat binary file size inconsistent with specified number of channels ({n_ch})')
            num_samples = int(file_size / (2 * n_ch))
            interleaved = self._working_dir.is_analog_data_interleaved
            if not interleaved:
                ch_offset = (ch_idx * num_samples) * n_bytes_per_sample
            else:
                # for interleaved case, one "sample" is actually one scan of all N channels
                n_bytes_per_sample = 2 * n_ch
                n_samples_per_chunk_read = int(n_samples_per_chunk_read / n_ch)

        with open(src_path, 'rb', buffering=1024 * 1024) as fp:
            # special case: check for a spike clip that starts before recording began or ends after. For these, the
            # portion of the clip that wasn't sampled is set to zeros. We assume that NO clips ever end before the
            # recording began or start after!!!
            if clip_starts[0] < 0:
                fp.seek(ch_offset)
                chunk = np.frombuffer(fp.read(n_bytes_per_sample * (clip_dur + clip_starts[0])), dtype='<h')
                if interleaved:
                    chunk = chunk[ch_idx::n_ch].copy()
                out[0, -clip_starts[0]:] = chunk
                n_clips_so_far += 1
            if (clip_starts[-1] + clip_dur) > num_samples:
                fp.seek(ch_offset + clip_starts[-1] * n_bytes_per_sample)
                chunk = np.frombuffer(fp.read((num_samples - clip_starts[-1]) * n_bytes_per_sample), dtype='<h')
                if interleaved:
                    chunk = chunk[ch_idx::n_ch].copy()
                out[n_clips - 1, 0:len(chunk)] = chunk
                n_clips -= 1

            while n_clips_so_far < n_clips:
                if self._cancelled:
                    raise Exception("Operation cancelled")
                chunk_start = clip_starts[n_clips_so_far]
                fp.seek(ch_offset + chunk_start * n_bytes_per_sample)

                n_chunk_samples = min(n_samples_per_chunk_read, num_samples - chunk_start)
                chunk = np.frombuffer(fp.read(n_chunk_samples * n_bytes_per_sample), dtype='<h')
                if interleaved:
                    chunk = chunk[ch_idx::n_ch].copy()
                n_clips_in_chunk = 0
                while (n_clips_so_far + n_clips_in_chunk < n_clips) and \
                        (clip_starts[n_clips_so_far + n_clips_in_chunk] - chunk_start + clip_dur < n_chunk_samples):
                    n_clips_in_chunk += 1
                for k in range(n_clips_in_chunk):
                    chunk_idx = clip_starts[n_clips_so_far + k] - chunk_start
                    out[n_clips_so_far + k, :] = chunk[chunk_idx:chunk_idx + clip_dur]
                n_clips_so_far += n_clips_in_chunk

        return out
