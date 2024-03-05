import random
import time
from pathlib import Path
from queue import Queue, Empty
from threading import Event
from typing import Optional, Dict, List, Tuple

import numpy as np
import scipy

from xsort.data import PL2
from xsort.data.files import WorkingDirectory, CHANNEL_CACHE_FILE_PREFIX, UNIT_CACHE_FILE_PREFIX
from xsort.data.neuron import Neuron, ChannelTraceSegment

"""
NOTES: The methods here were developed during testing of multi-threading (MT) and multi-processing (MP) strategies for 
handling time-consuming tasks in XSort. Generally speaking, mostly IO-bound tasks are suited for an MT approach, while
CPU-bound tasks may finish faster with an MP solution. For example, generating the individual analog data channel cache 
files from an Omniplex source file is a primarily IO-bound task (even though the data streams are typically wideband and
must be bandpass-filtered before caching). For this task, a MT solution using 16 threads was 3x faster than the 
single-threaded solution, and a MP solution did not improve performance much further.

On the other hand, calculating and caching neural unit metrics for N units across M units is a task with both CPU-bound
and IO-bound components, but in this case the CPU-bound component was significant emough to merit an MP approach. For
one sample experiment with 9 neural units and 16 analog channels, the MP approach we settled on here (in which one task
calculates the noise level ond all unit templates on a single analog channel) was on the order of 7x faster than the
MT solution when all spikes were taken into account to compute spike templates. If the number of spikes taken into
account was reduced to 10000 (or less if a unit spike train had fewer spikes than that), the performance improvement
was only 2x -- a reflection of the greater overhead of the MP solution.
"""


def _compute_unit_templates_on_channel(
        work_dir: WorkingDirectory, ch_idx: int, units: List[Neuron], progress_q: Queue, cancel: Event,
        n_max_spks: int = -1) \
        -> Tuple[int, float, Dict[str, np.ndarray]]:
    """
    Helper method for _mp_compute_unit_templates_on_channel() which does most of the work.

    :param work_dir: The working directory.
    :param ch_idx: Index of the analog channel to process.
    :param units: The list of neural units for which metrics are to be computed
    :param progress_q: A process-safe queue for delivering progress updates. Each "update" is in the form of a
        3-tuple (idx, pct, emsg), where idx is the analog channel index (int), pct is the percent complete (int), and
        emsg is an error description if the task has failed (otherwise an empty string).
    :param cancel: A process-safe event object which is set to request premature cancellation of this task function.
    :param n_max_spks: The maximum number of spike waveform clips to process when computing the spike templates.
        If <= 0, then the entire spike train is included in the calculation.
    :return: A 3-tuple [ch_idx, noise, template_dict], where: ch_idx is the index of the analog channel processed;
        noise is the calculated noise level on the channel (needed to compute SNR for each unit); and template_dict is a
        dictionary, keyed by unit UID, holding the computed spike templates. **Both noise level and template samples
        are in raw ADC units (NOT converted to microvolts).**
    :raises Exception: If task fails on an IO or other error.
    """
    # we either read directly from the original, prefiltered flat binary source or an internal cache file for the
    # analog channel index specified.
    if work_dir.need_analog_cache:
        ch_file = Path(work_dir.path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(ch_idx)}")
        if not ch_file.is_file():
            raise Exception(f"Missing internal cache file for recorded data on analog channel {str(ch_idx)}.")
    else:
        ch_file = work_dir.analog_source
        if not ch_file.is_file():
            raise Exception(f"Original flat binary analog source is missing in working directory.")

    samples_per_sec = work_dir.analog_sampling_rate
    template_len = int(samples_per_sec * 0.01)
    n_units = len(units)

    # the template array and the number of clips actually accumulated for each unit (so we can compute average)
    unit_num_clips: List[int] = [0] * n_units
    unit_templates: List[np.ndarray] = [np.zeros(template_len, dtype='<f') for _ in range(n_units)]

    # compute start indices of ALL 10-ms waveform clips to be culled from analog channle stream, across ALL units. Each
    # clip start index is tupled with the corresponding unit index: (clip_start, unit_idx), ...
    clip_starts: List[Tuple[int, int]] = list()
    for unit_idx, u in enumerate(units):
        n_spks = min(u.num_spikes, n_max_spks) if n_max_spks > 0 else u.num_spikes
        if n_spks < u.num_spikes:
            spike_indices = sorted(random.sample(range(u.num_spikes), n_spks))
            unit_clip_starts = [(int((u.spike_times[i] - 0.001) * samples_per_sec), unit_idx)
                                for i in spike_indices]
        else:
            unit_clip_starts = [(int((t - 0.001) * samples_per_sec), unit_idx) for t in u.spike_times]
        clip_starts.extend(unit_clip_starts)

    # clips must be in chronological order!
    clip_starts.sort(key=lambda x: x[0])

    # extract and accumulate all spike waveform clips, across all units, from the source file. If we're dealing with
    # a flat binary file with interleaving, one sample is actually one scan of the N channels recorded!
    n_ch = work_dir.num_analog_channels()
    interleaved = work_dir.is_analog_data_interleaved
    n_bytes_per_sample = 2 * n_ch if interleaved else 2
    block_size = 1024 * 1024
    num_samples_recorded = ch_file.stat().st_size / n_bytes_per_sample
    block_medians: List[float] = list()   # for computing noise level on channel
    with (open(ch_file, 'rb') as src):
        prev_block: Optional[np.ndarray] = None
        clip_idx = 0

        # skip any clips (probably one at most) that start before recording began
        while (clip_starts[clip_idx][0] < 0) and (clip_idx < len(clip_starts)):
            clip_idx += 1

        next_update_pct = 5
        while (clip_idx < len(clip_starts)) and (clip_starts[clip_idx][0] + template_len < num_samples_recorded):
            # seek to start of next clip to be processed, then read in a big block of samples
            sample_idx = clip_starts[clip_idx][0]
            src.seek(sample_idx * n_bytes_per_sample)
            curr_block = np.frombuffer(src.read(block_size), dtype='<h')

            # if reading an interleaved analog source, extract the interleaved samples for the current channel index
            if interleaved:
                curr_block = curr_block[ch_idx::n_ch].copy()
            block_medians.append(np.median(np.abs(curr_block)))  # for later SNR calc

            # accumulate all spike template clips that are fully contained in the current block OR straddle
            # the previous and current block
            num_samples_in_block = len(curr_block)
            while clip_idx < len(clip_starts):
                # clip start and end indices with respect to the current block
                unit_idx = clip_starts[clip_idx][1]
                template: np.ndarray = unit_templates[unit_idx]
                start = clip_starts[clip_idx][0] - sample_idx
                end = start + template_len
                if end >= num_samples_in_block:
                    break  # no more spike clips fully contained in current block
                elif start >= 0:
                    np.add(template, curr_block[start:end], out=template)
                    unit_num_clips[unit_idx] += 1
                elif isinstance(prev_block, np.ndarray):
                    np.add(template, np.concatenate((prev_block[start:], curr_block[0:end])), out=template)
                    unit_num_clips[unit_idx] += 1
                clip_idx += 1

            # get ready for next block, check for cancellation, and report progress
            prev_block = curr_block
            if cancel.is_set():
                raise Exception("Task cancelled")
            pct_done = int(clip_idx * 100 / len(clip_starts))
            if pct_done >= next_update_pct:
                progress_q.put_nowait((ch_idx, pct_done, ""))
                next_update_pct += 5

    # prepare mean spike waveform template and compute SNR for each unit on the specified channel.
    template_dict: Dict[str, np.ndarray] = dict()
    noise = np.median(block_medians) * 1.4826
    for i in range(n_units):
        num_clips = unit_num_clips[i]
        template: np.ndarray = unit_templates[i]
        if num_clips > 0:
            template /= num_clips
        template_dict[units[i].uid] = template

    return ch_idx, noise, template_dict


def _mp_compute_unit_templates_on_channel(dir_path: str, ch_idx: int, progress_q: Queue, cancel: Event,
                                          n_max_spks: int) -> Tuple[str, int, float, Dict[str, np.ndarray]]:
    """
    Calculate the mean spike waveform (aka template) on the specified analog channel for each neural unit found in the
    XSort working directory's neural unit source file.

    Computing a unit's per-channel template requires averaging some or all of the spike waveform clips extracted from
    the analog channel data stream. The method expects the analog cache files to exist, or it fails. The task has both
    an IO-bound (reading the analog cache file) and a CPU-bound component (accumulating clips).

    If a unit's spike train contains hundreds of thousands of spikes, it can take a considerable amount of time to
    compute the template. If there are hundreds of analog channels, this task can take a very long time. To reduce the
    execution time, the third argument limits the number of spikes N included in the analysis. If N is specified and is
    less than the spike train length, the method averages a random sampling of all spike clips to compute the template.

    This function is written for use in a multiprocessing context using multiprocessing.Pool.starmap.async().

    :param dir_path: Full file system path for the working directory
    :param ch_idx: Index of the analog channel to process.
    :param progress_q: A process-safe queue for delivering progress updates. Each "update" is in the form of a
        3-tuple (idx, pct, emsg), where idx is the analog channel index (int), pct is the percent complete (int), and
        emsg is an error description if the task has failed (otherwise an empty string).
    :param cancel: A process-safe event object which is set to request premature cancellation of this task function.
    :param n_max_spks: The maximum number of spike waveform clips to process when computing the spike templates.
        If <= 0, then the entire spike train is included in the calculation.
     :return: A 4-tuple [emsg, ch_idx, noise, template_dict], where: emsg is an error message (str) that is empty if
        the task succeeded; ch_idx is the index of the analog channel processed; noise is the calculated noise level on
        the channel (needed to compute SNR for each unit); and template_dict is a dictionary, keyed by unit UID, holding
        the computed spike templates.  **Both noise level and template samples are in raw ADC units (NOT converted to
        microvolts).** Note that, if the method argument is invalid, the return value will be [emsg, -1, 0, {}].
    """
    try:
        emsg, work_dir = WorkingDirectory.load_working_directory(Path(dir_path))
        if work_dir is None:
            raise Exception(emsg)
        emsg, neurons = work_dir.load_neural_units()
        if len(emsg) > 0:
            raise Exception(emsg)
        if not (ch_idx in work_dir.analog_channel_indices):
            raise Exception("Invalid analog channel index")
        _, noise, template_dict = _compute_unit_templates_on_channel(
            work_dir, ch_idx, neurons, progress_q, cancel, n_max_spks)
        return "", ch_idx, noise, template_dict
    except Exception as e:
        return f"Failed to compute unit templates on channel {ch_idx}: {str(e)}", ch_idx, 0, {}


def _cache_analog_channel(work_dir: WorkingDirectory, idx: int, progress_q: Queue, cancel: Event) -> None:
    """
    Extract the analog data stream for one analog channel in the working directory's analog data source, bandpass-filter
    it if necessary, and store it in a separate internal cache file (".xs.ch.<idx>", where <idx> is the channel index)
    within the directory.

    :param work_dir: The XSort working directory.
    :param idx: The index of the analog channel to be cached.
    :param progress_q: A thread-safe queue for delivering progress updates. Most "updates" are in the form of a 3-tuple
        (idx, pct, emsg), where idx is the analog channel index (int), pct is the percent complete (int), and emsg is
        an error description if the task has failed (otherwise an empty string). However, after retrieving the first
        second's worth of the analog channel recording, the method delivers a 2-tuple (idx, trace), where trace is an
        instance of :class:`ChannelTraceSegment` containing that first second's worth of samples.
    :param cancel: A thread-safe event object which is set to request premature cancellation of this task function. On
        cancellation, the internal cache file is removed.
    """
    # if analog source file is a flat binary file containing prefiltered streams, per-channel caching is unnecessary
    if not work_dir.need_analog_cache:
        return

    cache_file = Path(work_dir.path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(idx)}")
    if cache_file.is_file():
        cache_file.unlink(missing_ok=True)

    samples_per_sec = work_dir.analog_sampling_rate

    # prepare bandpass filter in case analog signal is wide-band. The filter delays are initialized with zero-vector
    # initial condition and the delays are updated as each block is filtered...
    [b, a] = scipy.signal.butter(2, [2 * 300 / samples_per_sec, 2 * 8000 / samples_per_sec], btype='bandpass')
    filter_ic = scipy.signal.lfiltic(b, a, np.zeros(max(len(b), len(a)) - 1))

    next_update_pct = 5
    try:
        with open(work_dir.analog_source, 'rb') as src:
            # CASE 1: Analog source is Omniplex PL2 file
            if work_dir.uses_omniplex_as_analog_source:
                info = work_dir.omniplex_file_info
                channel_dict = info['analog_channels'][idx]
                num_blocks = len(channel_dict['block_num_items'])
                block_idx = 0
                is_wideband = (channel_dict['source'] == PL2.PL2_ANALOG_TYPE_WB)

                with open(cache_file, 'wb', buffering=1024 * 1024) as dst:
                    while block_idx < num_blocks:
                        # read in next block of samples and bandpass-filter if signal is wide-band. Filtering converts
                        # block of samples from int16 to float32, so we need to convert back!
                        curr_block: np.ndarray = PL2.load_analog_channel_block_faster(src, idx, block_idx, info)
                        if is_wideband:
                            curr_block, filter_ic = scipy.signal.lfilter(b, a, curr_block, axis=-1, zi=filter_ic)
                            curr_block = curr_block.astype(np.int16)

                        # then write that block to  cache file. Blocks are 1D 16-bit signed integer arrays
                        dst.write(curr_block.tobytes())

                        # check for cancellation
                        if cancel.is_set():
                            raise Exception("Task cancelled")

                        # deliver a channel trace segment containing the first second's worth of recording (we ASSUME
                        # here that one block of samples is at least one second long
                        if block_idx == 0:
                            trace = ChannelTraceSegment(idx, 0, samples_per_sec,
                                                        work_dir.analog_channel_sample_to_uv(idx),
                                                        curr_block[0:samples_per_sec])
                            progress_q.put_nowait((idx, trace))

                        # update progress after every 5% of blocks processed
                        block_idx += 1
                        if int(block_idx * 100 / num_blocks) >= next_update_pct:
                            progress_q.put_nowait((idx, next_update_pct, ""))
                            next_update_pct += 5

            # CASE 2: Analog source is flat binary file containing raw unfiltered NON-INTERLEAVED streams
            elif not work_dir.is_analog_data_interleaved:
                # how many samples are stored for each analog channel
                n_ch = work_dir.num_analog_channels()
                file_size = work_dir.analog_source.stat().st_size
                if file_size % (2 * n_ch) != 0:
                    raise Exception(f'Flat binary file size inconsistent with specified number of channels ({n_ch})')
                n_samples = int(file_size / (2 * n_ch))
                n_bytes_per_sample = 2

                with open(cache_file, 'wb', buffering=1024 * 1024) as dst:
                    # seek to start of contiguous block containing stream for specified channel
                    # if not interleaved, seek to start of stream for specified channel
                    src.seek(idx * n_samples * 2)

                    num_samples_read = 0
                    while num_samples_read < n_samples:
                        n_samples_to_read = min(3*samples_per_sec, n_samples - num_samples_read)
                        curr_block = np.frombuffer(src.read(n_samples_to_read * n_bytes_per_sample), dtype='<h')

                        # filter the raw stream
                        curr_block, filter_ic = scipy.signal.lfilter(b, a, curr_block, axis=-1, zi=filter_ic)
                        curr_block = curr_block.astype(np.int16)

                        # write block to cache file
                        dst.write(curr_block.tobytes())

                        # check for cancellation
                        if cancel.is_set():
                            raise Exception("Task cancelled")

                        # deliver a channel trace segment containing the first second's worth of recording (note that
                        # here we're reading in 3 seconds at a time.
                        if num_samples_read == 0:
                            trace = ChannelTraceSegment(idx, 0, samples_per_sec,
                                                        work_dir.analog_channel_sample_to_uv(idx),
                                                        curr_block[0:samples_per_sec])
                            progress_q.put_nowait((idx, trace))

                        # update progress after every 5% of the stream has been cached
                        # update progress roughly once per second. Also check for cancellation
                        num_samples_read += n_samples_to_read
                        if int(num_samples_read * 100 / n_samples) >= next_update_pct:
                            progress_q.put_nowait((idx, next_update_pct, ""))
                            next_update_pct += 5

            # CASE 3: Analog source is flat binary file containing raw unfiltered INTERLEAVED streams
            else:
                # how many samples are stored for each analog channel
                n_ch = work_dir.num_analog_channels()
                file_size = work_dir.analog_source.stat().st_size
                if file_size % (2 * n_ch) != 0:
                    raise Exception(f'Flat binary file size inconsistent with specified number of channels ({n_ch})')
                n_samples = int(file_size / (2 * n_ch))
                interleaved = work_dir.is_analog_data_interleaved

                with open(cache_file, 'wb', buffering=1024 * 1024) as dst:
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

                        # filter the raw stream
                        curr_block, filter_ic = scipy.signal.lfilter(b, a, curr_block, axis=-1, zi=filter_ic)
                        curr_block = curr_block.astype(np.int16)

                        # write block to cache file
                        dst.write(curr_block.tobytes())

                        # check for cancellation
                        if cancel.is_set():
                            raise Exception("Task cancelled")

                        # deliver a channel trace segment containing the first second's worth of recording (note that
                        # here we're reading in one second at a time.
                        if num_samples_read == 0:
                            trace = ChannelTraceSegment(idx, 0, samples_per_sec,
                                                        work_dir.analog_channel_sample_to_uv(idx),
                                                        curr_block[0:samples_per_sec])
                            progress_q.put_nowait((idx, trace))

                        # update progress after every 5% of the stream has been cached
                        # update progress roughly once per second. Also check for cancellation
                        num_samples_read += n_samples_to_read
                        if int(num_samples_read * 100 / n_samples) >= next_update_pct:
                            progress_q.put_nowait((idx, next_update_pct, ""))
                            next_update_pct += 5

    except Exception as e:
        progress_q.put_nowait((idx, next_update_pct, str(e)))
        cache_file.unlink(missing_ok=True)


def read_and_filer_interleaved_analog_source(
        work_dir: WorkingDirectory, channel_qs: List[Queue], progress_q: Queue, cancel: Event) -> None:
    """
    Digest the specified XSort working directory's binary analog source file containing unfiltered, interleaved data.

    Reads the data in one-second blocks, disentangles the streams into N separate blocks containing the one-second
    trace for each channel [0 .. N-1], filters the blocks, then pushes them onto one of 10 FIFO queues, where queue K
    takes blocks from each channel M such that M % 10 == K. Each block is delivered as a 2-tuple (ch_idx, buf), where
    ch_idx is the analog channel index and buf is a byte buffer holding the one-second block, ready to be written
    directly to the corresponding cache file. To signal the end of data, the method will deliver (-1, None) on each
    queue before returning.

    The caller is responsible for setting up the channel queues, as well as assiging writer threads to service each
    queue.

    :param work_dir: The XSort working directory.
    :param channel_qs: The 10 queues on which the channel blocks are streamed to writer threads, as described.
    :param progress_q: A thread-safe queue for delivering progress updates. Each update is in the form of a 2-tuple
        (pct, emsg), where pct is the percent complete (int), and emsg is an error description if the task has failed
        (otherwise an empty string).
    :param cancel: A thread-safe event object which is set to request premature cancellation of this task function. On
        cancellation, the method puts the end-of-data marker on each of the channel queues and returns.
    """
    n_ch = work_dir.num_analog_channels()
    samples_per_sec = work_dir.analog_sampling_rate

    # prepare bandpass filter in case analog signal is wide-band. The filter delays are initialized with zero-vector
    # initial condition and the delays are updated as each block is filtered. SO WE NEED TO MAINTAIN A SEPARATE FILTER
    # DELAY FOR EACH CHANNEL
    [b, a] = scipy.signal.butter(2, [2 * 300 / samples_per_sec, 2 * 8000 / samples_per_sec], btype='bandpass')
    filter_ic = scipy.signal.lfiltic(b, a, np.zeros(max(len(b), len(a)) - 1))
    delays: List[np.ndarray] = [filter_ic.copy() for _ in range(n_ch)]

    next_update_pct = 5
    try:
        file_size = work_dir.analog_source.stat().st_size
        if file_size % (2 * n_ch) != 0:
            raise Exception(f'Flat binary file size inconsistent with specified number of channels ({n_ch})')
        n_samples = int(file_size / (2 * n_ch))
        n_bytes_per_sample = n_ch * 2

        with open(work_dir.analog_source, 'rb') as src:
            num_samples_read = 0
            while num_samples_read < n_samples:
                # read in one-second at a time -- this is a BIG chunk if there are many channels and high sample rate
                n_samples_to_read = min(samples_per_sec, n_samples - num_samples_read)
                curr_block = np.frombuffer(src.read(n_samples_to_read * n_bytes_per_sample), dtype='<h')

                curr_block = curr_block.reshape(-1, n_ch)
                curr_block = curr_block.transpose()

                # disentangle the channel streams, filter, and deliver to the channel queues
                for idx in range(n_ch):
                    ch_block = curr_block[idx].copy()
                    ch_block, delays[idx] = scipy.signal.lfilter(b, a, ch_block, axis=-1, zi=delays[idx])
                    ch_block = ch_block.astype(np.int16)
                    channel_qs[idx % 10].put_nowait((idx, ch_block.tobytes()))

                # check for cancellation
                if cancel.is_set():
                    raise Exception("Task cancelled")

                # update progress after every 5% of the source file has been processed
                num_samples_read += n_samples_to_read
                if int(num_samples_read * 100 / n_samples) >= next_update_pct:
                    progress_q.put_nowait((next_update_pct, ""))
                    next_update_pct += 5
    except Exception as e:
        progress_q.put_nowait((next_update_pct, str(e)))
    finally:
        # put end-of-data marker on all channel queues
        for q in channel_qs:
            q.put_nowait((-1, None))


def mp_cache_interleaved_analog_source(dir_path: str, start: int, count: int, progress_q: Queue, cancel: Event) -> str:
    """
    Filter and cache N unfiltered analog channel streams from the specified XSort working directory's binary analog
    source file containing interleaved channels.

    Reads the data in one-second blocks, disentangles the streams into N separate blocks containing the one-second
    trace for each channel [K .. K+N-1], filters the blocks, then appends each block to the appropriate cache file.

    This "task function" is intended to be run in a separate background process. Progress updates are delivered via the
    progress-safe queue provided.

    :param dir_path: Full file system path to the XSort working directory.
    :param start: Index of first channel to cache, K.
    :param count: The number of channels to cache, starting at K.
    :param progress_q: A process-safe queue for delivering progress updates. Each update is in the form of a 3-tuple
        (start, pct, emsg), where start is the index of the first channel cached (int), pct is the percent complete
        (int), and emsg is an error description if the task has failed (otherwise an empty string).
    :param cancel: A process-safe event object which is set to request premature cancellation of this task function.
    :return: An error message if task failed, else an empty string.
    """
    t_start = time.perf_counter()
    t_start_proc = time.process_time()
    emsg, work_dir = WorkingDirectory.load_working_directory(Path(dir_path))
    if work_dir is None:
        return emsg

    n_ch = work_dir.num_analog_channels()
    samples_per_sec = work_dir.analog_sampling_rate

    # the channel cache files to be written
    if start >= n_ch:
        return "Invalid channel start index."
    elif start + count > n_ch:
        count = n_ch - start
    cache_files: List[Path] = list()
    for i in range(count):
        cache_files.append(Path(work_dir.path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(start + i)}"))

    # prepare bandpass filter in case analog signal is wide-band. The filter delays are initialized with zero-vector
    # initial condition and the delays are updated as each block is filtered. SO WE NEED TO MAINTAIN A SEPARATE FILTER
    # DELAY FOR EACH CHANNEL.
    [b, a] = scipy.signal.butter(2, [2 * 300 / samples_per_sec, 2 * 8000 / samples_per_sec], btype='bandpass')
    filter_ic = scipy.signal.lfiltic(b, a, np.zeros(max(len(b), len(a)) - 1))
    delays: List[np.ndarray] = [filter_ic.copy() for _ in range(count)]

    next_update_pct = 5
    emsg = ""
    t_read, t_proc, t_wrt = 0, 0, 0
    try:
        file_size = work_dir.analog_source.stat().st_size
        if file_size % (2 * n_ch) != 0:
            raise Exception(f'Flat binary file size inconsistent with specified number of channels ({n_ch})')
        n_samples = int(file_size / (2 * n_ch))
        n_bytes_per_sample = n_ch * 2

        with open(work_dir.analog_source, 'rb') as src:
            num_samples_read = 0
            while num_samples_read < n_samples:
                # read in one-second at a time -- this is a BIG chunk
                t0 = time.perf_counter()
                n_samples_to_read = min(samples_per_sec, n_samples - num_samples_read)
                curr_block = np.frombuffer(src.read(n_samples_to_read * n_bytes_per_sample), dtype='<h')

                curr_block = curr_block.reshape(-1, n_ch)
                curr_block = curr_block.transpose()
                t_read += time.perf_counter() - t0

                # disentangle the channel streams, filter, and deliver to the channel queues
                for idx in range(start, start + count):
                    t0 = time.perf_counter()
                    ch_block = curr_block[idx].copy()
                    ch_block, delays[idx-start] = scipy.signal.lfilter(b, a, ch_block, axis=-1,
                                                                       zi=delays[idx-start])
                    ch_block = ch_block.astype(np.int16)
                    t1 = time.perf_counter()
                    with open(cache_files[idx-start], 'ab') as f:
                        f.write(ch_block.tobytes())
                    t2 = time.perf_counter()
                    t_proc += t1 - t0
                    t_wrt += t2 - t1

                # check for cancellation
                if cancel.is_set():
                    raise Exception("Task cancelled")

                # update progress after every 5% of the source file has been processed
                num_samples_read += n_samples_to_read
                if int(num_samples_read * 100 / n_samples) >= next_update_pct:
                    progress_q.put_nowait((start, next_update_pct, ""))
                    next_update_pct += 5

    except Exception as e:
        emsg = str(e)
        progress_q.put_nowait((start, next_update_pct, emsg))

    t_total = time.perf_counter() - t_start
    t_total_proc = time.process_time() - t_start_proc
    t_est = t_read + t_proc + t_wrt
    print(f"DEBUG: start_index={start}: t_total_proc < t_total > read+proc+wrt: {t_total_proc:.3f} < "
          f"{t_total:.3f} > {t_est:.3f} = {t_read:.3f} + {t_proc:.3f} + {t_wrt:.3f}", flush=True)

    return emsg


def mp2_cache_interleaved_analog_source(dir_path: str, block_q: Queue, progress_q: Queue, cancel: Event) -> str:
    """
    Digest the specified XSort working directory's binary analog source file containg unfiltered interleaved channels,
    delivering the de-interleaved streams as a sequence of blocks to a process-safe queue.

    Reads the data in one-second blocks, disentangles the streams into N separate blocks containing the one-second
    trace for each channel [0 .. N-1], filters the blocks, then pushes each block onto the queue as a tuple
    (ch_idx, buf), where ch_idx is the channel index and buf is the byte buffer containing the next block for that
    channel, ready to be written to file.

    This "task function" is intended to be run in a separate background process. Progress updates are delivered via the
    progress-safe queue provided. It is assumed a pool of writer threads have been "spun up" to service the block queue.

    :param dir_path: Full file system path to the XSort working directory.
    :param block_q: A process-safe queue by which this task delivers the de-interleaved and filtered analog channel
        streams in a long sequence of tuples (ch_idx, buf), where ch_idx is the analog channel index and buf is the
        byte buffer holding the next block of bytes to be written to the corresponding analog channel cache file.
    :param progress_q: A process-safe queue for delivering progress updates. Each update is in the form of a 2-tuple
        (pct, emsg), pct is the percent complete (int), and emsg is an error description if the task has failed
        (otherwise an empty string).
    :param cancel: A process-safe event object which is set to request premature cancellation of this task function.
    :return: An error message if task failed, else an empty string.
    """
    emsg, work_dir = WorkingDirectory.load_working_directory(Path(dir_path))
    if work_dir is None:
        return emsg

    n_ch = work_dir.num_analog_channels()
    samples_per_sec = work_dir.analog_sampling_rate

    # prepare bandpass filter in case analog signal is wide-band. The filter delays are initialized with zero-vector
    # initial condition and the delays are updated as each block is filtered. SO WE NEED TO MAINTAIN A SEPARATE FILTER
    # DELAY FOR EACH CHANNEL.
    [b, a] = scipy.signal.butter(2, [2 * 300 / samples_per_sec, 2 * 8000 / samples_per_sec], btype='bandpass')
    filter_ic = scipy.signal.lfiltic(b, a, np.zeros(max(len(b), len(a)) - 1))
    delays: List[np.ndarray] = [filter_ic.copy() for _ in range(n_ch)]

    emsg = ""
    try:
        file_size = work_dir.analog_source.stat().st_size
        if file_size % (2 * n_ch) != 0:
            raise Exception(f'Flat binary file size inconsistent with specified number of channels ({n_ch})')
        n_samples = int(file_size / (2 * n_ch))
        n_bytes_per_sample = n_ch * 2

        with open(work_dir.analog_source, 'rb') as src:
            num_samples_read = 0
            while num_samples_read < n_samples:
                # read in one-second at a time -- this is a BIG chunk
                n_samples_to_read = min(samples_per_sec, n_samples - num_samples_read)
                curr_block = np.frombuffer(src.read(n_samples_to_read * n_bytes_per_sample), dtype='<h')

                curr_block = curr_block.reshape(-1, n_ch)
                curr_block = curr_block.transpose()

                # disentangle the channel streams, filter, and deliver to the channel queues
                for idx in range(n_ch):
                    ch_block = curr_block[idx].copy()
                    ch_block, delays[idx] = scipy.signal.lfilter(b, a, ch_block, axis=-1, zi=delays[idx])
                    ch_block = ch_block.astype(np.int16)
                    block_q.put_nowait((idx, ch_block.tobytes()))

                # check for cancellation
                if cancel.is_set():
                    raise Exception("Task cancelled")

                # update progress after every 5% of the source file has been processed
                num_samples_read += n_samples_to_read
                progress_q.put_nowait((int(num_samples_read * 100 / n_samples), ""))

    except Exception as e:
        emsg = str(e)
        progress_q.put_nowait((-1, emsg))
    finally:
        # put end-of-data_marker on block queue so that any threads monitoring that queue will stop
        block_q.put_nowait((-1, None))

    return emsg


def service_analog_channel_blocks_queue(work_dir: WorkingDirectory, which: int,
                                        q: Queue, progress_q: Queue, cancel: Event) -> None:
    """
    Task function for a "writer thread" that services the block queue on which mp2_cache_interleaved_analog_source()
    delivers blocks of analog channel data. The idea is to have multiple threads servicing the queue, so that channel
    blocks do not build up too deeply in that queue.

    :param work_dir: The XSort working directory in which the channel cache files are written.
    :param which: Queue/thread index -- included in error description if task fails on an error.
    :param q: A process-safe queue on which channel data blocks are streamed by the reader task (in separate process).
    :param progress_q: A process-safe queue for delivering progress updates. This method only posts to the queue if
        the task fails on an error: the update is a 2-tuple (-1, emsg), where emsg is the error description.
    :param cancel: A process-safe event object which is set to request premature cancellation of this task function. Any
        cache files written by this task are NOT removed in the event of an error or cancellation.
    """
    try:
        while True:
            try:
                ch_idx, buf = q.get(timeout=0.2)
                # end-of-data-marker: put it back on queue and stop
                if ch_idx < 0:
                    q.put_nowait((-1, None))
                    break
                cache_file = Path(work_dir.path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(ch_idx)}")
                with open(cache_file, 'ab') as f:
                    f.write(buf)
            except Empty:
                pass
            if cancel.is_set():
                raise Exception("Task cancelled")

    except Exception as e:
        if not cancel.is_set():
            progress_q.put_nowait((-1, f"Writer task (index {which}) failed: {str(e)}"))


def cache_analog_channels_from_queue(work_dir: WorkingDirectory, which: int, q: Queue, progress_q: Queue,
                                     cancel: Event) -> None:
    """
    Task function for one of the "writer threads" that service the channel queues on which
    :method:`read_and_filer_interleaved_analog_source()` delivers blocks of channel data.

    This method continuously checks its queue for a block of channel data, in the form of a tuple (ch_idx, buf), where
    ch_idx is the analog channel index and buf is a byte buffer holding the next block of bytes to append to the
    internal cache file for that channel.

    Since there's generally a system limit on the number of open file descriptors, this method opens the relevant file,
    appends, the channel block, and closes the file before continuing. When it pops the end-of-data marker (-1, None),
    off the its channel queue, it cleans up and exits.

    NOTE: I tested a version of this which kept a cache file open rather than closing it after every write. The test
    data was a NeuroPixel session with 385 interleaved analog channels (so 385 separate files open at once during the
    caching task). This version ran much slower, even though the system limit for number of open file descriptors was
    10240 on my machine.

    :param work_dir: The XSort working directory in which the channel cache files are written.
    :param which: Queue/thread index -- included in error description if task fails on an error. This index N indicates
        which channels are streamed on this task's channel queue: ch_idx % 10 == N.
    :param q: A thread-safe queue on which channel data blocks are streamed by the "reader thread".
    :param progress_q: A thread-safe queue for delivering progress updates. This method only posts to the queue if
        the task fails on an error: the update is a 2-tuple (-1, emsg), where emsg is the error description.
    :param cancel: A thread-safe event object which is set to request premature cancellation of this task function. Any
        cache files written by this task are removed in the event of an error or cancellation.
    """
    cache_files_written: Dict[int, Path] = dict()
    ok = True
    try:
        while True:
            ch_idx, buf = q.get()
            if ch_idx < 0:
                break
            cache_file = cache_files_written.get(ch_idx, None)
            if cache_file is None:
                cache_file = Path(work_dir.path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(ch_idx)}")
                cache_files_written[ch_idx] = cache_file
            with open(cache_file, 'ab') as f:
                f.write(buf)
            if cancel.is_set():
                raise Exception("Task cancelled")

    except Exception as e:
        if not cancel.is_set():
            progress_q.put_nowait((-1, f"Writer task (index {which}) failed: {str(e)}"))
        ok = False
    finally:
        # delete any cache files written if an error occurred or task cancelled.
        if not ok:
            for _, p in cache_files_written:
                p.unlink(missing_ok=True)


def delete_internal_cache_files(work_dir: WorkingDirectory) -> None:
    """
    Remove all internal analog data and unit metrics cache files from an XSort working directory.

    :param work_dir: The working directory.
    """
    for p in work_dir.path.iterdir():
        if p.name.startswith(CHANNEL_CACHE_FILE_PREFIX) or p.name.startswith(UNIT_CACHE_FILE_PREFIX):
            p.unlink(missing_ok=True)
