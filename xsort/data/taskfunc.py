import random
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from queue import Queue, Empty
from threading import Event
from typing import Dict, List, Tuple, Union, Optional

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

N_TEMPLATE_CLIPS: int = 10000   # -1 --> use all spike clips
"""
Number of randomly selected spike clips to calculate the mean spike waveform for each unit for inclusion in the unit
metrics cache file. For units with fewer than this many spikes, all spike clips are used. If set to -1, then all
spike clips are used regardless.
"""
N_TEMPLATE_CLIPS_MIN: int = 100
""" 
Number of randomly selected spike clips used to determine primary channel (via mean spike waveform and SNR computation)
for each unit when the total number of analog channels exceeds 16. When there are hundreds of channels, it takes too
long to compute templates/SNR for every unit on every channel using a large number of clips.
"""
N_MAX_TEMPLATES_PER_UNIT: int = 16
""" Maximum number of per-channel spike templates computed and cached in neural unit metrics file. """
_READ_CHUNK_SIZE: int = 2400 * 1024
""" Read chunk size when processing individual channel cache file or non-interleaved analog source file. """


def retrieve_trace_from_channel_cache_file(work_dir: WorkingDirectory, ch_idx: int, start: int, count: int) \
        -> Union[ChannelTraceSegment, str]:
    """
    Retrieve a trace segment from an analog data channel cache file within the XSort working directory. This method
    expects the cache file to exist and fails otherwise. It is not cancellable.

    :param work_dir: The XSort working directory.
    :param ch_idx: The index of the requested analog data channel.
    :param start: Index of the first sample in the trace (relative to recording start at index 0).
    :param count: The trace length in # of samples.
    :return: The requested trace segment. On failure, returns a brief error description (str).
    """
    try:
        trace = work_dir.retrieve_cached_channel_trace(ch_idx, start, count)
        return trace
    except Exception as e:
        return f"Error retrieving trace for channel {ch_idx}: {str(e)}"


def cache_neural_units_all_channels(
        dir_path: str, task_id: int, uids: List[str], progress_q: Queue, cancel: Event) -> None:
    """
    Calculate per-channel mean spike waveforms for a set of neural units across all available analog channels and save
    unit metrics to dedicated internal cache files within the XSort working directory.

    Use case: Recording sessions in which the total number of analog data channels is <=16. Analog data stored in
    individual channel cache files or in a prefiltered flat binary source file.

    This task function is intended to run in a separate process. The task manager should split up the entire set of
    units into "banks", each of which is assigned to a different process.

    :param dir_path: File system path name for the XSort working directory.
    :param task_id: An integer identifying this task -- included in progress updates.
    :param uids: UIDs identifying which units to process.
    :param progress_q: A process-safe queue for delivering progress updates back to the task manager. Each "update" is
        in the form of a 3-tuple (id, pct, emsg), where id is the argument task_id (int), pct is the percent complete
        (int), and emsg is an error description if the task has failed (otherwise an empty string). **The task function
        finishes immediately after delivering an error message to the queue.**
    :param cancel: A process-safe event object which is set to request premature cancellation of this task function.
    """
    try:
        emsg, work_dir = WorkingDirectory.load_working_directory(Path(dir_path))
        if work_dir is None:
            raise Exception(emsg)
        if work_dir.num_analog_channels() > N_MAX_TEMPLATES_PER_UNIT:
            raise Exception(f"Not supported for recording session with >{N_MAX_TEMPLATES_PER_UNIT} analog channels.")
        emsg, neurons = work_dir.load_neural_units()
        if len(emsg) > 0:
            raise Exception(emsg)

        if cancel.is_set():
            raise Exception("Task canceled")

        # pare down the unit list to only those we care about
        neurons = [u for u in neurons if u.uid in uids]

        # spawn a thread per channel to calculate unit templates
        with ThreadPoolExecutor(max_workers=work_dir.num_analog_channels()) as mt_exec:
            progress_per_ch: Dict[int, int] = {idx: 0 for idx in work_dir.analog_channel_indices}
            next_progress_update_pct = 0

            thrd_q = Queue()
            futures: List[Future] = \
                [mt_exec.submit(_compute_unit_templates_on_channel, work_dir, idx, neurons, thrd_q, cancel,
                                N_TEMPLATE_CLIPS)
                 for idx in work_dir.analog_channel_indices]

            first_error: Optional[str] = None
            while 1:
                try:
                    # progress updates are: (ch_idx, pct_complete, error_msg)
                    update = thrd_q.get(timeout=0.2)
                    if len(update[2]) > 0:
                        # an error has occurred. If not bc task was cancelled, signal cancel now so that remaining
                        # threads stop. And, just in case, cancel any threads that have not started.
                        if not cancel.is_set():
                            cancel.set()
                            first_error = f"Error computing unit templates on channel {update[0]}: {update[2]}"
                        for future in futures:
                            future.cancel()
                    else:
                        progress_per_ch[update[0]] = update[1]
                        total_progress = sum(progress_per_ch.values()) / len(progress_per_ch)
                        if total_progress >= next_progress_update_pct:
                            progress_q.put_nowait((task_id, next_progress_update_pct, ""))
                            next_progress_update_pct += 5
                except Empty:
                    pass

                if all([future.done() for future in futures]):
                    break

            # since threads deliver error message via queue, empty the queue in case any error message wasn't rcvd yet
            if first_error is None:
                while not thrd_q.empty():
                    update = thrd_q.get_nowait()
                    if len(update[2]) > 0:
                        first_error = f"Error computing unit templates on channel {update[0]}: {update[2]}"
                        break

            if not (first_error is None):
                progress_q.put_nowait((task_id, 0, first_error))
                return

            # colllate results
            progress_q.put_nowait((task_id, 95, ""))
            templates_for_unit: Dict[str, Dict[int, np.ndarray]] = {uid: dict() for uid in uids}
            best_snr_for_unit: Dict[str, float] = {uid: 0.0 for uid in uids}
            primary_ch_for_unit: Dict[str, int] = {uid: -1 for uid in uids}
            for future in futures:
                ch_idx, noise, template_dict = future.result()
                for uid in uids:
                    template = template_dict[uid]
                    snr = ((np.max(template) - np.min(template)) / (1.96 * noise)) if noise > 0 else 0.0
                    if snr > best_snr_for_unit[uid]:
                        best_snr_for_unit[uid] = snr
                        primary_ch_for_unit[uid] = ch_idx
                    templates_for_unit[uid][ch_idx] = template * work_dir.analog_channel_sample_to_uv(ch_idx)

            # write unit metric cache files
            for uid in uids:
                unit = next((u for u in neurons if u.uid == uid))
                unit.update_metrics(primary_ch_for_unit[uid], best_snr_for_unit[uid], templates_for_unit[uid])
                if not work_dir.save_neural_unit_to_cache(unit):
                    progress_q.put_nowait((task_id, 0,
                                           f"Error occurred while writing unit metrics to internal cache: uid={uid}"))
                    return
            progress_q.put_nowait((task_id, 100, ""))
    except Exception as e:
        progress_q.put_nowait((task_id, 0, f"Error caching unit metrics (task_id={task_id}): {str(e)}"))


def _compute_unit_templates_on_channel(
        work_dir: WorkingDirectory, ch_idx: int, units: List[Neuron], progress_q: Queue,
        cancel: Event, n_max_spks: int) \
        -> Optional[Tuple[int, float, Dict[str, np.ndarray]]]:
    """
    Helper task for cache_neural_units_all_channels() calculates neural unit mean spike waveforms on a single analog
    channel. Intended to be performed on a separate thread.

    :param work_dir: The XSort working directory.
    :param ch_idx: Index of the analog channel to process.
    :param units: The list of neural units for which metrics are to be computed.
    :param progress_q: A thread-safe queue for delivering progress updates. Each "update" is in the form of a
        3-tuple (idx, pct, emsg), where idx is the analog channel index (int), pct is the percent complete (int), and
        emsg is an error description if the task has failed (otherwise an empty string).
    :param cancel: A thread-safe event object which is set to request premature cancellation of this task function.
    :param n_max_spks: The maximum number of spike waveform clips to process when computing the spike templates.
        If <= 0, then the entire spike train is included in the calculation.
    :return: A 3-tuple [ch_idx, noise, template_dict], where: ch_idx is the index of the analog channel processed;
        noise is the calculated noise level on the channel (needed to compute SNR for each unit); and template_dict is a
        dictionary, keyed by unit UID, holding the computed spike templates. **Both noise level and template samples
        are in raw ADC units (NOT converted to microvolts).** If task fails, returns None.
    """
    try:
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

        n_ch = work_dir.num_analog_channels()
        interleaved = work_dir.is_analog_data_interleaved
        n_bytes_per_sample = 2 * n_ch if interleaved else 2
        if ch_file.stat().st_size % n_bytes_per_sample != 0:
            raise Exception(f"Bad file size for analog cache or source file: {ch_file.name}")
        num_samples_recorded = int(ch_file.stat().st_size / n_bytes_per_sample)
        samples_per_sec = work_dir.analog_sampling_rate
        template_len = int(samples_per_sec * 0.01)

        # the template array and the number of clips actually accumulated for each unit (so we can compute average)
        n_units = len(units)
        unit_num_clips: List[int] = [0] * n_units
        unit_templates: List[np.ndarray] = [np.zeros(template_len, dtype='<f') for _ in range(n_units)]

        # compute start indices of ALL 10-ms waveform clips to be culled from analog channel stream, across ALL units.
        # Each  clip start index is tupled with the corresponding unit index: (clip_start, unit_idx), ...
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

        # strip off clips that are cut off at beginning or end of recording
        while (len(clip_starts) > 0) and (clip_starts[0][0] < 0):
            clip_starts.pop(0)
        while (len(clip_starts) > 0) and (clip_starts[-1][0] + template_len >= num_samples_recorded):
            clip_starts.pop()
        if len(clip_starts) == 0:
            raise Exception("No spike clips to process!")

        # desired block read size in bytes: one template length!
        n_bytes_per_clip = template_len * n_bytes_per_sample

        def check_progress(next_upd: int) -> int:
            """
             Check for task cancellation and report progress periodically.
            :param next_upd: The percent complete at which next progress update should be posted to queue.
            :return: The percent complete at which next progress update should be posted to queu
            """
            if cancel.is_set():
                raise Exception("Task cancelled")
            _p = int(clip_idx * 100 / len(clip_starts))
            if _p >= next_upd:
                progress_q.put_nowait((ch_idx, _p, ""))
                next_upd += 5
            return next_upd

        def process_next_chunk(file_ofs: int, i_clip: int) -> int:
            """
            Process the next read chunk of clips from the analog channel stream in a non-interleaved source.
            :param file_ofs: File offset to the start of the stream. Will be 0 if the source file is an individual
                channel cache file.
            :param i_clip: The current clip index.
            :return: The number of clips processed
            """
            # seek to start of next clip to be processed
            chunk_s = clip_starts[i_clip][0]
            src.seek(file_ofs + chunk_s * n_bytes_per_sample)

            # read a chunk of size M <= _READ_CHUNK_SIZE that fully contains 1 or more clips. The goal here is to
            # reduce the total number of reads required to process file. This significantly improved performance.
            n = 1
            while ((i_clip + n < len(clip_starts)) and
                   ((clip_starts[i_clip + n - 1][0] - chunk_s) * n_bytes_per_sample +
                    n_bytes_per_clip < _READ_CHUNK_SIZE)):
                n += 1
            chunk_sz = (clip_starts[i_clip + n - 1][0] - chunk_s) * n_bytes_per_sample + n_bytes_per_clip
            chunk = np.frombuffer(src.read(chunk_sz), dtype='<h')

            # process all clips in the chunk
            for _k in range(n):
                _s = clip_starts[i_clip + _k][0] - chunk_s
                clip = chunk[_s:_s + template_len]
                block_medians.append(np.median(np.abs(clip)))

                # accumulate extracted spike clip for the relevant unit
                i_unit = clip_starts[i_clip + _k][1]
                _template = unit_templates[i_unit]
                np.add(_template, clip, out=_template)
                unit_num_clips[i_unit] += 1
            return n

        # extract and accumulate all spike waveform clips, across all units, from the source file. If we're dealing with
        # a flat binary file with interleaving, one sample is actually one scan of the N channels recorded!
        block_medians: List[float] = list()  # for computing noise level on channel
        with open(ch_file, 'rb') as src:
            clip_idx = 0
            # for the prefiltered non-interleaved flat binary source, we need the offset to the start of the
            # contiguous block for the target channel
            ofs_to_ch = 0 if (work_dir.need_analog_cache or interleaved) else \
                (ch_idx * num_samples_recorded * n_bytes_per_sample)

            next_update_pct = 0
            while clip_idx < len(clip_starts):
                if interleaved:
                    # seek to start of next clip to be processed, then read in THAT CLIP ONLY
                    sample_idx = clip_starts[clip_idx][0]
                    src.seek(ofs_to_ch + sample_idx * n_bytes_per_sample)
                    curr_clip = np.frombuffer(src.read(n_bytes_per_clip), dtype='<h')

                    # for interleaved source: deinterleave channels and extract clip for target channel
                    if interleaved:
                        curr_clip = curr_clip.reshape(-1, n_ch).transpose()
                        curr_clip = curr_clip[ch_idx]

                    block_medians.append(np.median(np.abs(curr_clip)))  # for later noise calc

                    # accumulate extracted spike clip for the relevant unit
                    unit_idx = clip_starts[clip_idx][1]
                    template = unit_templates[unit_idx]
                    np.add(template, curr_clip, out=template)
                    unit_num_clips[unit_idx] += 1
                    clip_idx += 1
                else:
                    n_clips = process_next_chunk(ofs_to_ch, clip_idx)
                    clip_idx += n_clips

                next_update_pct = check_progress(next_update_pct)

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

    except Exception as e:
        progress_q.put_nowait((ch_idx, 0, str(e)))
        return None


def cache_neural_units_select_channels(
        dir_path: str, task_id: int, unit_to_primary: Dict[str, int], progress_q: Queue, cancel: Event) -> None:
    """
    Calculate per-channel mean spike waveforms for a set of neural units -- for a small contiguous block of analog
    data channels near each unit's "primary channel" -- and save unit metrics to dedicated internal cache files within
    the XSort working directory.

    Use case: Recording sessions in which total number of analog data channels K>N, where N = N_MAX_TEMPLATES_PER_UNIT
    (16). Analog data stored in individual channel cache files or in a prefiltered flat binary source file. The
    "primary channel" for each unit -- the analog data channel P exhibiting the best SNR for that unit -- must be
    identified for all units prior to running this task. The method will compute mean spike waveforms -- aka templates
    -- for each unit only on a small set of channels N "near" the primary channel P. Since XSort does not yet support
    the notion of "probe geometry", we currently look at the numerically continguous block of channel indices
    [P-N/2 .. P-N/2 + N-1]. If P < N/2, then [0 .. N-1] is used; if P > K-N/2, then [K-N, K-1] is used.

    NOTE: This method could be used when the total number of analog channels K<=N, but it is less efficient than
    cache_neural_units_all_channels(), which takes advantage of the fact that the set of channels to be processed is
    the same for ALL units. That is not the case here.

    This task function is intended to run in a separate process. The task manager should split up the entire set of
    units into "banks", each of which is assigned to a different process.

    :param dir_path: File system path name for the XSort working directory.
    :param task_id: An integer identifying this task -- included in progress updates.
    :param unit_to_primary: Maps UID of a unit to be cached to the unit's identified primary channel index P. This
        determines which analog channels are analyzed to compute per-channel spike templates for the unit, as described.
    :param progress_q: A process-safe queue for delivering progress updates back to the task manager. Each "update" is
        in the form of a 3-tuple (id, pct, emsg), where id is the argument task_id (int), pct is the percent complete
        (int), and emsg is an error description if the task has failed (otherwise an empty string). **The task function
        finishes immediately after delivering an error message to the queue.**
    :param cancel: A process-safe event object which is set to request premature cancellation of this task function.
    """
    try:
        emsg, work_dir = WorkingDirectory.load_working_directory(Path(dir_path))
        if work_dir is None:
            raise Exception(emsg)
        emsg, neurons = work_dir.load_neural_units()
        if len(emsg) > 0:
            raise Exception(emsg)

        if cancel.is_set():
            raise Exception("Task canceled")

        # pare down the unit list to only those we care about
        neurons = [u for u in neurons if u.uid in unit_to_primary]

        # spawn a thread per unit to calculate unit templates and cache metrics to file
        with ThreadPoolExecutor(max_workers=16) as mt_exec:
            progress_per_unit: Dict[str, int] = {u.uid: 0 for u in neurons}
            next_progress_update_pct = 0

            thrd_q = Queue()
            futures: List[Future] = \
                [mt_exec.submit(_compute_templates_and_cache_metrics_for_unit, work_dir, u, unit_to_primary[u.uid],
                                thrd_q, cancel, N_TEMPLATE_CLIPS) for u in neurons]

            first_error: Optional[str] = None
            while 1:
                try:
                    # progress updates are: (uid, pct_complete, error_msg)
                    update = thrd_q.get(timeout=0.2)
                    if len(update[2]) > 0:
                        # an error has occurred. If error is not bc task was cancelled, signal cancel now so that
                        # remaining running threads abort. Also cancel any unstarted threads.
                        if not cancel.is_set():
                            cancel.set()
                            first_error = f"Error caching metrics for unit {update[0]}: {update[2]}"
                        for future in futures:
                            future.cancel()
                    else:
                        progress_per_unit[update[0]] = update[1]
                        total_progress = sum(progress_per_unit.values()) / len(progress_per_unit)
                        if total_progress >= next_progress_update_pct:
                            progress_q.put_nowait((task_id, next_progress_update_pct, ""))
                            next_progress_update_pct += 5
                except Empty:
                    pass

                if all([future.done() for future in futures]):
                    break

            # since threads deliver error message via queue, empty the queue in case any error message wasn't rcvd yet
            if first_error is None:
                while not thrd_q.empty():
                    update = thrd_q.get_nowait()
                    if len(update[2]) > 0:
                        first_error = f"Error caching metrics for unit {update[0]}: {update[2]}"
                        break

            if first_error is None:
                progress_q.put_nowait((task_id, 100, ""))
            else:
                progress_q.put_nowait((task_id, 0, first_error))

    except Exception as e:
        progress_q.put_nowait((task_id, 0, f"Error caching unit metrics (task_id={task_id}): {str(e)}"))


def _compute_templates_and_cache_metrics_for_unit(work_dir: WorkingDirectory, unit: Neuron, primary_ch: int,
                                                  progress_q: Queue, cancel: Event, n_max_spks: int) -> None:
    """
    Helper task for cache_neural_units_select_channels() calculates the mean spike waveforms (templates) for the
    specified neural unit on a contiguous bank of analog channels near the unit's specified "primary channel", then
    writes the unit metrics to an internal cache file in the XSort working directory. Intended to be performed on a
    separate thread.

    NOTE: The primary channel is "identified" by computing templates and SNR for a small random sampling of clips
    across ALL available channels, then choosing the channel on which the unit's SNR was greatest. To compute the
    cached templates, many more clips are averaged, so the identify of the primary channel could change -- although it
    is assumed to be "in the neighborhood" of the originally identified primary channel.

    :param work_dir: The XSort working directory.
    :param unit: The neural unit for which metrics are to be computed.
    :param primary_ch: Index of the analog data channel designated as the primary channel for the neural unit. Given
        N = N_MAX_TEMPLATES_PER_UNIT and K = total # of analog channels, the method compute spike templates on
        channels [P-N/2 .. P-N/2 + N-1]. If P < N/2, then [0 .. N-1] is used; if P > K-N/2, then [K-N, K-1] is used.
    :param progress_q: A thread-safe queue for delivering progress updates. Each "update" is in the form of a
        3-tuple (uid, pct, emsg), where uid is the unit's UID, pct is the percent complete (int), and emsg is an error
        description if the task has failed (otherwise an empty string).
    :param cancel: A thread-safe event object which is set to request premature cancellation of this task function.
    :param n_max_spks: The maximum number of spike waveform clips to process when computing the spike templates.
        If <= 0, then the entire spike train is included in the calculation.
    :return: A 4-tuple [uid, snr, primary_ch, template_dict], where: uid is the neural unit's UID; snr is the best
        SNR across the set of analog channels analyzed, primary_ch is the index of the analog channel exhibiting the
        best SNR (**this could be different than the initially specified primary channel!!**); and template_dict is a
        dictionary, keyed by channel index, holding the unit's computed spike template for that channel. **Template
        samples are converted to microvolts.** If task fails, returns None.
    """
    try:
        # the list of channel indices on which we'll compute unit templates
        n_ch, half_w = work_dir.num_analog_channels(), int(N_MAX_TEMPLATES_PER_UNIT/2)
        first_ch = 0 if primary_ch < half_w else (n_ch-half_w if (primary_ch > n_ch-half_w) else primary_ch-half_w)
        ch_indices = [i + first_ch for i in range(N_MAX_TEMPLATES_PER_UNIT)]

        # we either read directly from the original, prefiltered flat binary source or a set of internal cache files
        interleaved = False
        n_bytes_per_sample = 2
        num_samples_recorded = 0
        if work_dir.need_analog_cache:
            for i in ch_indices:
                ch_file = Path(work_dir.path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(i)}")
                if not ch_file.is_file():
                    raise Exception(f"Missing internal analog cache file {ch_file.name}")
                elif ch_file.stat().st_size % n_bytes_per_sample != 0:
                    raise Exception(f"Bad file size for internal analog cache file {ch_file.name}")
                elif num_samples_recorded == 0:
                    num_samples_recorded = int(ch_file.stat().st_size / n_bytes_per_sample)
                elif num_samples_recorded != int(ch_file.stat().st_size / n_bytes_per_sample):
                    raise Exception(f"All internal analog cache files must be the same size!")
        else:
            interleaved = work_dir.is_analog_data_interleaved
            n_bytes_per_sample = 2 * n_ch if interleaved else 2
            if not work_dir.analog_source.is_file():
                raise Exception(f"Original flat binary analog source is missing in working directory.")
            if work_dir.analog_source.stat().st_size % n_bytes_per_sample != 0:
                raise Exception(f"Bad file size for analog source file: {work_dir.analog_source.name}")
            num_samples_recorded = int(work_dir.analog_source.stat().st_size / n_bytes_per_sample)

        samples_per_sec = work_dir.analog_sampling_rate
        template_len = int(samples_per_sec * 0.01)

        # for accumulating clips and noise level medians on each channel
        template_dict: Dict[int, np.ndarray] = {i: np.zeros(template_len, dtype='<f') for i in ch_indices}
        clip_medians: Dict[int, List[float]] = {i: list() for i in ch_indices}

        # compute start indices of ALL 10-ms waveform clips to be culled from analog channels
        clip_starts: List[int]
        n_spks = min(unit.num_spikes, n_max_spks) if n_max_spks > 0 else unit.num_spikes
        if n_spks < unit.num_spikes:
            spike_indices = sorted(random.sample(range(unit.num_spikes), n_spks))
            clip_starts = [int((unit.spike_times[i] - 0.001) * samples_per_sec) for i in spike_indices]
        else:
            clip_starts = [int((t - 0.001) * samples_per_sec) for t in unit.spike_times]

        # clips must be in chronological order!
        clip_starts.sort()

        # strip off clips that are cut off at beginning or end of recording
        while (len(clip_starts) > 0) and (clip_starts[0] < 0):
            clip_starts.pop(0)
        while (len(clip_starts) > 0) and (clip_starts[-1] + template_len >= num_samples_recorded):
            clip_starts.pop()
        if len(clip_starts) == 0:
            raise Exception("No spike clips to process!")

        # desired block read size in bytes: one template length!
        n_bytes_per_clip = template_len * n_bytes_per_sample

        # for tracking overall progress
        next_update_pct = 0
        total_clips = len(ch_indices) * len(clip_starts)
        total_clips_so_far = 0

        # extract and accumulate all spike waveform clips for the unit from the source file(s).
        if work_dir.need_analog_cache:
            # CASE 1: individual binary cache file for each analog channel stream
            for ch_idx in ch_indices:
                ch_file = Path(work_dir.path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(ch_idx)}")
                per_ch_template = template_dict[ch_idx]
                with open(ch_file, 'rb') as src:
                    clip_idx = 0
                    while clip_idx < len(clip_starts):
                        # seek to start of next clip to be processed, then read in THAT CLIP ONLY
                        sample_idx = clip_starts[clip_idx]
                        src.seek(sample_idx * n_bytes_per_sample)
                        curr_clip = np.frombuffer(src.read(n_bytes_per_clip), dtype='<h')

                        clip_medians[ch_idx].append(np.median(np.abs(curr_clip)))  # for later noise calc

                        # accumulate extracted spike clip for the relevant unit
                        np.add(per_ch_template, curr_clip, out=per_ch_template)
                        clip_idx += 1

                        # get ready for next clip, check for cancellation, and report progress
                        if cancel.is_set():
                            return
                        total_clips_so_far += 1
                        pct_done = int(total_clips_so_far * 100 / total_clips)
                        if pct_done >= next_update_pct:
                            progress_q.put_nowait((unit.uid, pct_done, ""))
                            next_update_pct += 5
        elif not interleaved:
            # CASE 2: Analog channel streams in original non-interleaved flat binary source file
            with open(work_dir.analog_source, 'rb') as src:
                for ch_idx in ch_indices:
                    # offset to start of contiguous block for next channel to process
                    ofs = ch_idx * num_samples_recorded * n_bytes_per_sample
                    clip_idx = 0
                    per_ch_template = template_dict[ch_idx]
                    while clip_idx < len(clip_starts):
                        # seek to start of next clip to be processed, then read in THAT CLIP ONLY
                        sample_idx = clip_starts[clip_idx]
                        src.seek(ofs + sample_idx * n_bytes_per_sample)
                        curr_clip = np.frombuffer(src.read(n_bytes_per_clip), dtype='<h')

                        clip_medians[ch_idx].append(np.median(np.abs(curr_clip)))  # for later noise calc

                        # accumulate extracted spike clip for the relevant unit
                        np.add(per_ch_template, curr_clip, out=per_ch_template)
                        clip_idx += 1

                        # get ready for next clip, check for cancellation, and report progress
                        if cancel.is_set():
                            return
                        total_clips_so_far += 1
                        pct_done = int(total_clips_so_far * 100 / total_clips)
                        if pct_done >= next_update_pct:
                            progress_q.put_nowait((unit.uid, pct_done, ""))
                            next_update_pct += 5
        else:
            # CASE 3: Analog channel streams in original interleaved flat binary source file
            with open(work_dir.analog_source, 'rb') as src:
                clip_idx = 0
                while clip_idx < len(clip_starts):
                    # seek to start of next clip to be processed, then read in THAT CLIP ONLY
                    sample_idx = clip_starts[clip_idx]
                    src.seek(sample_idx * n_bytes_per_sample)
                    curr_clip = np.frombuffer(src.read(n_bytes_per_clip), dtype='<h')

                    # deinterleave channel streams for the current clip
                    curr_clip = curr_clip.reshape(-1, n_ch).transpose()

                    # accumulate clips and clip medians for all the channels we care about
                    for ch_idx in ch_indices:
                        per_ch_clip = curr_clip[ch_idx]
                        clip_medians[ch_idx].append(np.median(np.abs(per_ch_clip)))
                        per_ch_template = template_dict[ch_idx]
                        np.add(per_ch_template, per_ch_clip, out=per_ch_template)
                    clip_idx += 1

                    # get ready for next clip, check for cancellation, and report progress
                    if cancel.is_set():
                        return
                    total_clips_so_far += len(ch_indices)
                    pct_done = int(total_clips_so_far * 100 / total_clips)
                    if pct_done >= next_update_pct:
                        progress_q.put_nowait((unit.uid, pct_done, ""))
                        next_update_pct += 5

        # calculate per-channel templates and SNR, select primary channel (could change from that originally specified).
        best_snr: float = 0
        primary_ch = -1
        for i in ch_indices:
            noise = np.median(clip_medians[i]) * 1.4826
            template = template_dict[i]
            template = template / len(clip_starts)
            snr = (np.max(template) - np.min(template)) / (1.96 * noise)
            if snr > best_snr:
                best_snr = snr
                primary_ch = i
            template *= work_dir.analog_channel_sample_to_uv(i)

        # cache unit metrics to internal file
        unit.update_metrics(primary_ch, best_snr, template_dict)
        if not work_dir.save_neural_unit_to_cache(unit):
            progress_q.put_nowait((unit.uid, 0,
                                   f"Error occurred while writing unit metrics to internal cache: uid={unit.uid}"))
        else:
            progress_q.put_nowait((unit.uid, 100, ""))   # success!
    except Exception as e:
        progress_q.put_nowait((unit.uid, 0, str(e)))


def identify_unit_primary_channels_in_range(
        dir_path: str, start: int, count: int, progress_q: Queue, cancel: Event) -> Dict[str, Tuple[int, float]]:
    """
    Estimate SNR for each unit in the XSort working directory's neural unit source file for each analog channel in the
    range specified, and for each unit return the index of the channel for which the unit's SNR was greatest.

    This task function is intended to run in a separate process. The task manager should split up the entire set of
    analog channels into smaller ranges, each of which is assigned to a different process.

    NOTES:

    - Use case: When a recording session involves hundreds of channels and hundreds of units, it takes too long to
      compute accurate spike templates for every unit on every channel. Instead, taking advantage of the fact that a
      given unit is typically detectable on only a few channels, we do a quick computation of SNR (template peak-to-peak
      amplitude / noise level on channel) across all units and channels **using a random sampling of a relatively small
      number of clips per unit.** The "primary channel" for each unit is then identified as the analog channel on which
      SNR is greatest. Then more accurate templates (averaging many more spike waveform clips) can be computed only for
      a small number of data channels "in the neighborhood" of the primary channel.
    - This method esimates the channel noise level by sampling the median value of each extracted clip rather than the
      the entire recording on that channel. But tests have indicated this still gives a reasonable estimate of unit SNR
      on the channel.
    - The analog data is located in individual channel cache files, or in the original prefiltered flat binary file,
      with the channels streams interleaved or not.
    - Extracting one clip at a time can be problematic -- the total number of file seek-and-read operations would be
      #channels * #units * 100 clips/unit (N_TEMPLATE_CLIPS_MIN). With hundreds of channels and units, that is an
      inefficient approach. When the extracting from a prefiltered, interleaved binary file the approach works because
      we have to read in all the channel clips in one go since the streams are interleaved. With a non-interleaved
      binary file or individual cache files, we need to read in a larger multi-clip chunk (~2400KB).

    :param dir_path: Full file system path to the XSort working directory.
    :param start: Index of first channel in the range, K
    :param count: The number of channels N in the range. If K+N is greater than the number of channels
        recorded, then the task processes all remaining channels starting at K.
    :param progress_q: A process-safe queue for delivering progress updates. Each update is in the form of a 3-tuple
        (start, pct, emsg), where start is the index of the first channel processed (int), pct is the percent complete
        (int), and emsg is an error description if the task has failed (otherwise an empty string).
    :param cancel: A process-safe event object which is set to request premature cancellation of this task function.
    :return: Dictionary mapping each neural unit's UID to a 2-tuple (ch_idx, snr) containing the index of the channel
        within the range specified) that exhibited the highest SNR for that unit, and the SNR value. Returns an empty
        mapping on failure.
    """
    try:
        emsg, work_dir = WorkingDirectory.load_working_directory(Path(dir_path))
        if work_dir is None:
            raise Exception(emsg)
        emsg, units = work_dir.load_neural_units()
        if len(emsg) > 0:
            raise Exception(emsg)
        n_ch = work_dir.num_analog_channels()
        if start < 0 or start >= n_ch:
            raise Exception("Invalid channel range")
        elif start + count > n_ch:
            count = n_ch - start

        if cancel.is_set():
            raise Exception("Task canceled")

        # we either read directly from the original, prefiltered flat binary source or a set of internal cache files
        interleaved = False
        n_bytes_per_sample = 2
        num_samples_recorded = 0
        if work_dir.need_analog_cache:
            for i in range(count):
                ch_file = Path(work_dir.path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(i+start)}")
                if not ch_file.is_file():
                    raise Exception(f"Missing internal analog cache file {ch_file.name}")
                elif ch_file.stat().st_size % n_bytes_per_sample != 0:
                    raise Exception(f"Bad file size for internal analog cache file {ch_file.name}")
                elif num_samples_recorded == 0:
                    num_samples_recorded = int(ch_file.stat().st_size / n_bytes_per_sample)
                elif num_samples_recorded != int(ch_file.stat().st_size / n_bytes_per_sample):
                    raise Exception(f"All internal analog cache files must be the same size!")
        else:
            interleaved = work_dir.is_analog_data_interleaved
            n_bytes_per_sample = 2 * n_ch if interleaved else 2
            if not work_dir.analog_source.is_file():
                raise Exception(f"Original flat binary analog source is missing in working directory.")
            if work_dir.analog_source.stat().st_size % n_bytes_per_sample != 0:
                raise Exception(f"Bad file size for analog source file: {work_dir.analog_source.name}")
            num_samples_recorded = int(work_dir.analog_source.stat().st_size / (2*n_ch))

        samples_per_sec = work_dir.analog_sampling_rate
        template_len = int(samples_per_sec * 0.01)

        # the template array and the number of clips actually accumulated for each unit (so we can compute average)
        n_units = len(units)
        unit_num_clips: List[Dict[int, int]] = list()
        unit_templates: List[Dict[int, np.ndarray]] = list()
        for _ in range(n_units):
            template_dict: Dict[int, np.ndarray] = dict()
            num_clips_dict: Dict[int, int] = dict()
            for i in range(count):
                template_dict[i+start] = np.zeros(template_len, dtype='<f')
                num_clips_dict[i+start] = 0
            unit_templates.append(template_dict)
            unit_num_clips.append(num_clips_dict)

        # compute start indices of ALL 10-ms waveform clips to be culled from analog channel streams, across ALL units.
        # Each clip start index is tupled with the corresponding unit index: (clip_start, unit_idx), ...
        clip_starts: List[Tuple[int, int]] = list()
        for unit_idx, u in enumerate(units):
            n_spks = min(u.num_spikes, N_TEMPLATE_CLIPS_MIN)
            if n_spks < u.num_spikes:
                spike_indices = sorted(random.sample(range(u.num_spikes), n_spks))
                unit_clip_starts = [(int((u.spike_times[i] - 0.001) * samples_per_sec), unit_idx)
                                    for i in spike_indices]
            else:
                unit_clip_starts = [(int((t - 0.001) * samples_per_sec), unit_idx) for t in u.spike_times]
            clip_starts.extend(unit_clip_starts)

        # clips must be in chronological order!
        clip_starts.sort(key=lambda x: x[0])

        # strip off clips that are cut off at beginning or end of recording
        while (len(clip_starts) > 0) and (clip_starts[0][0] < 0):
            clip_starts.pop(0)
        while (len(clip_starts) > 0) and (clip_starts[-1][0] + template_len >= num_samples_recorded):
            clip_starts.pop()
        if len(clip_starts) == 0:
            raise Exception("No spike clips to process!")

        # accumulate clip medians for computing noise level on each channel
        clip_medians: Dict[int, List[float]] = dict()
        for i in range(count):
            clip_medians[i+start] = list()

        # for tracking overall progress
        next_update_pct = 0
        total_clips = count * len(clip_starts)
        total_clips_so_far = 0

        def check_progress(next_upd: int) -> int:
            """
             Check for task cancellation and report progress periodically.
            :param next_upd: The percent complete at which next progress update should be posted to queue.
            :return: The percent complete at which next progress update should be posted to queu
            """
            if cancel.is_set():
                raise Exception("Task cancelled")
            _p = int(total_clips_so_far * 100 / total_clips)
            if _p >= next_upd:
                progress_q.put_nowait((start, _p, ""))
                next_upd += 5
            return next_upd

        def process_next_chunk(file_ofs: int, i_clip: int, i_ch: int) -> int:
            """
            Process the next read chunk of clips in a non-interleaved source.
            :param file_ofs: File offset to the start of the stream for the specified analog data channel. Will be 0
                if the source file is an individual channel cache file.
            :param i_clip: The current clip index.
            :param i_ch: The channel index.
            :return: The number of clips processed
            """
            # seek to start of next clip to be processed
            chunk_s = clip_starts[i_clip][0]
            src.seek(file_ofs + chunk_s * n_bytes_per_sample)

            # read a chunk of size M <= _READ_CHUNK_SIZE that fully contains 1 or more clips. The goal here is to
            # reduce the total number of reads required to process file. This significantly improved performance.
            n = 1
            while ((i_clip + n < len(clip_starts)) and
                   ((clip_starts[i_clip + n - 1][0] - chunk_s) * n_bytes_per_sample +
                    n_bytes_per_clip < _READ_CHUNK_SIZE)):
                n += 1
            chunk_sz = (clip_starts[i_clip + n - 1][0] - chunk_s) * n_bytes_per_sample + n_bytes_per_clip
            chunk = np.frombuffer(src.read(chunk_sz), dtype='<h')

            # process all clips in the chunk
            for _k in range(n):
                _s = clip_starts[i_clip + _k][0] - chunk_s
                clip = chunk[_s:_s + template_len]
                clip_medians[i_ch].append(np.median(np.abs(clip)))

                # accumulate extracted spike clip for the relevant unit
                i_unit = clip_starts[i_clip + _k][1]
                _template = unit_templates[i_unit][i_ch]
                np.add(_template, clip, out=_template)
                unit_num_clips[i_unit][i_ch] += 1
            return n

        # accumulate spike waveform clips across all units, plus clip medians, from the channel cache files or the
        # flat binary source file.
        n_bytes_per_clip = template_len * n_bytes_per_sample
        if work_dir.need_analog_cache:
            # CASE 1: individual binary cache file for each analog channel stream
            for i in range(count):
                ch_idx = i + start
                ch_file = Path(work_dir.path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(ch_idx)}")
                with open(ch_file, 'rb') as src:
                    clip_idx = 0
                    while clip_idx < len(clip_starts):
                        n_clips = process_next_chunk(0, clip_idx, ch_idx)
                        clip_idx += n_clips
                        total_clips_so_far += n_clips
                        next_update_pct = check_progress(next_update_pct)

        elif not interleaved:
            # CASE 2: Analog channel streams in original non-interleaved flat binary source file
            with open(work_dir.analog_source, 'rb') as src:
                for i in range(count):
                    # offset to start of contiguous block for next channel to process
                    ch_idx = i + start
                    ofs = ch_idx * num_samples_recorded * n_bytes_per_sample
                    clip_idx = 0
                    while clip_idx < len(clip_starts):
                        n_clips = process_next_chunk(ofs, clip_idx, ch_idx)
                        clip_idx += n_clips
                        total_clips_so_far += n_clips
                        next_update_pct = check_progress(next_update_pct)

        else:
            # CASE 3: Analog channel streams in original interleaved flat binary source file
            with open(work_dir.analog_source, 'rb') as src:
                clip_idx = 0
                while clip_idx < len(clip_starts):
                    # seek to start of next clip to be processed, then read in THAT CLIP ONLY. Because of interleaving,
                    # if contains the recorded clip on EVERY analog data channel!
                    sample_idx = clip_starts[clip_idx][0]
                    src.seek(sample_idx * n_bytes_per_sample)
                    curr_clip = np.frombuffer(src.read(n_bytes_per_clip), dtype='<h')

                    # deinterleave channel streams for the current clip
                    curr_clip = curr_clip.reshape(-1, n_ch).transpose()

                    # accumulate clips for the relevant unit and clip medians across all channels in the range
                    unit_idx = clip_starts[clip_idx][1]
                    for i in range(count):
                        ch_idx = i + start
                        per_ch_clip = curr_clip[ch_idx]
                        clip_medians[ch_idx].append(np.median(np.abs(per_ch_clip)))
                        per_ch_template = unit_templates[unit_idx][ch_idx]
                        np.add(per_ch_template, per_ch_clip, out=per_ch_template)
                        unit_num_clips[unit_idx][ch_idx] += 1
                    clip_idx += 1

                    total_clips_so_far += count
                    next_update_pct = check_progress(next_update_pct)

        # noise estimate on each channel in channel bank
        noise_dict: Dict[int, float] = dict()
        for i in range(count):
            noise_dict[i+start] = np.median(clip_medians[i+start]) * 1.4826

        # for each unit, find which channel in the range had the highest SNR.
        res: Dict[str, Tuple[int, float]] = dict()
        for i in range(n_units):
            num_clips_dict = unit_num_clips[i]
            best_snr, primary_ch = 0.0, -1
            for k in range(count):
                ch_idx = k + start
                t: np.ndarray = unit_templates[i][ch_idx]
                if num_clips_dict[ch_idx] > 0:
                    t /= num_clips_dict[ch_idx]
                # if zero noise (in case a channel is dead/connected to ground) SNR is set to 0.
                snr = 0 if noise_dict[ch_idx] <= 0 else (np.max(t) - np.min(t)) / (1.96 * noise_dict[ch_idx])
                if snr > best_snr:
                    best_snr = snr
                    primary_ch = ch_idx
            res[units[i].uid] = (primary_ch, best_snr)

        progress_q.put_nowait((start, 100, ""))
        return res
    except Exception as e:
        progress_q.put_nowait((start, 0, str(e)))
        return {}


def cache_analog_channels_in_range(dir_path: str, start: int, count: int, progress_q: Queue, cancel: Event) -> None:
    """
    Extract, filter, and cache each analog data stream in the XSort working directory's analog source file for the range
    of data channel indices specified, writing each filtered stream to a separate internal cache file in the directory.

    This task function is intended to run in a separate process. The task manager should split up the entire set of
    analog channels into smaller ranges, each of which is assigned to a different process.

    NOTES:

    - Two major use cases -- interleaved or noninterleaved source. The Omniplex file is a noninterleaved source with a
      complex structure compared to a flat binary noninterleaved source.
    - Prefiltered flat binary source files need not be cached. The method fails is the source is prefiltered. An
      Omniplex file is always cached because its file structure is complex and it typically contains wideband data.
    - Strategy for noninterleaved case: Use a 4-count thread pool and assign each channel to a separate thread. Because
      the source is noninterleaved, each thread processes a different part of the source file. Read in and process in
      1200K chunks. [In testing with the 37GB 385-channel Neuropixel data, using 4 threads and 1200K read chunk sizes
      got the total execution time -- with the 385 channels split into 6 ranges on a 6-core machine -- down to ~55s;
      increasing the number of threads or the read chunk size did not further improve performance.]
    - When I used 5 or more threads in each process for the noninterleaved strategy, the task sometimes fails on a
      Errno 61 - Connection refused socket error. A CPython bug report indicated this is a MacOS-specific bug in
      Python's multiprocessing library, and a backport bug fix is provided for 3.11.
    - Strategy for interleaved case: The method sequentially reads in a large chunk, deinterleaves the streams, filters
      the streams for the channels in the range specified, and appends each channel's block to the corresponding cache
      file. Typical execution time is 150s on a 6-core machine. It is fundamentally slower than the noninterleaved case
      bc each process must read and process the entire source file to extract, cache and filter the channels in its
      assigned range. At least it is not 6x slower.

    We tried a number of other approaches to improve performance in the interleaved case, none of which worked as
    well (tested using the 37GB, 385-channel Neuropixel data):
     - One-thread-per-channel strategy (32 threads running in the XSort process). Took 130s to cache 32 of 385
       channels; estimate about 1600s to cache all 385.
     - MT pipeline strategy: One "reader thread" consumes the source file, reading reading one second's worth of
       channel data at a time and pushing each channel's one-second block onto one of 10 thread-safe queues.
       Each of ten "writer threads" service one of these queues, popping the next block off the queue and
       writing it to the corresponding channel's cache file. Execution time was highly variable, between 240 and
       660s.
     - MP pipeline strategy: Similar to the MT pipeline, but using one separate "reader process" to digest
       the source and push de-interleaved, filtered channel blocks onto a process-safe queue serviced by 10
       writer threads (spawned by the main XSort process). The idea here was to only read and process the source
       file once, in a separate process running on a different core. In reality, this did not work well: 80-115s
       to process only 20% of the 37GB file, which extrapolates to roughly 400-575s for the entire file. This might not
       work well bc the channel blocks have to be pickled and unpickled via the process-safe queue.
     - Variants on our one-process-per-channel-range strategy: The work of the main thread in each process is
       "serialized": read a raw buffer, process that buffer (deinterleave into N streams and filter each), and
       append the N channel blocks to the corresponding cache files. Tried introducing a block queue serviced
       by a pool of M writer threads so that main thread could keep working while the writer threads did the
       appends. Even tried a separate reader thread to read raw blocks sequentially from the source file into
       a "read queue". The queues were size-limited to ensure they did not grow too large. Nevertheless, the
       overall performance of these variants was somewhat worse than the basic serialized approach.

    :param dir_path: Full file system path to the XSort working directory.
    :param start: Index of first channel in the range, K
    :param count: The number of channels N in the range. If K+N is greater than the number of channels
        recorded, then the task processes all remaining channels starting at K.
    :param progress_q: A process-safe queue for delivering progress updates. Each update is in the form of a 3-tuple
        (start, pct, emsg), where start is the index of the first channel processed (int), pct is the percent complete
        (int), and emsg is an error description if the task has failed (otherwise an empty string). **The task aborts
        shortly after delivering an error message on the progress queue.**
    :param cancel: A process-safe event object which is set to request premature cancellation of this task function.
    """
    try:
        emsg, work_dir = WorkingDirectory.load_working_directory(Path(dir_path))
        if work_dir is None:
            raise Exception(emsg)

        # caching not needed for a prefiltered flat binary source
        if work_dir.is_analog_data_prefiltered:
            raise Exception("Analog data caching not required for a prefiltered source!")

        n_ch = work_dir.num_analog_channels()
        samples_per_sec = work_dir.analog_sampling_rate
        if start < 0 or start >= n_ch:
            raise Exception("Invalid channel start index.")
        elif start + count > n_ch:
            count = n_ch - start

        # for noninterleaved source, we use a small thread pool and cache each channel separately
        if work_dir.uses_omniplex_as_analog_source or not work_dir.is_analog_data_interleaved:
            progress_per_ch: Dict[int, int] = {i+start: 0 for i in range(count)}
            next_update_pct = 0
            task_func = _cache_pl2_analog_channel if work_dir.uses_omniplex_as_analog_source else \
                _cache_noninterleaved_analog_channel

            with ThreadPoolExecutor(max_workers=4) as mt_exec:
                thrd_q = Queue()
                futures: List[Future] = [mt_exec.submit(task_func, work_dir, i+start, thrd_q, cancel)
                                         for i in range(count)]

                first_error: Optional[str] = None
                while 1:
                    try:
                        # progress updates from per-channel thread tasks are: (ch_idx, pct_complete, error_msg)
                        update = thrd_q.get(timeout=0.2)
                        if len(update[2]) > 0:
                            # an error has occurred. If not bc task was cancelled, signal cancel now so that remaining
                            # threads stop. Also cancel any tasks that have not begun!
                            if not cancel.is_set():
                                cancel.set()
                                first_error = f"Error caching channel {update[0]}: {update[2]}"
                            for future in futures:
                                future.cancel()
                        else:
                            progress_per_ch[update[0]] = update[1]
                            total_progress = sum(progress_per_ch.values()) / len(progress_per_ch)
                            if total_progress >= next_update_pct:
                                progress_q.put_nowait((start, int(total_progress), ""))
                                next_update_pct += 5
                    except Empty:
                        pass

                    if all([future.done() for future in futures]):
                        break

                # report first error encountered, if any
                progress_q.put_nowait((start, 100, "" if first_error is None else first_error))

            return

        # HANDLE THE NONINTERLEAVED CASE FROM HERE

        # the channel cache files to be written
        cache_files: List[Path] = list()
        for i in range(count):
            cache_files.append(Path(work_dir.path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(start + i)}"))

        # prepare bandpass filter in case analog signal is wide-band. The filter delays are initialized with zero-vector
        # initial condition and the delays are updated as each block is filtered. SO MAINTAIN A SEPARATE FILTER DELAY
        # FOR EACH CHANNEL.
        [b, a] = scipy.signal.butter(2, [2 * 300 / samples_per_sec, 2 * 8000 / samples_per_sec], btype='bandpass')
        filter_ic = scipy.signal.lfiltic(b, a, np.zeros(max(len(b), len(a)) - 1))
        delays: List[np.ndarray] = [filter_ic.copy() for _ in range(count)]

        next_update_pct = 0
        file_size = work_dir.analog_source.stat().st_size
        if file_size % (2 * n_ch) != 0:
            raise Exception(f'Flat binary file size inconsistent with specified number of channels ({n_ch})')
        n_samples = int(file_size / (2 * n_ch))

        # for interleaved flat binary source, each "sample" is an array of N samples - one per channel
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
                for idx in range(start, start + count):
                    ch_block = curr_block[idx]
                    ch_block, delays[idx-start] = scipy.signal.lfilter(b, a, ch_block, axis=-1, zi=delays[idx-start])
                    ch_block = ch_block.astype(np.int16)
                    with open(cache_files[idx - start], 'ab') as f:
                        f.write(ch_block.tobytes())

                num_samples_read += n_samples_to_read

                if cancel.is_set():
                    raise Exception("Task cancelled")
                pct = int(num_samples_read * 100 / n_samples)
                if pct >= next_update_pct:
                    progress_q.put_nowait((start, pct, ""))
                    next_update_pct += 5

    except Exception as e:
        progress_q.put_nowait((start, 0, str(e)))


def _cache_pl2_analog_channel(work_dir: WorkingDirectory, idx: int, upd_q: Queue, cancel: Event) -> None:
    """
    Extract the analog data stream for one analog channel in the working directory's Omniplex analog data source,
    bandpass-filter it if necessary, and store it in a separate internal cache file (".xs.ch.<idx>", where <idx> is the
    channel index) within the directory.

    This method serves as the task function for a worker thread in which the work is done.

    :param work_dir: The XSort working directory.
    :param idx: The index of the analog channel to be cached.
    :param upd_q: A thread-safe queue for delivering progress updates. Each update is in the form of a 3-tuple
        (idx, pct, emsg), where idx is the analog channel index (int), pct is the percent complete (int), and emsg is
        an error description if the task has failed (otherwise an empty string).
    :param cancel: A thread-safe event object which is set to request premature cancellation of this task function. On
        cancellation, the internal cache file is removed.
    """
    cache_file = Path(work_dir.path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(idx)}")
    if cache_file.is_file():
        cache_file.unlink(missing_ok=True)

    samples_per_sec = work_dir.analog_sampling_rate

    # prepare bandpass filter in case analog signal is wide-band. The filter delays are initialized with zero-vector
    # initial condition and the delays are updated as each block is filtered...
    [b, a] = scipy.signal.butter(2, [2 * 300 / samples_per_sec, 2 * 8000 / samples_per_sec], btype='bandpass')
    filter_ic = scipy.signal.lfiltic(b, a, np.zeros(max(len(b), len(a)) - 1))

    next_update_pct = 0
    try:
        info = work_dir.omniplex_file_info
        channel_dict = info['analog_channels'][idx]
        num_blocks = len(channel_dict['block_num_items'])
        block_idx = 0
        is_wideband = (channel_dict['source'] == PL2.PL2_ANALOG_TYPE_WB)

        with open(work_dir.analog_source, 'rb') as src, open(cache_file, 'wb', buffering=1024*1024) as dst:
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

                # update progress after every 5% of blocks processed
                block_idx += 1
                if int(block_idx * 100 / num_blocks) >= next_update_pct:
                    upd_q.put_nowait((idx, next_update_pct, ""))
                    next_update_pct += 5

        upd_q.put_nowait((idx, 100, ""))
    except Exception as e:
        upd_q.put_nowait((idx, next_update_pct, str(e)))


def _cache_noninterleaved_analog_channel(work_dir: WorkingDirectory, idx: int, upd_q: Queue, cancel: Event) -> None:
    """
    Extract the analog data stream for one analog channel in the working directory's noninterleaved flat binary analog
    data source, bandpass-filter it, and store it in a separate internal cache file (".xs.ch.<idx>", where <idx> is the
    channel index) within the directory.

    This method serves as the task function for a worker thread in which the work is done.

    :param work_dir: The XSort working directory.
    :param idx: The index of the analog channel to be cached.
    :param upd_q: A thread-safe queue for delivering progress updates. Each update is in the form of a 3-tuple
        (idx, pct, emsg), where idx is the analog channel index (int), pct is the percent complete (int), and emsg is
        an error description if the task has failed (otherwise an empty string).
    :param cancel: A thread-safe event object which is set to request premature cancellation of this task function.
    """
    cache_file = Path(work_dir.path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(idx)}")
    if cache_file.is_file():
        cache_file.unlink(missing_ok=True)

    samples_per_sec = work_dir.analog_sampling_rate

    # prepare bandpass filter in case analog signal is wide-band. The filter delays are initialized with zero-vector
    # initial condition and the delays are updated as each block is filtered...
    [b, a] = scipy.signal.butter(2, [2 * 300 / samples_per_sec, 2 * 8000 / samples_per_sec], btype='bandpass')
    filter_ic = scipy.signal.lfiltic(b, a, np.zeros(max(len(b), len(a)) - 1))

    next_update_pct = 0
    try:
        # how many samples are stored for each analog channel
        n_ch = work_dir.num_analog_channels()
        file_size = work_dir.analog_source.stat().st_size
        if file_size % (2 * n_ch) != 0:
            raise Exception(f'Flat binary file size inconsistent with specified number of channels ({n_ch})')
        n_samples = int(file_size / (2 * n_ch))
        n_bytes_per_sample = 2

        # read chunk size in # samples
        chunk_sz = int(_READ_CHUNK_SIZE / n_bytes_per_sample)

        with open(work_dir.analog_source, 'rb') as src, open(cache_file, 'wb', buffering=1024*1024) as dst:
            # seek to start of contiguous block containing stream for specified channel in the noninterleaved source
            src.seek(idx * n_samples * 2)

            num_samples_read = 0
            while num_samples_read < n_samples:
                n_samples_to_read = min(chunk_sz, n_samples - num_samples_read)
                curr_block = np.frombuffer(src.read(n_samples_to_read * n_bytes_per_sample), dtype='<h')

                # filter the raw stream
                curr_block, filter_ic = scipy.signal.lfilter(b, a, curr_block, axis=-1, zi=filter_ic)
                curr_block = curr_block.astype(np.int16)

                # write block to cache file
                dst.write(curr_block.tobytes())

                # check for cancellation
                if cancel.is_set():
                    raise Exception("Task cancelled")

                # update progress after every 5% of the stream has been cached
                num_samples_read += n_samples_to_read
                if int(num_samples_read * 100 / n_samples) >= next_update_pct:
                    upd_q.put_nowait((idx, next_update_pct, ""))
                    next_update_pct += 5

        upd_q.put_nowait((idx, 100, ""))
    except Exception as e:
        upd_q.put_nowait((idx, 100, str(e)))


def delete_internal_cache_files(work_dir: WorkingDirectory, del_analog: bool = True, del_units: bool = True) -> None:
    """
    Remove all internal analog data and unit metrics cache files from an XSort working directory.

    :param work_dir: The working directory.
    :param del_analog: If True, analog data cache files are removed. Default = True.
    :param del_units: If True, neural unit metrics cache files are removed. Default = True.
    """
    if not (del_analog or del_units):
        return
    for p in work_dir.path.iterdir():
        if ((del_analog and p.name.startswith(CHANNEL_CACHE_FILE_PREFIX)) or
                (del_units and p.name.startswith(UNIT_CACHE_FILE_PREFIX))):
            p.unlink(missing_ok=True)


def extract_template_clips_interleaved_analog_src(
        work_dir: WorkingDirectory, n_clips: int, n_units: int, n_ch_proc: int, upd_q: Queue, cancel: Event) -> None:
    """
    Task function called by text fixture -- assessing single-threaded performance when extracting unit template clips
    from a very large interleaved flat binary source.

    :param work_dir: The XSort working directory.
    :param n_clips: Number of clips per unit.
    :param n_units: Number of units to process.
    :param n_ch_proc: Number of channels to process.
    :param upd_q: A thread-safe queue for delivering progress updates. Each update is in the form of a 2-tuple
        (pct, emsg), pct is the percent complete (int), and emsg is an error description if the task has failed
        (otherwise an empty string).
    :param cancel: A thread-safe event object which is set to request premature cancellation of this task function.
    """
    try:
        if not work_dir.is_analog_data_interleaved:
            raise Exception("Expected interleaved analog source")
        n_ch = work_dir.num_analog_channels()
        samples_per_sec = work_dir.analog_sampling_rate
        file_size = work_dir.analog_source.stat().st_size
        if file_size % (2 * n_ch) != 0:
            raise Exception(f'Flat binary file size inconsistent with specified number of channels ({n_ch})')
        n_samples = int(file_size / (2 * n_ch))
        template_len = int(samples_per_sec * 0.01)

        # for interleaved flat binary source, each "sample" is an array of N samples - one per channel
        n_bytes_per_sample = n_ch * 2
        n_bytes_per_clip = template_len * n_bytes_per_sample

        emsg, neurons = work_dir.load_neural_units()
        if len(emsg) > 0:
            raise Exception(emsg)
        n_units = min(n_units, len(neurons))

        n_ch_proc = min(n_ch_proc, n_ch)
        unit_num_clips: List[Dict[int, int]] = list()
        unit_templates: List[Dict[int, np.ndarray]] = list()
        clip_starts: List[Tuple[int, int]] = list()
        for _ in range(n_units):
            template_dict: Dict[int, np.ndarray] = dict()
            num_clips_dict: Dict[int, int] = dict()
            for i in range(n_ch_proc):
                template_dict[i] = np.zeros(template_len, dtype='<f')
                num_clips_dict[i] = 0
            unit_templates.append(template_dict)
            unit_num_clips.append(num_clips_dict)

        for unit_idx in range(n_units):
            u = neurons[unit_idx]
            n_spks = min(u.num_spikes, n_clips)
            if n_spks < u.num_spikes:
                spike_indices = sorted(random.sample(range(u.num_spikes), n_spks))
                unit_clip_starts = [(int((u.spike_times[i] - 0.001) * samples_per_sec), unit_idx)
                                    for i in spike_indices]
            else:
                unit_clip_starts = [(int((t - 0.001) * samples_per_sec), unit_idx) for t in u.spike_times]
            clip_starts.extend(unit_clip_starts)

        # clips must be in chronological order!
        clip_starts.sort(key=lambda x: x[0])

        # strip off clips that are cut off at beginning or end of recording
        while (len(clip_starts) > 0) and (clip_starts[0][0] < 0):
            clip_starts.pop(0)
        while (len(clip_starts) > 0) and (clip_starts[-1][0] + template_len >= n_samples):
            clip_starts.pop()
        if len(clip_starts) == 0:
            raise Exception("No spike clips to process!")

        # we compute per-channel clip medians on a limited number of extracted clips to assess noise on each channel
        # TODO: CONTINUE HERE 3/27 -- Want to calculate noise using different numbers of clips and see what how the
        #   noise levels change across channels. Do in separate test fixture....
        clip_medians: Dict[int, List[float]] = dict()
        for i in range(n_ch_proc):
            clip_medians[i] = list()

        with open(work_dir.analog_source, 'rb') as src:
            clip_idx, next_update_pct = 0, 0
            while clip_idx < len(clip_starts):
                # seek to start of next clip to be processed
                chunk_s = clip_starts[clip_idx][0]
                src.seek(chunk_s * n_bytes_per_sample)

                # read a chunk of size M <= _READ_CHUNK_SIZE that fully contains 1 or more clips. The goal here is to
                # reduce the total number of reads required to process file. This significantly improved performance.
                n = 1
                while ((clip_idx + n < len(clip_starts)) and
                       ((clip_starts[clip_idx + n - 1][0] - chunk_s) * n_bytes_per_sample +
                        n_bytes_per_clip < _READ_CHUNK_SIZE)):
                    n += 1
                chunk_sz = (clip_starts[clip_idx + n - 1][0] - chunk_s) * n_bytes_per_sample + n_bytes_per_clip
                chunk = np.frombuffer(src.read(chunk_sz), dtype='<h')

                # deinterleave the chunk
                chunk = chunk.reshape(-1, n_ch).transpose()

                # process all clips in the chunk
                for k in range(n):
                    start, unit_idx = clip_starts[clip_idx + k][0] - chunk_s, clip_starts[clip_idx + k][1]
                    for ch_idx in range(n_ch_proc):
                        per_ch_clip = chunk[ch_idx][start:start + template_len]
                        template = unit_templates[unit_idx][ch_idx]
                        np.add(template, per_ch_clip, out=template)
                        unit_num_clips[unit_idx][ch_idx] += 1
                clip_idx += n

                if cancel.is_set():
                    raise Exception("Task cancelled")
                pct = int(clip_idx * 100 / len(clip_starts))
                if pct >= next_update_pct:
                    upd_q.put_nowait((pct, ""))
                    next_update_pct += 5

        avg_clip_delta = 0
        for i in range(len(clip_starts)-1):
            avg_clip_delta += clip_starts[i+1][0] - clip_starts[i][0]
        avg_clip_delta /= len(clip_starts) - 1
        print(f"Total clips = {len(clip_starts) * n_ch_proc}, avg clip start delta = {avg_clip_delta:.1f}", flush=True)

    except Exception as e:
        upd_q.put_nowait((0, str(e)))


def estimate_noise_interleaved_analog_src(work_dir: WorkingDirectory, upd_q: Queue, cancel: Event) -> None:
    """
    Task function called by test fixture -- that estimates noise level on every channel in an interleaved flat binary
    analog source using different strategies.

    :param work_dir: The XSort working directory.
    :param upd_q: A thread-safe queue for delivering progress updates. Each update is in the form of a 2-tuple
        (pct, emsg), pct is the percent complete (int), and emsg is an error description if the task has failed
        (otherwise an empty string).
    :param cancel: A thread-safe event object which is set to request premature cancellation of this task function.
    """
    try:
        if not work_dir.is_analog_data_interleaved:
            raise Exception("Expected interleaved analog source")
        n_ch = work_dir.num_analog_channels()
        file_size = work_dir.analog_source.stat().st_size
        if file_size % (2 * n_ch) != 0:
            raise Exception(f'Flat binary file size inconsistent with specified number of channels ({n_ch})')
        n_samples = int(file_size / (2 * n_ch))
        dur = n_samples / work_dir.analog_sampling_rate
        emsg, neurons = work_dir.load_neural_units()
        if len(emsg) > 0:
            raise Exception(emsg)

        print("Testing different methods for measuring channel noise...", flush=True)
        print(f"Source: {work_dir.analog_source.name}, #channels={n_ch}, #samples={n_samples}, duration={dur:.3f}s.",
              flush=True)
        _noise_method_1(work_dir, neurons, upd_q, cancel)
        _noise_method_2(work_dir, neurons, upd_q, cancel)
        _noise_method_3(work_dir, upd_q, cancel)

    except Exception as e:
        upd_q.put_nowait((0, str(e)))


def _noise_method_1(work_dir: WorkingDirectory, neurons: List[Neuron], upd_q: Queue, cancel: Event) -> None:
    """
    Estimates noise using random clips selected based on unit spike times, concatenating clips into a long 1D
    vector V, then noise = 1.4826*np.median(np.abs(V)).
    """
    n_ch = work_dir.num_analog_channels()
    samples_per_sec = work_dir.analog_sampling_rate
    file_size = work_dir.analog_source.stat().st_size
    n_samples = int(file_size / (2 * n_ch))
    template_len = int(samples_per_sec * 0.01)
    n_bytes_per_sample = n_ch * 2
    n_bytes_per_clip = template_len * n_bytes_per_sample
    n_units = len(neurons)

    print("\nMETHOD 1: Randomly select clips from spike times. noise = 1.4826*np.median(np.abs(concatenated clips):")
    # we'll repeat our noise estimate for different numbers of extracted clips
    noise_clips: List[np.ndarray] = list()
    per_loop_pct = 5
    next_update_pct = 0
    for loop_idx, n_clips in enumerate(reversed([100, 200, 500, 1000, 10000])):
        t0 = time.perf_counter()
        # for every channel, we concatenate extracted clips (horzcat) into one long vector which we then use to
        # estimate noise
        noise_clips.clear()
        for _ in range(n_ch):
            noise_clips.append(np.zeros(template_len * n_clips, dtype='<h'))

        # first, we create a big list of clip starts time by randomly choosing 100 spikes (or less) from each unit
        clip_starts: List[int] = list()
        for unit_idx in range(n_units):
            u = neurons[unit_idx]
            n_spks = min(u.num_spikes, n_clips)
            if n_spks < u.num_spikes:
                spike_indices = sorted(random.sample(range(u.num_spikes), n_spks))
                unit_clip_starts = [int((u.spike_times[i] - 0.001) * samples_per_sec) for i in spike_indices]
            else:
                unit_clip_starts = [int((t - 0.001) * samples_per_sec) for t in u.spike_times]
            clip_starts.extend(unit_clip_starts)
        clip_starts.sort()

        # strip off clips that are cut off at beginning or end of recording
        while (len(clip_starts) > 0) and (clip_starts[0] < 0):
            clip_starts.pop(0)
        while (len(clip_starts) > 0) and (clip_starts[-1] + template_len >= n_samples):
            clip_starts.pop()
        if len(clip_starts) < n_clips:
            raise Exception("Not enough clips to process!")

        # now we randomly select just a few of these clips to extract and estimate noise.
        clip_indices = sorted(random.sample(range(len(clip_starts)), n_clips))
        noise_clip_starts = [clip_starts[i] for i in clip_indices]
        noise_clip_starts.sort()
        clip_starts.clear()

        # scan file and extract per-channel clips, concatenating them into one long vector (per channel)
        with open(work_dir.analog_source, 'rb') as src:
            clip_idx = 0
            while clip_idx < n_clips:
                # seek to start of next clip to be processed, then read in THAT CLIP ONLY
                sample_idx = noise_clip_starts[clip_idx]
                src.seek(sample_idx * n_bytes_per_sample)
                curr_clip = np.frombuffer(src.read(n_bytes_per_clip), dtype='<h')

                # deinterleave
                curr_clip = curr_clip.reshape(-1, n_ch).transpose()

                # for each channel, concatenate clip to the growing voltage vector for that channel
                start = clip_idx * template_len
                end = start + template_len
                for k in range(n_ch):
                    noise_clips[k][start:end] = curr_clip[k]

                clip_idx += 1

                if cancel.is_set():
                    raise Exception("Task cancelled")
                pct = loop_idx * per_loop_pct + int(clip_idx * per_loop_pct / n_clips)
                if pct >= next_update_pct:
                    upd_q.put_nowait((pct, ""))
                    next_update_pct += 5

        # estimate noise level on each channel
        noise_estimates: List[float] = [1.4826 * np.median(np.abs(noise_clips[k])) for k in range(n_ch)]

        t0 = time.perf_counter() - t0
        exec_time = f"{t0:6.3f}"
        listing = ", ".join([f"ch{k}={noise_estimates[k]:5.3f}" for k in range(10)])
        last_ch = f"ch{n_ch - 1}={noise_estimates[-1]:5.3f} ({np.count_nonzero(noise_clips[-1])} nonzeros)"
        min_max = f"[{noise_clip_starts[0]} .. {noise_clip_starts[-1]}]"
        print(f"n_clips={n_clips:>5}: T={exec_time}s. {min_max: >24} | {listing}, .. {last_ch}", flush=True)


def _noise_method_2(work_dir: WorkingDirectory, neurons: List[Neuron], upd_q: Queue, cancel: Event) -> None:
    """
    Estimates noise using random clips selected based on unit spike times. Gather list L of clip medians, where clip
    median = np.median(np.abs(clip)), then noise = 1.4826*np.median(L).
    """
    n_ch = work_dir.num_analog_channels()
    samples_per_sec = work_dir.analog_sampling_rate
    file_size = work_dir.analog_source.stat().st_size
    n_samples = int(file_size / (2 * n_ch))
    template_len = int(samples_per_sec * 0.01)
    n_bytes_per_sample = n_ch * 2
    n_bytes_per_clip = template_len * n_bytes_per_sample
    n_units = len(neurons)

    print("\nMETHOD 2: Randomly select clips from spike times. noise = 1.4826*(median of clip medians):")
    # we'll repeat our noise estimate for different numbers of extracted clips
    clip_medians: List[List[float]] = list()
    per_loop_pct = 10
    init_pct = 25
    next_update_pct = 30
    for loop_idx, n_clips in enumerate(reversed([100, 200, 500, 1000, 10000])):
        t0 = time.perf_counter()

        # for accumulating clip medians
        clip_medians.clear()
        for _ in range(n_ch):
            clip_medians.append(list())

        # first, we create a big list of clip starts time by randomly choosing 100 spikes (or less) from each unit
        clip_starts: List[int] = list()
        for unit_idx in range(n_units):
            u = neurons[unit_idx]
            n_spks = min(u.num_spikes, n_clips)
            if n_spks < u.num_spikes:
                spike_indices = sorted(random.sample(range(u.num_spikes), n_spks))
                unit_clip_starts = [int((u.spike_times[i] - 0.001) * samples_per_sec) for i in spike_indices]
            else:
                unit_clip_starts = [int((t - 0.001) * samples_per_sec) for t in u.spike_times]
            clip_starts.extend(unit_clip_starts)
        clip_starts.sort()

        # strip off clips that are cut off at beginning or end of recording
        while (len(clip_starts) > 0) and (clip_starts[0] < 0):
            clip_starts.pop(0)
        while (len(clip_starts) > 0) and (clip_starts[-1] + template_len >= n_samples):
            clip_starts.pop()
        if len(clip_starts) < n_clips:
            raise Exception("Not enough clips to process!")

        # now we randomly select just a few of these clips to extract and estimate noise.
        clip_indices = sorted(random.sample(range(len(clip_starts)), n_clips))
        noise_clip_starts = [clip_starts[i] for i in clip_indices]
        noise_clip_starts.sort()
        clip_starts.clear()

        # scan file and extract per-channel clips, concatenating them into one long vector (per channel)
        with open(work_dir.analog_source, 'rb') as src:
            clip_idx = 0
            while clip_idx < n_clips:
                # seek to start of next clip to be processed, then read in THAT CLIP ONLY
                sample_idx = noise_clip_starts[clip_idx]
                src.seek(sample_idx * n_bytes_per_sample)
                curr_clip = np.frombuffer(src.read(n_bytes_per_clip), dtype='<h')

                # deinterleave
                curr_clip = curr_clip.reshape(-1, n_ch).transpose()

                # for each channel, compute clip median and append to growing list of clip medians for that channel
                for k in range(n_ch):
                    clip_medians[k].append(np.median(np.abs(curr_clip[k])))

                clip_idx += 1

                if cancel.is_set():
                    raise Exception("Task cancelled")
                pct = init_pct + loop_idx * per_loop_pct + int(clip_idx * per_loop_pct / n_clips)
                if pct >= next_update_pct:
                    upd_q.put_nowait((pct, ""))
                    next_update_pct += 5

        # estimate noise level on each channel
        noise_estimates: List[float] = [1.4826 * np.median(clip_medians[k]) for k in range(n_ch)]

        t0 = time.perf_counter() - t0
        exec_time = f"{t0:6.3f}"
        listing = ", ".join([f"ch{k}={noise_estimates[k]:5.3f}" for k in range(10)])
        last_ch = f"ch{n_ch - 1}={noise_estimates[-1]:5.3f} ({np.count_nonzero(clip_medians[-1])} nonzeros)"
        min_max = f"[{noise_clip_starts[0]} .. {noise_clip_starts[-1]}]"
        print(f"n_clips={n_clips:>5}: T={exec_time}s. {min_max: >24} | {listing}, .. {last_ch}", flush=True)


def _noise_method_3(work_dir: WorkingDirectory, upd_q: Queue, cancel: Event) -> None:
    """
    Estimates noise using random clips selected across entire recording without regard to unit spike times,
    concatenating clips into a long 1D vector V, then noise = 1.4826*np.median(np.abs(V)).
    """
    n_ch = work_dir.num_analog_channels()
    samples_per_sec = work_dir.analog_sampling_rate
    file_size = work_dir.analog_source.stat().st_size
    n_samples = int(file_size / (2 * n_ch))
    template_len = int(samples_per_sec * 0.01)
    n_bytes_per_sample = n_ch * 2
    n_bytes_per_clip = template_len * n_bytes_per_sample

    print("\nMETHOD 3: Randomly select clips from entire recording without regard to unit spikes. "
          "noise = 1.4826*np.median(np.abs(concatenated clips):")

    # we'll repeat our noise estimate for different numbers of extracted clips
    noise_clips: List[np.ndarray] = list()
    per_loop_pct = 5
    init_pct = 75
    next_update_pct = 80
    for loop_idx, n_clips in enumerate(reversed([100, 200, 500, 1000, 10000])):
        t0 = time.perf_counter()
        # for every channel, we concatenate extracted clips (horzcat) into one long vector which we then use to
        # estimate noise
        noise_clips.clear()
        for _ in range(n_ch):
            noise_clips.append(np.zeros(template_len * n_clips, dtype='<h'))

        # randomly select 50000 clip start times across entire recording and strip off clips cut off at the end
        clip_starts = sorted(random.sample(range(n_samples), 50000))
        while (len(clip_starts) > 0) and (clip_starts[-1] + template_len >= n_samples):
            clip_starts.pop()

        # from those, randomly select the desired number of clips to process
        chosen = sorted(random.sample(range(len(clip_starts)), n_clips))
        noise_clip_starts = [clip_starts[i] for i in chosen]
        noise_clip_starts.sort()
        clip_starts.clear()

        # scan file and extract per-channel clips, concatenating them into one long vector (per channel)
        with open(work_dir.analog_source, 'rb') as src:
            clip_idx = 0
            while clip_idx < n_clips:
                # seek to start of next clip to be processed, then read in THAT CLIP ONLY
                sample_idx = noise_clip_starts[clip_idx]
                src.seek(sample_idx * n_bytes_per_sample)
                curr_clip = np.frombuffer(src.read(n_bytes_per_clip), dtype='<h')

                # deinterleave
                curr_clip = curr_clip.reshape(-1, n_ch).transpose()

                # for each channel, concatenate clip to the growing voltage vector for that channel
                start = clip_idx * template_len
                end = start + template_len
                for k in range(n_ch):
                    noise_clips[k][start:end] = curr_clip[k]

                clip_idx += 1

                if cancel.is_set():
                    raise Exception("Task cancelled")
                pct = init_pct + loop_idx * per_loop_pct + int(clip_idx * per_loop_pct / n_clips)
                if pct >= next_update_pct:
                    upd_q.put_nowait((pct, ""))
                    next_update_pct += 5

        # estimate noise level on each channel
        # TODO: Here we're using noise = 1.4826 * np.median(np.abs(V-np.median(V)))
        noise_estimates: List[float] = list()
        for k in range(n_ch):
            m = np.median(noise_clips[k])
            noise_estimates.append(1.4826 * np.median(np.abs(noise_clips[k]-m)))

        t0 = time.perf_counter() - t0
        exec_time = f"{t0:6.3f}"
        listing = ", ".join([f"ch{k}={noise_estimates[k]:5.3f}" for k in range(10)])
        last_ch = f"ch{n_ch - 1}={noise_estimates[-1]:5.3f} ({np.count_nonzero(noise_clips[-1])} nonzeros)"
        min_max = f"[{noise_clip_starts[0]} .. {noise_clip_starts[-1]}]"
        print(f"n_clips={n_clips:>5}: T={exec_time}s. {min_max: >24} | {listing}, .. {last_ch}", flush=True)


def read_interleaved_analog_source(
        work_dir: WorkingDirectory, chunk_sz: int,  upd_q: Queue, cancel: Event) -> None:
    """
    Task function called by test fixture -- for evaluating how read chunk size affects how long it takes to read a
    very large interleaved flat binary file. The raw data is converted to 16-bit samples and deinterleaved, but no
    other processing is done.

    :param work_dir: The XSort working directory.
    :param chunk_sz: Maximum size of each chunk read in bytes, B. Since the channel streams are interleaved, the actual
        chunk size may be adjusted lower to ensure that read in an integer number of scans. Each "scan" contains one
        sample for each analog channel, or N*2 bytes for N channels.
    :param upd_q: A thread-safe queue for delivering progress updates. Each update is in the form of a 2-tuple
        (pct, emsg), pct is the percent complete (int), and emsg is an error description if the task has failed
        (otherwise an empty string).
    :param cancel: A thread-safe event object which is set to request premature cancellation of this task function.
    """
    try:
        if not work_dir.is_analog_data_interleaved:
            raise Exception("Expected interleaved analog source")
        n_ch = work_dir.num_analog_channels()
        file_size = work_dir.analog_source.stat().st_size
        if file_size % (2 * n_ch) != 0:
            raise Exception(f'Flat binary file size inconsistent with specified number of channels ({n_ch})')
        n_samples = int(file_size / (2 * n_ch))

        # for interleaved flat binary source, each "sample" is an array of N samples - one per channel
        n_bytes_per_sample = n_ch * 2
        n_samples_per_chunk = int(chunk_sz / n_bytes_per_sample)
        if n_samples_per_chunk == 0:
            raise Exception(f"Requested read chunk size is too small")
        n_bytes_per_chunk = n_samples_per_chunk * n_bytes_per_sample

        total_samples = 0
        with open(work_dir.analog_source, 'rb') as src:
            n_samples_read, next_update_pct = 0, 0
            while n_samples_read < n_samples:
                n_samples_to_read = min(n_samples_per_chunk, n_samples - n_samples_read)
                curr_block = np.frombuffer(src.read(n_samples_to_read * n_bytes_per_sample), dtype='<h')

                curr_block = curr_block.reshape(-1, n_ch).transpose()
                for i in range(n_ch):
                    total_samples += len(curr_block[i])

                n_samples_read += n_samples_to_read
                if cancel.is_set():
                    raise Exception("Task cancelled")
                pct = int(n_samples_read * 100 / n_samples)
                if pct >= next_update_pct:
                    upd_q.put_nowait((pct, ""))
                    next_update_pct += 5

        print(f"Read chunk size={n_bytes_per_chunk}, total samples={total_samples}", flush=True)

    except Exception as e:
        upd_q.put_nowait((0, str(e)))
