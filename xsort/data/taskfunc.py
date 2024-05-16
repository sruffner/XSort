import random
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from queue import Queue, Empty
from threading import Event
from typing import Dict, List, Tuple, Union, Optional, Any, Set

import numpy as np
import scipy

from xsort.data import PL2, stats
from xsort.data.files import WorkingDirectory, CHANNEL_CACHE_FILE_PREFIX
from xsort.data.neuron import Neuron, ChannelTraceSegment, DataType

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
NCLIPS_FOR_PCA: int = 5000
"""
When PCA is performed on a single neural unit, we use this many randomly selected spike multi-clips (horizontal concat
of spike clip on each of the up to 16 channels "near" the unit's primary channel) to calculate principal components. 
When more than one unit is selected, PCA calculation uses the spike templates for all units on the 16 (or fewer)
channels that are common to all units.
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
SPIKES_PER_BATCH: int = 20000
""" Batch size used when projecting all spike clips for a unit to 2D space defined by 2 principal components. """


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
        in the form of a 2-tuple: (task_id, pct), where pct is the integer completion percentage; or (task_id, emsg),
        where emsg is an error description. **The task function finishes immediately after delivering an error message
        to the queue.**
    :param cancel: A process-safe event object which is set to request premature cancellation of this task function.
    """
    try:
        emsg, work_dir = WorkingDirectory.load_working_directory(Path(dir_path))
        if work_dir is None:
            raise Exception(emsg)
        if work_dir.num_analog_channels() > N_MAX_TEMPLATES_PER_UNIT:
            raise Exception(f"Not supported for recording session with >{N_MAX_TEMPLATES_PER_UNIT} analog channels.")

        emsg, neurons = work_dir.load_select_neural_units(uids)
        if len(emsg) > 0:
            raise Exception(emsg)

        noise_levels = work_dir.load_channel_noise_from_cache()
        if noise_levels is None:
            raise Exception("Missing internal cache file containing channel noise estimates")

        if cancel.is_set():
            raise Exception("Task canceled")

        # spawn a thread per channel to calculate unit templates
        with ThreadPoolExecutor(max_workers=work_dir.num_analog_channels()) as mt_exec:
            progress_per_ch: Dict[int, int] = {idx: 0 for idx in work_dir.analog_channel_indices}
            next_progress_update_pct = 0

            # NOTE: Previously used the multiprocess event object supplied as an argument as the cancel signal for
            # the threads spawned here, but that has caused issues on MacOS. Instead, each process has its own
            # thread-safe cancel event
            thrd_cancel = Event()

            thrd_q = Queue()
            futures: List[Future] = \
                [mt_exec.submit(_compute_unit_templates_on_channel, work_dir, idx, neurons, thrd_q, thrd_cancel,
                                N_TEMPLATE_CLIPS)
                 for idx in work_dir.analog_channel_indices]

            first_error: Optional[str] = None
            cancelled_already = False
            while 1:
                trigger_cancel = False
                try:
                    # progress updates are: (ch_idx, pct_complete, error_msg)
                    update = thrd_q.get(timeout=0.2)
                    if len(update[2]) > 0:
                        if first_error is None:
                            first_error = f"Error computing unit templates on channel {update[0]}: {update[2]}"
                        trigger_cancel = True
                    else:
                        progress_per_ch[update[0]] = update[1]
                        total_progress = sum(progress_per_ch.values()) / len(progress_per_ch)
                        if total_progress >= next_progress_update_pct:
                            progress_q.put_nowait((task_id, next_progress_update_pct, ""))
                            next_progress_update_pct += 5
                except Empty:
                    pass

                # if no error and we haven't cancelled, check for cancel from master
                if not (trigger_cancel or cancelled_already):
                    trigger_cancel = cancel.is_set()

                # trigger cancel on error or master cancel: Must cancel both running and not yet started tasks
                # spawned by this process, and inform other processes spawned by master
                if trigger_cancel and not cancelled_already:
                    thrd_cancel.set()
                    for future in futures:
                        future.cancel()
                    cancel.set()
                    cancelled_already = True  # only need to do this once!

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
                ch_idx, template_dict = future.result()
                noise = noise_levels[ch_idx]
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
        -> Optional[Tuple[int, Dict[str, np.ndarray]]]:
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
    :return: A 2-tuple [ch_idx, template_dict], where: ch_idx is the index of the analog channel processed; and
        template_dict is a dictionary, keyed by unit UID, holding the computed spike templates. **Template samples
        are in raw ADC units (NOT converted to microvolts).** If task fails, returns None.
    """
    try:
        # we either read directly from the original, prefiltered flat binary source or an internal cache file for the
        # analog channel index specified.
        if work_dir.need_analog_cache:
            ch_file = Path(work_dir.cache_path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(ch_idx)}")
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

                # accumulate extracted spike clip for the relevant unit
                i_unit = clip_starts[i_clip + _k][1]
                _template = unit_templates[i_unit]
                np.add(_template, clip, out=_template)
                unit_num_clips[i_unit] += 1
            return n

        # extract and accumulate all spike waveform clips, across all units, from the source file. If we're dealing with
        # a flat binary file with interleaving, one sample is actually one scan of the N channels recorded!
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
        for i in range(n_units):
            num_clips = unit_num_clips[i]
            template: np.ndarray = unit_templates[i]
            if num_clips > 0:
                template /= num_clips
            template_dict[units[i].uid] = template - np.median(template)   # remove DC offset

        return ch_idx, template_dict

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
        in the form of a 2-tuple: (task_id, pct), where pct is the percent complete (int); OR (task_id, emsg), where
        emsg is an error description (str) indicating why the task failed. **The task function finishes immediately
        after delivering an error message to the queue.**
    :param cancel: A process-safe event object which is set to request premature cancellation of this task function.
    """
    try:
        emsg, work_dir = WorkingDirectory.load_working_directory(Path(dir_path))
        if work_dir is None:
            raise Exception(emsg)

        emsg, neurons = work_dir.load_select_neural_units([k for k in unit_to_primary.keys()])
        if len(emsg) > 0:
            raise Exception(emsg)
        if not work_dir.channel_noise_cache_file_exists():
            raise Exception('Missing channel noise cache')
        if cancel.is_set():
            raise Exception("Task canceled")

        # spawn a thread per unit to calculate unit templates and cache metrics to file
        with ThreadPoolExecutor(max_workers=min(16, len(neurons))) as mt_exec:
            progress_per_unit: Dict[str, int] = {u.uid: 0 for u in neurons}
            next_progress_update_pct = 0

            # NOTE: Previously used the multiprocess event object supplied as an argument as the cancel signal for
            # the threads spawned here, but that has caused issues on MacOS. Instead, each process has its own
            # thread-safe cancel event
            thrd_cancel = Event()

            thrd_q = Queue()
            futures: List[Future] = \
                [mt_exec.submit(_compute_templates_and_cache_metrics_for_unit, work_dir, u, unit_to_primary[u.uid],
                                thrd_q, thrd_cancel, N_TEMPLATE_CLIPS) for u in neurons]

            first_error: Optional[str] = None
            cancelled_already = False
            while 1:
                trigger_cancel = False
                try:
                    # progress updates are: (uid, pct_complete, error_msg)
                    update = thrd_q.get(timeout=0.2)
                    if len(update[2]) > 0:
                        if first_error is None:
                            first_error = f"Error caching metrics for unit {update[0]}: {update[2]}"
                        trigger_cancel = True
                    else:
                        progress_per_unit[update[0]] = update[1]
                        total_progress = sum(progress_per_unit.values()) / len(progress_per_unit)
                        if total_progress >= next_progress_update_pct:
                            progress_q.put_nowait((task_id, next_progress_update_pct))
                            next_progress_update_pct += 5
                except Empty:
                    pass

                # if no error and we haven't cancelled, check for cancel from master
                if not (trigger_cancel or cancelled_already):
                    trigger_cancel = cancel.is_set()

                # trigger cancel on error or master cancel: Must cancel both running and not yet started tasks
                # spawned by this process, and inform other processes spawned by master
                if trigger_cancel and not cancelled_already:
                    thrd_cancel.set()
                    for future in futures:
                        future.cancel()
                    cancel.set()
                    cancelled_already = True  # only need to do this once!

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
                progress_q.put_nowait((task_id, 100))
            else:
                progress_q.put_nowait((task_id, first_error))

    except Exception as e:
        progress_q.put_nowait((task_id, f"Error caching unit metrics (task_id={task_id}): {str(e)}"))


def _compute_templates_and_cache_metrics_for_unit(work_dir: WorkingDirectory, unit: Neuron, primary_ch: int,
                                                  progress_q: Queue, cancel: Event, n_max_spks: int) -> None:
    """
    Helper task for cache_neural_units_select_channels() calculates the mean spike waveforms (templates) for the
    specified neural unit on a contiguous bank of analog channels near the unit's specified "primary channel", then
    writes the unit metrics to an internal cache file in the XSort working directory. Intended to be performed on a
    separate thread.

    NOTE: The primary channel is "identified" by computing templates and SNR for a small random sampling of clips
    across ALL available channels, then choosing the channel on which the unit's SNR was greatest. To compute the
    cached templates, many more clips are averaged, so the identity of the primary channel could change -- although it
    is assumed to be "in the neighborhood" of the originally identified primary channel.

    :param work_dir: The XSort working directory.
    :param unit: The neural unit for which metrics are to be computed.
    :param primary_ch: Index P of the analog data channel designated as the primary channel for the neural unit. Given
        N = N_MAX_TEMPLATES_PER_UNIT and K = total # of analog channels, the method compute spike templates on
        channels [P-N/2 .. P-N/2 + N-1]. If P < N/2, then [0 .. N-1] is used; if P > K-N/2, then [K-N, K-1] is used.
    :param progress_q: A thread-safe queue for delivering progress updates. Each "update" is in the form of a
        3-tuple (uid, pct, emsg), where uid is the unit's UID, pct is the percent complete (int), and emsg is an error
        description if the task has failed (otherwise an empty string).
    :param cancel: A thread-safe event object which is set to request premature cancellation of this task function.
    :param n_max_spks: The maximum number of spike waveform clips to process when computing the spike templates.
        If <= 0, then the entire spike train is included in the calculation.
    """
    try:
        # expect to find per-channel noise levels in internal cache file
        noise_levels = work_dir.load_channel_noise_from_cache()
        if noise_levels is None:
            raise Exception("Missing or unreadable channel noise cache")

        # the list of channel indices on which we'll compute unit templates
        n_ch, half_w = work_dir.num_analog_channels(), int(N_MAX_TEMPLATES_PER_UNIT/2)
        first_ch = 0 if primary_ch < half_w else \
            (n_ch-N_MAX_TEMPLATES_PER_UNIT if (primary_ch > n_ch-half_w) else primary_ch-half_w)
        ch_indices = [i + first_ch for i in range(N_MAX_TEMPLATES_PER_UNIT)]

        # we either read directly from the original, prefiltered flat binary source or a set of internal cache files
        interleaved = False
        n_bytes_per_sample = 2
        num_samples_recorded = 0
        if work_dir.need_analog_cache:
            for i in ch_indices:
                ch_file = Path(work_dir.cache_path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(i)}")
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
            if work_dir.analog_source.stat().st_size % (n_ch * 2) != 0:
                raise Exception(f"Bad file size for analog source file: {work_dir.analog_source.name}")
            num_samples_recorded = int(work_dir.analog_source.stat().st_size / (n_ch * 2))

        samples_per_sec = work_dir.analog_sampling_rate
        template_len = int(samples_per_sec * 0.01)

        # for accumulating clips on each channel
        template_dict: Dict[int, np.ndarray] = {i: np.zeros(template_len, dtype='<f') for i in ch_indices}

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

        n_bytes_per_clip = template_len * n_bytes_per_sample

        # for tracking overall progress
        next_update_pct = 0
        total_clips = len(ch_indices) * len(clip_starts)
        total_clips_so_far = 0

        # extract and accumulate all spike waveform clips for the unit from the source file(s).
        if work_dir.need_analog_cache:
            # CASE 1: individual binary cache file for each analog channel stream
            for ch_idx in ch_indices:
                ch_file = Path(work_dir.cache_path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(ch_idx)}")
                per_ch_template = template_dict[ch_idx]
                with open(ch_file, 'rb') as src:
                    clip_idx = 0
                    while clip_idx < len(clip_starts):
                        # seek to start of next clip to be processed
                        chunk_s = clip_starts[clip_idx]
                        src.seek(chunk_s * n_bytes_per_sample)

                        # read a chunk of size M <= _READ_CHUNK_SIZE that fully contains 1 or more clips
                        n = 1
                        while ((clip_idx + n < len(clip_starts)) and
                               ((clip_starts[clip_idx + n - 1] - chunk_s) * n_bytes_per_sample +
                                n_bytes_per_clip < _READ_CHUNK_SIZE)):
                            n += 1
                        chunk_sz = (clip_starts[clip_idx + n - 1] - chunk_s) * n_bytes_per_sample + n_bytes_per_clip
                        chunk = np.frombuffer(src.read(chunk_sz), dtype='<h')

                        # accumulate all clips in that chunk
                        for k in range(n):
                            s = clip_starts[clip_idx + k] - chunk_s
                            clip = chunk[s:s + template_len]

                            # accumulate extracted spike clip for the relevant unit
                            np.add(per_ch_template, clip, out=per_ch_template)
                        clip_idx += n

                        # get ready for next clip, check for cancellation, and report progress
                        if cancel.is_set():
                            raise Exception("Task cancelled")
                        total_clips_so_far += n
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
                        # seek to start of next clip to be processed
                        chunk_s = clip_starts[clip_idx]
                        src.seek(ofs + chunk_s * n_bytes_per_sample)

                        # read a chunk of size M <= _READ_CHUNK_SIZE that fully contains 1 or more clips
                        n = 1
                        while ((clip_idx + n < len(clip_starts)) and
                               ((clip_starts[clip_idx + n - 1] - chunk_s) * n_bytes_per_sample +
                                n_bytes_per_clip < _READ_CHUNK_SIZE)):
                            n += 1
                        chunk_sz = (clip_starts[clip_idx + n - 1] - chunk_s) * n_bytes_per_sample + n_bytes_per_clip
                        chunk = np.frombuffer(src.read(chunk_sz), dtype='<h')

                        # accumulate all clips in that chunk
                        for k in range(n):
                            s = clip_starts[clip_idx + k] - chunk_s
                            clip = chunk[s:s + template_len]

                            # accumulate extracted spike clip for the relevant unit
                            np.add(per_ch_template, clip, out=per_ch_template)
                        clip_idx += n

                        # get ready for next clip, check for cancellation, and report progress
                        if cancel.is_set():
                            raise Exception("Task cancelled")
                        total_clips_so_far += n
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
                        per_ch_template = template_dict[ch_idx]
                        np.add(per_ch_template, per_ch_clip, out=per_ch_template)
                    clip_idx += 1

                    # get ready for next clip, check for cancellation, and report progress
                    total_clips_so_far += len(ch_indices)
                    if cancel.is_set():
                        raise Exception("Task cancelled")
                    pct_done = int(total_clips_so_far * 100 / total_clips)
                    if pct_done >= next_update_pct:
                        progress_q.put_nowait((unit.uid, pct_done, ""))
                        next_update_pct += 5

        # calculate per-channel templates and SNR, select primary channel (could change from that originally specified).
        best_snr: float = 0
        primary_ch = -1
        for i in ch_indices:
            noise = noise_levels[i]
            template = template_dict[i]
            template = template / len(clip_starts)
            template = template - np.median(template)  # remove DC offset
            snr = 0 if noise <= 0 else (np.max(template) - np.min(template)) / (1.96 * noise)
            if snr > best_snr:
                best_snr = snr
                primary_ch = i
            template_dict[i] = template * work_dir.analog_channel_sample_to_uv(i)

        # cache unit metrics to internal file
        unit.update_metrics(primary_ch, best_snr, template_dict)
        if not work_dir.save_neural_unit_to_cache(unit):
            progress_q.put_nowait((unit.uid, 0,
                                   f"Error occurred while writing unit metrics to internal cache: uid={unit.uid}"))
        else:
            progress_q.put_nowait((unit.uid, 100, ""))   # success!
    except Exception as e:
        progress_q.put_nowait((unit.uid, 0, str(e)))


def identify_unit_primary_channels(dir_path: str, task_id: int, ch_indices: List[int], uids: List[str],
                                   progress_q: Queue, cancel: Event) -> Dict[str, Tuple[int, float]]:
    """
    Estimate SNR for the specified neural units across the specified bank of analog channels, and for each unit return
    the index of the channel for which the unit's SNR was greatest.

    This task function is intended to run in a separate process. The task manager should split up the entire set of
    analog channels into smaller banks, each of which is assigned to a different process.

    NOTES:

    - Use case: When a recording session involves hundreds of channels and hundreds of units, it takes too long to
      compute accurate spike templates for every unit on every channel. Instead, taking advantage of the fact that a
      given unit is typically detectable on only a few channels, we do a quick computation of SNR (template peak-to-peak
      amplitude / noise level on channel) across all units and channels **using a random sampling of a relatively small
      number of clips per unit.** The "primary channel" for each unit is then identified as the analog channel on which
      SNR is greatest. Then more accurate templates (averaging many more spike waveform clips) can be computed only for
      a small number of data channels "in the neighborhood" of the primary channel.
    - Use case: Because the list of units to be processed is specified, this method can be used to process any subset or
      all units defined in the working directory. For example, when a new unit is derived, XSort immediately writes the
      derived unit's spike times to an **incomplete** unit cache file, and eventually spawns a background task to
      find the unit's primary channel, calculate spike templates, and write a complete cache file. To handle this
      scenario, simply specify a list containing the UID of the derived unit(s) to be cached.
    - This method assumes channel noise levels have already been estimated and cached in a dedicated internal cache
      file ('.xs.noise') within the working directory.
    - The analog data is located in individual channel cache files, or in the original prefiltered flat binary file,
      with the channels streams interleaved or not.
    - Extracting one clip at a time can be problematic -- the total number of file seek-and-read operations would be
      #channels * #units * 100 clips/unit (N_TEMPLATE_CLIPS_MIN). With hundreds of channels and units, that is an
      inefficient approach. When the extracting from a prefiltered, interleaved binary file the approach works because
      we have to read in all the channel clips in one go since the streams are interleaved. With a non-interleaved
      binary file or individual cache files, we need to read in a larger multi-clip chunk (~2400KB).

    :param dir_path: Full file system path to the XSort working directory.
    :param task_id: Task ID for progress updates.
    :param ch_indices: The bank of analog channel indices to process.
    :param uids: The UIDs of the neural unists for which a primary channel designation is required.
    :param progress_q: A process-safe queue for delivering progress updates back to the task manager. Each "update" is
        in the form of a 2-tuple: (task_id, pct), where pct is the integer completion percentage; or (task_id, emsg),
        where emsg is an error description. **The task function finishes immediately after delivering an error message
        to the queue.**
    :param cancel: A process-safe event object which is set to request premature cancellation of this task function.
    :return: Dictionary mapping each neural unit's UID to a 2-tuple (ch_idx, snr) containing the index of the channel
        within the range specified) that exhibited the highest SNR for that unit, and the SNR value. Returns an empty
        mapping on failure.
    """
    try:
        emsg, work_dir = WorkingDirectory.load_working_directory(Path(dir_path))
        if work_dir is None:
            raise Exception(emsg)
        emsg, units = work_dir.load_select_neural_units(uids)
        if len(emsg) > 0:
            raise Exception(emsg)
        noise_levels = work_dir.load_channel_noise_from_cache()
        if noise_levels is None:
            raise Exception('Missing or unreadable channel noise cache')
        n_ch = work_dir.num_analog_channels()

        if cancel.is_set():
            raise Exception("Task canceled")

        # we either read directly from the original, prefiltered flat binary source or a set of internal cache files
        interleaved = False
        n_bytes_per_sample = 2
        num_samples_recorded = 0
        if work_dir.need_analog_cache:
            for ch_idx in ch_indices:
                ch_file = Path(work_dir.cache_path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(ch_idx)}")
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
            if work_dir.analog_source.stat().st_size % (2*n_ch) != 0:
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
            for ch_idx in ch_indices:
                template_dict[ch_idx] = np.zeros(template_len, dtype='<f')
                num_clips_dict[ch_idx] = 0
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

        # for tracking overall progress
        next_update_pct = 0
        total_clips = len(ch_indices) * len(clip_starts)
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
                progress_q.put_nowait((task_id, _p))
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

                # accumulate extracted spike clip for the relevant unit
                i_unit = clip_starts[i_clip + _k][1]
                _template = unit_templates[i_unit][i_ch]
                np.add(_template, clip, out=_template)
                unit_num_clips[i_unit][i_ch] += 1
            return n

        # accumulate spike waveform clips across all units from the channel cache files or the flat binary source file.
        n_bytes_per_clip = template_len * n_bytes_per_sample
        if work_dir.need_analog_cache:
            # CASE 1: individual binary cache file for each analog channel stream
            for ch_idx in ch_indices:
                ch_file = Path(work_dir.cache_path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(ch_idx)}")
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
                for ch_idx in ch_indices:
                    # offset to start of contiguous block for next channel to process
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
                    for ch_idx in ch_indices:
                        per_ch_clip = curr_clip[ch_idx]
                        per_ch_template = unit_templates[unit_idx][ch_idx]
                        np.add(per_ch_template, per_ch_clip, out=per_ch_template)
                        unit_num_clips[unit_idx][ch_idx] += 1
                    clip_idx += 1

                    total_clips_so_far += len(ch_indices)
                    next_update_pct = check_progress(next_update_pct)

        # for each unit, find which channel in the range had the highest SNR.
        res: Dict[str, Tuple[int, float]] = dict()
        for i in range(n_units):
            num_clips_dict = unit_num_clips[i]
            best_snr, primary_ch = 0.0, -1
            for ch_idx in ch_indices:
                t: np.ndarray = unit_templates[i][ch_idx]
                if num_clips_dict[ch_idx] > 0:
                    t /= num_clips_dict[ch_idx]
                # if zero noise (in case a channel is dead/connected to ground) SNR is set to 0.
                snr = 0 if noise_levels[ch_idx] <= 0 else (np.max(t) - np.min(t)) / (1.96 * noise_levels[ch_idx])
                if snr > best_snr:
                    best_snr = snr
                    primary_ch = ch_idx
            res[units[i].uid] = (primary_ch, best_snr)

        progress_q.put_nowait((task_id, 100))
        return res
    except Exception as e:
        progress_q.put_nowait((task_id, str(e)))
        return {}


def cache_analog_channels(dir_path: str, task_id: int, ch_indices: List[int], progress_q: Queue, cancel: Event) -> None:
    """
    Extract, filter, and cache each analog data stream in the XSort working directory's analog source file for the bank
    of data channel indices specified, writing each filtered stream to a separate internal cache file in the directory.

    This task function is intended to run in a separate process. The task manager should split up the entire set of
    analog channels to be cached into smaller ranges, each of which is assigned to a different process.

    NOTES:

    - Two major use cases -- interleaved or noninterleaved source. The Omniplex file is a noninterleaved source with a
      complex structure compared to a flat binary noninterleaved source.
    - Prefiltered flat binary source files need not be cached. The method fails if the source is prefiltered. An
      Omniplex file is always cached because its file structure is complex and it typically contains wideband data.
    - Strategy for noninterleaved case: Use a 16-count thread pool and assign each channel to a separate thread. Because
      the source is noninterleaved, each thread processes a different part of the source file. Read in and process in
      1200K chunks. [In testing with the 37GB 385-channel Neuropixel data, using 4 threads and 1200K read chunk sizes
      got the total execution time -- with the 385 channels split into 6 ranges on a 6-core machine -- down to ~55s;
      increasing the number of threads or the read chunk size did not further improve performance.]
    - When I used 5 or more threads in each process for the noninterleaved strategy, the task sometimes fails on a
      Errno 61 - Connection refused socket error. A CPython bug report indicated this is a MacOS-specific bug in
      Python's multiprocessing library, and a backport bug fix is provided for 3.11. Later I realized this was
      happening because I had lots of threads across multiple processes accessing the same process-safe Event object
      to test for user cancel. That required establishing lots of socket connections (in multiprocessing.manager), which
      led to the Errno 61 issues. I changed the implementation to avoid this.
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
    :param task_id: Task ID -- for reporting progress back to the task manager.
    :param ch_indices: Indices of the channels to be cached.
    :param progress_q: A process-safe queue for delivering progress updates. Each update is in the form of a 2-tuple:
        (task_id, int) to indicate progress (second element is % complete) OR (task_id, emsg) to indicate the task has
        failed (second element is the error description). **The task aborts shortly after delivering an error message
        on the progress queue.**
    :param cancel: A process-safe event object which is set to request premature cancellation of this task function.
    """
    try:
        emsg, work_dir = WorkingDirectory.load_working_directory(Path(dir_path))
        if work_dir is None:
            raise Exception(emsg)

        # caching not needed for a prefiltered flat binary source
        if work_dir.is_analog_data_prefiltered:
            raise Exception("Analog data caching not required for a prefiltered source!")

        # for noninterleaved source, we use a small thread pool and cache each channel separately
        if work_dir.uses_omniplex_as_analog_source or not work_dir.is_analog_data_interleaved:
            progress_per_ch: Dict[int, int] = {ch_idx: 0 for ch_idx in ch_indices}
            next_update_pct = 0
            task_func = _cache_pl2_analog_channel if work_dir.uses_omniplex_as_analog_source else \
                _cache_noninterleaved_analog_channel

            with ThreadPoolExecutor(max_workers=min(16, len(ch_indices))) as mt_exec:
                # NOTE: Previously used the multiprocess event object supplied as an argument as the cancel signal for
                # the threads spawned here, but that has caused issues on MacOS. Instead, each process has its own
                # thread-safe cancel event
                thrd_cancel = Event()

                thrd_q = Queue()
                futures: List[Future] = [mt_exec.submit(task_func, work_dir, ch_idx, thrd_q, thrd_cancel)
                                         for ch_idx in ch_indices]

                first_error: Optional[str] = None
                cancelled_already = False
                while 1:
                    trigger_cancel = False
                    try:
                        # progress updates from per-channel thread tasks are: (ch_idx, pct_complete, error_msg)
                        update = thrd_q.get(timeout=0.2)
                        if len(update[2]) > 0:
                            # an error has occurred. If not bc task was cancelled, signal cancel now so that remaining
                            # threads stop. Also cancel any tasks that have not begun!
                            if first_error is None:
                                first_error = f"Error caching channel {update[0]}: {update[2]}"
                            trigger_cancel = True
                        else:
                            progress_per_ch[update[0]] = update[1]
                            total_progress = sum(progress_per_ch.values()) / len(progress_per_ch)
                            if total_progress >= next_update_pct:
                                progress_q.put_nowait((task_id, int(total_progress)))
                                next_update_pct += 5
                    except Empty:
                        pass

                    # if no error and we haven't cancelled, check for cancel from master
                    if not (trigger_cancel or cancelled_already):
                        trigger_cancel = cancel.is_set()

                    # trigger cancel on error or master cancel: Must cancel both running and not yet started tasks
                    # spawned by this process, and inform other processes spawned by master
                    if trigger_cancel and not cancelled_already:
                        thrd_cancel.set()
                        for future in futures:
                            future.cancel()
                        cancel.set()
                        cancelled_already = True  # only need to do this once!

                    if all([future.done() for future in futures]):
                        break

                # report first error encountered, if any
                progress_q.put_nowait((task_id, 100) if first_error is None else (task_id, first_error))
            return

        # HANDLE THE NONINTERLEAVED CASE FROM HERE
        n_ch = work_dir.num_analog_channels()
        samples_per_sec = work_dir.analog_sampling_rate

        # the channel cache files to be written
        cache_file_dict: Dict[int, Path] = \
            {ch_idx: Path(work_dir.cache_path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(ch_idx)}")
             for ch_idx in ch_indices}

        # prepare bandpass filter in case analog signal is wide-band. The filter delays are initialized with zero-vector
        # initial condition and the delays are updated as each block is filtered. SO MAINTAIN A SEPARATE FILTER DELAY
        # FOR EACH CHANNEL.
        [b, a] = scipy.signal.butter(2, [2 * 300 / samples_per_sec, 2 * 8000 / samples_per_sec], btype='bandpass')
        filter_ic = scipy.signal.lfiltic(b, a, np.zeros(max(len(b), len(a)) - 1))
        delay_dict: Dict[int, np.ndarray] = {ch_idx: filter_ic.copy() for ch_idx in ch_indices}

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
                for ch_idx in ch_indices:
                    ch_block = curr_block[ch_idx]
                    ch_block, delay_dict[ch_idx] = scipy.signal.lfilter(b, a, ch_block, axis=-1, zi=delay_dict[ch_idx])
                    ch_block = ch_block.astype(np.int16)
                    with open(cache_file_dict[ch_idx], 'ab') as f:
                        f.write(ch_block.tobytes())

                num_samples_read += n_samples_to_read

                if cancel.is_set():
                    raise Exception("Task cancelled")
                pct = int(num_samples_read * 100 / n_samples)
                if pct >= next_update_pct:
                    progress_q.put_nowait((task_id, pct))
                    next_update_pct += 5

    except Exception as e:
        progress_q.put_nowait((task_id, str(e)))


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
    cache_file = Path(work_dir.cache_path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(idx)}")
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
    cache_file = Path(work_dir.cache_path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(idx)}")
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


def estimate_noise_on_channels_in_range(
        work_dir: WorkingDirectory, start: int, count: int, upd_q: Queue, cancel: Event) -> Dict[int, float]:
    """
    Estimates analog channel noise using 100 random 10-ms clips selected across the entire recording without regard to
    unit spike times, concatenating clips into a long 1D vector V, then: M = np.median(V) and noise =
    1.4826*np.median(np.abs(V-M)).

    This task function is intended to run in a separate thread. The task manager should split up the entire set of
    analog channels into smaller contiguous ranges, each of which is assigned to a different thread.

    NOTE: We conducted tests using 100, 200, 500, 100 and 10000 clips and the results were not significantly different.
    Using spike times to select clips for noise estimation also did not change the results. Performance was very
    drammatically improved by using the algorigthm described above, rather than computing the np.median(nb.abs(clip))
    for every extracted clip, then taking the median of the list of clip medians as the noise level. Also, the algorithm
    described above replicates the Wikipedia description of estimating standard deviation of normally distributed data
    from the median absolute deviation (MAD) of that data.

    :param work_dir: The XSort working directory. If analog channel caching is required, this method fails if any
        channel cache file is missing.
    :param start: Index K of first analog data channel to process.
    :param count: The number of channels to process, N. If K+N > the number of analog channels available, the method
        processes the remaining channels starting at K.
    :param upd_q: A thread-safe queue for delivering progress updates. Each update is in the form of a 3-tuple
        (start, pct, emsg), where start is the index of the first channel in the range handled by this task, pct is the
        percent complete (int), and emsg is an error description if the task has failed (otherwise an empty string).
    :param cancel: A thread-safe event object which is set to request premature cancellation of this task function.
    :return: A dictionary mapping analog channel index to the estimate noise on that channel. Note that noise is in
        raw ADC units, NOT volts. On failure, returns an empty dict.
    """
    try:
        n_ch = work_dir.num_analog_channels()
        if start < 0 or start >= n_ch:
            raise Exception("Invalid channel range")
        elif start + count > n_ch:
            count = n_ch - start

        # we either read directly from the original, prefiltered flat binary source or a set of internal cache files
        interleaved = False
        n_bytes_per_sample = 2
        n_samples = 0
        if work_dir.need_analog_cache:
            for i in range(count):
                ch_file = Path(work_dir.cache_path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(i+start)}")
                if not ch_file.is_file():
                    raise Exception(f"Missing internal analog cache file {ch_file.name}")
                elif ch_file.stat().st_size % n_bytes_per_sample != 0:
                    raise Exception(f"Bad file size for internal analog cache file {ch_file.name}")
                elif n_samples == 0:
                    n_samples = int(ch_file.stat().st_size / n_bytes_per_sample)
                elif n_samples != int(ch_file.stat().st_size / n_bytes_per_sample):
                    raise Exception(f"All internal analog cache files must be the same size!")
        else:
            interleaved = work_dir.is_analog_data_interleaved
            n_bytes_per_sample = 2 * n_ch if interleaved else 2
            if not work_dir.analog_source.is_file():
                raise Exception(f"Original flat binary analog source is missing in working directory.")
            if work_dir.analog_source.stat().st_size % n_bytes_per_sample != 0:
                raise Exception(f"Bad file size for analog source file: {work_dir.analog_source.name}")
            n_samples = int(work_dir.analog_source.stat().st_size / (2*n_ch))

        samples_per_sec = work_dir.analog_sampling_rate
        template_len = int(samples_per_sec * 0.01)

        # for each channel, preallocate a vector V of length = clip length * #clips
        noise_clips: List[np.ndarray] = list()
        n_clips = N_TEMPLATE_CLIPS_MIN
        for _ in range(count):
            noise_clips.append(np.zeros(template_len * n_clips, dtype='<h'))

        # randomly select 50000 clip start times across entire recording and strip off clips cut off at the end
        noise_clip_starts = sorted(random.sample(range(n_samples), 50000))
        while (len(noise_clip_starts) > 0) and (noise_clip_starts[-1] + template_len >= n_samples):
            noise_clip_starts.pop()
        if len(noise_clip_starts) < n_clips:
            raise Exception("Not enough data available to estimate noise on analog channels")

        # from those, randomly select the desired number of clips to process
        chosen = sorted(random.sample(range(len(noise_clip_starts)), n_clips))
        noise_clip_starts = [noise_clip_starts[i] for i in chosen]
        noise_clip_starts.sort()

        # for tracking overall progress
        next_update_pct = 0
        total_clips = count * n_clips

        # extract and put the clips into preallocated vector
        n_bytes_per_clip = template_len * n_bytes_per_sample
        if work_dir.need_analog_cache:
            # CASE 1: Analog data streams cached in individual files
            for i in range(count):
                ch_file = Path(work_dir.cache_path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(i+start)}")
                with open(ch_file, 'rb') as src:
                    clip_idx = 0
                    while clip_idx < n_clips:
                        src.seek(noise_clip_starts[clip_idx] * n_bytes_per_sample)
                        clip = np.frombuffer(src.read(n_bytes_per_clip), dtype='<h')
                        t0 = clip_idx * template_len
                        t1 = t0 + template_len
                        noise_clips[i][t0:t1] = clip
                        clip_idx += 1
                        if cancel.is_set():
                            raise Exception("Task cancelled")
                total_clips_so_far = (i+1) * n_clips
                pct = int(total_clips_so_far * 100 / total_clips)
                if pct >= next_update_pct:
                    upd_q.put_nowait((start, pct, ""))
                    next_update_pct = min(pct + 10, 100)

        elif not interleaved:
            # CASE 2: Prefiltered, non-interleaved flat binary analog source
            with open(work_dir.analog_source, 'rb') as src:
                for i in range(count):
                    ofs = (i + start) * n_samples * n_bytes_per_sample
                    clip_idx = 0
                    while clip_idx < n_clips:
                        src.seek(ofs + noise_clip_starts[clip_idx] * n_bytes_per_sample)
                        clip = np.frombuffer(src.read(n_bytes_per_clip), dtype='<h')
                        t0 = clip_idx * template_len
                        t1 = t0 + template_len
                        noise_clips[i][t0:t1] = clip
                        clip_idx += 1
                        if cancel.is_set():
                            raise Exception("Task cancelled")
                    total_clips_so_far = (i + 1) * n_clips
                    pct = int(total_clips_so_far * 100 / total_clips)
                    if pct >= next_update_pct:
                        upd_q.put_nowait((start, pct, ""))
                        next_update_pct = min(pct + 10, 100)
        else:
            # CASE 3: Prefiltered, interleaved flat binary analog source
            with open(work_dir.analog_source, 'rb') as src:
                total_clips_so_far = 0
                clip_idx = 0
                while clip_idx < n_clips:
                    src.seek(noise_clip_starts[clip_idx] * n_bytes_per_sample)
                    clip = np.frombuffer(src.read(n_bytes_per_clip), dtype='<h')
                    clip = clip.reshape(-1, n_ch).transpose()
                    t0 = clip_idx * template_len
                    t1 = t0 + template_len
                    for i in range(count):
                        ch_idx = i + start
                        noise_clips[i][t0:t1] = clip[ch_idx]
                    clip_idx += 1
                    total_clips_so_far += count
                    if cancel.is_set():
                        raise Exception("Task cancelled")
                    pct = int(total_clips_so_far * 100 / total_clips)
                    if pct >= next_update_pct:
                        upd_q.put_nowait((start, pct, ""))
                        next_update_pct = min(pct + 10, 100)

        # noise computation
        noise_dict: Dict[int, float] = dict()
        for i in range(count):
            ch_idx = i + start
            noise_vec = noise_clips[i]
            noise_dict[ch_idx] = 1.4826 * np.median(np.abs(noise_vec - np.median(noise_vec)))

        upd_q.put_nowait((start, 100, ""))
        return noise_dict
    except Exception as e:
        upd_q.put_nowait((start, 0, str(e)))
        return {}


def compute_statistics(dir_path: str, task_id: int, request: Tuple[Any], progress_q: Queue, cancel: Event) -> None:
    """
    Compute statistics for one or more neural units cached in the specified XSort working directory.

    The method expects to find a unit metrics cache file for each unit listed in the computation request. If any cache
    file is missing, the task aborts.

    This task function is intended to run in a separate process. The task manager assigns each different statistic
    type (ISI, ACG, etc) to a different process. Since the statistics computations are more CPU-bound than file
    IO-bound, this should improve overall performance.

    :param dir_path: Full file system path to the XSort working directory.
    :param task_id: The task_id (for reporting progress back to task manager).
    :param request: A tuple defining the unit statistic(s) requested: (DataType.ISI, uid1, ...) to compute interspike
        interval histogram for one or more distinct units; (DataType.ACG, uid1, ...) to compute autocorrelogram for one
        or more units; (DataType.ACG_VS_RATE, uid1, ...) to compute the 3D autocorrelogram vs instantaneous firing rate
        for one or more units; (DataType.CCG, uid1, uid2, ...) to compute crosscorrelograms for every possible
        pairing of units among a list of 2+ distinct units; (DataType.PCA, uid1[, uid2[, uid3]]) to perform
        principal component analysis on 1-3 distinct units and compute the PCA projections for each unit.
    :param progress_q: A process-safe queue for delivering progress updates and results to the task manager. A progress
        update is a 2-tuple (task_id, pct). An individual statistic is delivered as (DataType, result), where result is
        a tuple packaging the Numpy array(s) and identifying unit UID(s) in the form expected for the particular
        statistic. If the task fails on an error, that is reported as a 2-tuple (task_id, error_msg), after which the
        task function returns.
    :param cancel: A process-safe event object which is set to request premature cancellation of this task function.
        The function returns as soon as the cancellation is detected without taking further action.
    """
    # function passed to report progress and prematurely cancel long-running stats calculations
    def progress(frac_done: float) -> bool:
        """
        Callback function to update progress and check for task cancellation within stats methods
        :param frac_done Fraction of computation completed in [0..1.0].
        :return: True to continue computation, False to cancel.
        """
        if cancel.is_set():
            return False
        _p = int(100.0 * frac_done)
        progress_q.put_nowait((task_id, _p))
        return True

    try:
        emsg, work_dir = WorkingDirectory.load_working_directory(Path(dir_path))
        if work_dir is None:
            raise Exception(emsg)

        # validate request
        ok = isinstance(request, tuple) and (len(request) > 1) and isinstance(request[0], DataType) and \
            all([isinstance(request[i], str) for i in range(1, len(request))])
        if not ok:
            raise Exception("Invalid request")

        which: DataType = request[0]
        uids = [request[i] for i in range(1, len(request))]
        if which == DataType.ISI:
            sub_task_frac = 1.0 / len(uids)
            for k, uid in enumerate(uids):
                u = work_dir.load_neural_unit_from_cache(uid)
                if u is None:
                    raise Exception(f"Missing or invalid unit metrics cache file (uid={uid})")
                if cancel.is_set():
                    return

                out = stats.generate_isi_histogram(u.spike_times, Neuron.FIXED_HIST_SPAN_MS,
                                                   lambda x: progress(x*sub_task_frac))
                if cancel.is_set():
                    return

                max_count = max(out)
                if max_count > 0:
                    out = out * (1.0 / max_count)
                progress_q.put_nowait((which, (uid, out)))
                pct = int((k + 1) * sub_task_frac * 100)
                progress_q.put_nowait((task_id, pct))

        elif which == DataType.ACG:
            sub_task_frac = 1.0 / len(uids)
            for k, uid in enumerate(uids):
                u = work_dir.load_neural_unit_from_cache(uid)
                if u is None:
                    raise Exception(f"Missing or invalid unit metrics cache file (uid={uid})")
                if cancel.is_set():
                    return

                res = stats.generate_cross_correlogram(u.spike_times, u.spike_times, Neuron.FIXED_HIST_SPAN_MS,
                                                       lambda x: progress(x*sub_task_frac))
                if cancel.is_set():
                    return

                out, n = res[0], res[1]
                if n > 0:
                    out = out * (1.0 / n)
                progress_q.put_nowait((which, (uid, out)))
                pct = int((k + 1) * sub_task_frac * 100)
                progress_q.put_nowait((task_id, pct))

        elif which == DataType.ACG_VS_RATE:
            sub_task_frac = 1.0/len(uids)
            for k, uid in enumerate(uids):
                u = work_dir.load_neural_unit_from_cache(uid)
                if u is None:
                    raise Exception(f"Missing or invalid unit metrics cache file (uid={uid})")
                if cancel.is_set():
                    return
                out = stats.gen_cross_correlogram_vs_firing_rate(u.spike_times, u.spike_times,
                                                                 span_ms=Neuron.ACG_VS_RATE_SPAN_MS,
                                                                 progress=lambda x: progress(x*sub_task_frac))
                if cancel.is_set():
                    return
                progress_q.put_nowait((which, (uid, out)))
                pct = int((k + 1) * sub_task_frac * 100)
                progress_q.put_nowait((task_id, pct))

        elif which == DataType.CCG:
            # load all units for which CCGs are to be computed
            units: List[Neuron] = list()
            for uid in set(uids):
                u = work_dir.load_neural_unit_from_cache(uid)
                if u is None:
                    raise Exception(f"Missing or invalid unit metrics cache file (uid={uid})")
                units.append(u)
                if cancel.is_set():
                    return
            if len(units) < 2:
                raise Exception(f"Need at least 2 distinct units to compute crosscorrelograms")
            progress_q.put_nowait((task_id, 10))

            n_ccgs, n_done = len(units) * (len(units) - 1), 0
            sub_task_frac = 0.9 / n_ccgs
            for u in units:
                for u2 in units:
                    if u.uid != u2.uid:
                        res = stats.generate_cross_correlogram(u.spike_times, u2.spike_times, Neuron.FIXED_HIST_SPAN_MS,
                                                               lambda x: progress(x*sub_task_frac + 0.1))
                        if cancel.is_set():
                            return
                        out, n = res[0], res[1]
                        if n > 0:
                            out = out * (1.0 / n)
                        progress_q.put_nowait((which, (u.uid, u2.uid, out)))
                        n_done += 1
                        pct = int(n_done * 90 / n_ccgs) + 10
                        progress_q.put_nowait((task_id, pct))

        elif which == DataType.PCA:
            # load the units included in the analysis -- restricted to 1-3 units
            if (len(uids) != len(set(uids))) or not (1 <= len(uids) <= 3):
                raise Exception(f"PCA projection requires at least one and no more than 3 distinct units")
            units: List[Neuron] = list()
            for uid in set(uids):
                u = work_dir.load_neural_unit_from_cache(uid)
                if u is None:
                    raise Exception(f"Missing or invalid unit metrics cache file (uid={uid})")
                units.append(u)
                if cancel.is_set():
                    return
            progress_q.put_nowait((task_id, 2))

            _compute_pca_projections(work_dir, task_id, units, progress_q, cancel)
        else:
            raise Exception("Invalid request")

    except Exception as e:
        progress_q.put_nowait((task_id, str(e)))


def _compute_pca_projections(
        work_dir: WorkingDirectory, task_id: int, units: List[Neuron], progress_q: Queue, cancel: Event) -> None:
    """
    Helper method for :method:`compute_statistics`. Performs principal component analysis on spike waveform clips across
    a maximum of 16 recorded analog channels for up to 3 neural units. PCA provides a mechanism for detecting
    whether distinct neural units actually represent incorrectly segregated populations of spikes recorded from the same
    unit.

    By design XSort only computes mean spike waveforms -- aka "spike templates" -- on a maximum of 16 channels "near"
    the unit's primary channel -- call this the unit's **primary channel set**. If N<=16 analog channels were recorded,
    then all units will have the same primary channel set. **PCA is restricted to a unit's primary channel set to keep
    the computation time and memory usage reasonable.** This is very important for recording sessions with hundreds of
    analog channels.

    When a single unit is selected for PCA, only the channels in the unit's primary channel set are included in the
    analysis. When 2 or 3 units are selected for PCA, only those channels comprising the **intersection** of the units'
    primary channel sets are considered. **If the intersection is empty, then PCA cannot be  performed**.

    Let N = N1 + N2 + N3 represent the total number of spikes recorded across all K units (we're assuming K=3 here).
    Let the spike clip size be M analog samples long and the number of analog channels included in the analysis be P.
    Then every spike may be represented by an L=MxP vector, the concatenation of the clips for that spike across the P
    channels. The goal of PCA analysis is to reduce this L-dimensional space down to 2, which can then be easily
    visualized as a 2D scatter plot.

    The first step is to compute the principal components for the N samples in L-dimensional space. A great many of
    these clips will be mostly noise -- since, for every spike, we include the clip from each of up to 16 channels, not
    just the primary channel for a given unit. So, instead of using a random sampling of individual clips, we use the
    mean spike template waveforms computed on each channel for each unit. The per-channel spike templates -- which
    should be available already in the :class:`Neuron` instances -- are concatenated to form a KxL matrix, and the
    principal component analysis yields an Lx2 matrix in which the two columns represent the first 2 principal
    components of the data with the greatest variance and therefore most information. **However, if only 1 unit
    is included in the analysis, we revert to using a random sampling of individual clips (because we need at
    least two samples of the L=MxP space in order to compute the covariance matrix).**

    Then, to compute the PCA projection of unit 1 onto the 2D space defined by these two PCs, we form the N1xL
    matrix representing ALL the individual spike clips for that unit, then multiply that by the Lx2 PCA matrix to
    yield the N1x2 projection. Similarly for the other units.

    Progress messages are delivered regularly as the computation proceeds, and the PCA projection for each specified
    unit is delivered as a 2D Numpy array via the supplied process-safe queue. You can think of each row in the array as
    the (x,y) coordinates of each spike in the 2D space defined by the first 2 principal components of the analysis. If
    the number of spikes for a given unit exceeds 20000, the Nx2 PCA projection array is computed and delivered in
    batches of 20000 points -- that way the XSort GUI can display a "partial" PCA scatter plot as the computations
    continue in the background. (NOTE: Total execution time was similar for batch sizes of 10000, 20000 and 40000.)

    All spike clips used in the analysis are 2ms in duration and start 1ms prior to the spike occurrence time.

    :param work_dir: The XSort working directory.
    :param task_id: The task_id (for reporting progress back to task manager).
    :param units: The neural unit(s) for which PCA projection(s) are to be computed
    :param progress_q: A process-safe queue for delivering progress updates and results to the task manager. A progress
        update is a 2-tuple (task_id, pct). An individual statistic is delivered as (DataType, result), where result is
        a tuple packaging the Numpy array(s) and identifying unit UID(s) in the form expected for the particular
        statistic. If the task fails on an error, that is reported as a 2-tuple (task_id, error_msg), after which the
        task function returns.
    :param cancel: A process-safe event object which is set to request premature cancellation of this task function.
        The function returns as soon as the cancellation is detected without taking further action.
    :raises Exception: If any required files are missing from the current working directory, such as the
        analog channel data cache files, or if an IO error or other unexpected failure occurs.
    """
    def progress(frac_done: float) -> bool:
        """
        Callback function to update progress and check for task cancellation within stats methods
        :param frac_done Fraction of computation completed in [0..1.0].
        :return: True to continue computation, False to cancel.
        """
        if cancel.is_set():
            return False
        _p = int(100.0 * frac_done)
        progress_q.put_nowait((task_id, _p))
        return True

    # find the set of analog channels on which to perform the analysis. For multiple units, only perform on channels
    # shared by all units. If no shared units, then PCA cannot be performed.
    channel_set: Set[int] = set(units[0].template_channel_indices)
    for i in range(1, len(units)):
        channel_set = channel_set & set(units[i].template_channel_indices)
    if len(channel_set) == 0:
        return

    channel_list = list(channel_set)
    channel_list.sort()
    samples_per_sec = work_dir.analog_sampling_rate

    # phase 1: compute 2 highest-variance principal components
    all_clips: np.ndarray
    if len(units) == 1:
        # when doing PCA on a single units, chose a randomly selected set of individual multi-clips to calc PC
        u: Neuron = units[0]
        n_clips = min(NCLIPS_FOR_PCA, u.num_spikes)
        clip_dur = int(SPIKE_CLIP_DUR_SEC * samples_per_sec)
        if n_clips == u.num_spikes:
            clip_starts = [int((t - PRE_SPIKE_SEC) * samples_per_sec) for t in u.spike_times]
        else:
            spike_indices = sorted(random.sample(range(u.num_spikes), n_clips))
            clip_starts = [int((u.spike_times[i] - PRE_SPIKE_SEC) * samples_per_sec) for i in spike_indices]
        clip_starts.sort()
        all_clips = _retrieve_multi_clips(work_dir, channel_list, clip_starts, clip_dur, task_id,
                                          pct_start=2, pct_end=20, progress_q=progress_q, cancel=cancel)
    else:
        # when doing PCA on 2-3 units, use spike template waveform clips to form matrix for PCA: Each row is
        # the horizontal concatenation of the spike templates for a unit across all shared analog channels. Each
        # row corresponds to a different unit.
        clip_dur = int(SPIKE_CLIP_DUR_SEC * samples_per_sec)
        all_clips = np.zeros((len(units), clip_dur * len(channel_list)))
        for i_unit, u in enumerate(units):
            for i, ch_idx in enumerate(channel_list):
                all_clips[i_unit, i * clip_dur:(i + 1) * clip_dur] = u.get_template_for_channel(ch_idx)[0:clip_dur]
    if cancel.is_set():
        return
    progress_q.put_nowait((task_id, 10))

    pc_matrix = stats.compute_principal_components(all_clips, progress=lambda x: progress(x*0.1 + 0.1))
    if cancel.is_set():
        return
    progress_q.put_nowait((task_id, 20))

    # phase 2: compute the projection of each unit's spikes onto the 2D space defined by the 2 principal components
    # determine how many spike clips to be extracted across all units in the analysis. We omit clips that are cut off at
    # the beginning or end of the recording.
    total_spike_count = 0
    post_dur = SPIKE_CLIP_DUR_SEC - PRE_SPIKE_SEC
    for u in units:
        n = u.num_spikes
        while (n - 1 >= 0) and (u.spike_times[n-1] + post_dur >= work_dir.analog_channel_recording_duration_seconds):
            n -= 1
        k = 0
        while (k < n) and (u.spike_times[k] - PRE_SPIKE_SEC < 0):
            k += 1
        total_spike_count += n - k

    total_spikes_so_far = 0
    for u in units:
        # again, spike clips cut off at start or end of recording are excluded from the PCA projections
        n = u.num_spikes
        while (n - 1 >= 0) and (u.spike_times[n - 1] + post_dur >= work_dir.analog_channel_recording_duration_seconds):
            n -= 1
        k = 0
        while (k < n) and (u.spike_times[k] - PRE_SPIKE_SEC < 0):
            k += 1

        while k < n:
            n_spikes_in_chunk = min(SPIKES_PER_BATCH, n - k)
            clip_starts = [int((u.spike_times[i + k] - PRE_SPIKE_SEC) * samples_per_sec)
                           for i in range(n_spikes_in_chunk)]
            pct_start = int(20 + 80*total_spikes_so_far/total_spike_count)
            pct_end = int(20 + 80*(total_spikes_so_far + n_spikes_in_chunk)/total_spike_count)
            clips_in_chunk = _retrieve_multi_clips(work_dir, channel_list, clip_starts, clip_dur, task_id,
                                                   pct_start=pct_start, pct_end=pct_end, progress_q=progress_q,
                                                   cancel=cancel)
            if len(clips_in_chunk) == 0:  # an error occurred - proceed no further!
                return
            if cancel.is_set():
                return
            prj_chunk = np.matmul(clips_in_chunk, pc_matrix)   # this can take a significant amount of time!
            if cancel.is_set():
                return

            progress_q.put_nowait((task_id, pct_end))
            progress_q.put_nowait((DataType.PCA, (u.uid, k, prj_chunk)))
            k += len(clips_in_chunk)
            total_spikes_so_far += len(clips_in_chunk)


def _retrieve_multi_clips(work_dir: WorkingDirectory, ch_indices: List[int], clip_starts: List[int], clip_dur: int,
                          task_id: int, pct_start: int, pct_end: int, progress_q: Queue, cancel: Event) -> np.ndarray:
    """
    Helper method for :method:`_compute_pca_projections()'. Let P = the number of analog channels included in the PCA
    analysis, N = number of spike clip start indices specified, and M = per-channel clip duration in #samples. This
    method extracts N clips of duration M from each of the P channels and forms a 2D array of N L=MXP "multi-clips",
    where the i-th row contains the multi-clip corresponding to the i-th clip start index.

    NOTE: A previous incarnation spawned 16 threads to extract the clips from each channel (except when the source
    was a prefiltered, interleaved flat binary file). This was ~3.5x slower than the current solution, which extracts
    the clips one channel at a time ins the current thread.


    :param work_dir:  The XSort working directory.
    :param ch_indices: The indices of P<=16 analog channels included in PCA.
    :param clip_starts: The starting indices of the N clips to extract (in # of samples since start of recording).
    :param clip_dur: The duration of each per-channel clip, M, in # of analog samples.
    :param task_id: The task_id (for reporting progress back to task manager).
    :param pct_start: Percentage complete when this function was called.
    :param pct_end: Percentage complete when this function ends successfully.
    :param progress_q: A process-safe queue for delivering progress updates to the task manager.
    :param cancel: A process-safe event object which is set to request premature cancellation of this task function.
        The function returns as soon as the cancellation is detected without taking further action.
    :return: An NxL Numpy array containing N multi-clips of length L=MxP each. All samples are raw 16-bit ints.
    """
    # strip off any clips that start before or end after the recording
    num_samples = work_dir.analog_channel_recording_duration_samples
    while (len(clip_starts) > 0) and (clip_starts[0] < 0):
        clip_starts.pop(0)
    while (len(clip_starts) > 0) and (clip_starts[-1] + clip_dur >= num_samples):
        clip_starts.pop()
    if len(clip_starts) == 0:
        raise Exception("No spike clips to process!")   # should never happen!

    # with flat binary interleaved source, it's best to retrieve the clips across all relevant channels rather than
    # one channel at a time...
    if (not work_dir.need_analog_cache) and work_dir.is_analog_data_interleaved:
        n_ch = work_dir.num_analog_channels()
        n_clips = len(clip_starts)
        out = np.zeros((n_clips, clip_dur*len(ch_indices)), dtype='<h')
        with open(work_dir.analog_source, 'rb') as src:
            clip_idx = 0
            next_update_pct = 0
            while clip_idx < n_clips:
                # seek to start of next clip to be processed, then read in THAT CLIP ONLY
                sample_idx = clip_starts[clip_idx]
                src.seek(sample_idx * 2 * n_ch)
                curr_clip = np.frombuffer(src.read(clip_dur * 2 * n_ch), dtype='<h')

                # for interleaved source: deinterleave channels and extract clip for target channel
                curr_clip = curr_clip.reshape(-1, n_ch).transpose()

                # form the multi-clip across the specified channels
                multi_clip = np.zeros((clip_dur*len(ch_indices),), dtype='<h')
                for i, idx in enumerate(ch_indices):
                    multi_clip[i*clip_dur:(i+1)*clip_dur] = curr_clip[idx]

                out[clip_idx, :] = multi_clip
                clip_idx += 1

                # check for cancel and report progress
                if cancel.is_set():
                    return np.array([], dtype='<h')
                pct = int(100 * clip_idx / n_clips)
                if pct >= next_update_pct:
                    pct = pct_start + ((pct_end-pct_start) * pct / 100)
                    progress_q.put_nowait((task_id, pct))
                    next_update_pct += 10
        return out

    # otherwise, extract clips for each channel from the relevant source (individual cache file or the original
    # flat binary prefiltered and non-interleaved source), and horiz concatenate to form the multi-clips
    out = np.empty((len(clip_starts), 0), dtype='<h')
    n_done = 0
    for ch_idx in ch_indices:
        per_ch_clips = _extract_channel_clips(work_dir, ch_idx, clip_starts, clip_dur, cancel)
        if cancel.is_set():
            return np.array([], dtype='<h')
        if isinstance(per_ch_clips, str):
            raise Exception(per_ch_clips)
        out = np.hstack((out, per_ch_clips))

        n_done += 1
        pct = int(pct_start + (pct_end-pct_start) * n_done / len(ch_indices))
        progress_q.put_nowait((task_id, pct))

    return out


def _extract_channel_clips(work_dir: WorkingDirectory, ch: int, clip_starts: List[int], clip_dur: int,
                           cancel: Event) -> Union[str, np.ndarray]:
    """
    Helper function for :method:`_retrieve_multi_clips()`. Retrieves a series of short "clips" from a recorded analog
    data channel trace as stored in the relevant source -- either an individual channel cache file or the original,
    prefiltered flat binary analog source. Note that this is not ideal when the analog source interleaved. In that case
    -- because the streams are interleaved, it is more efficient to extract the clips from all relevant channels in
    one pass rather than calling this function once for each channel!

    :param work_dir: The XSort working directory.
    :param ch: The analog channel index.
    :param clip_starts: A Nx1 array containing the start index for each clip; MUST be in ascending order!
    :param clip_dur: The duration of each clip in # of analog samples, M.
    :param cancel: A process-safe event object which is set to request premature cancellation of this task function.
        The function returns as soon as the cancellation is detected without taking further action.
    :return: An NxM Numpy array of 16-bit integers containing the clips, one per row. Returns an empty array if the
        job was cancelled. Returns a brief error description if the operation failed (missing cache file, IO error).
    """
    src_path: Path
    interleaved = False
    n_ch = work_dir.num_analog_channels()
    n_bytes_per_sample = 2
    ch_offset = 0   # byte offset to start of specified channel's stream (non-interleaved flat binary file only)

    try:
        if work_dir.need_analog_cache:
            src_path = Path(work_dir.cache_path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(ch)}")
            if not src_path.is_file():
                raise Exception(f"Channel cache file missing for analog channel {str(ch)}")
        else:
            src_path = work_dir.analog_source
            if not src_path.is_file():
                raise Exception(f"Original flat binary analog source file is missing!")
            file_size = work_dir.analog_source.stat().st_size
            if file_size % (2 * n_ch) != 0:
                raise Exception(f'Flat binary file size inconsistent with specified number of channels ({n_ch})')
            num_samples = int(file_size / (2 * n_ch))
            interleaved = work_dir.is_analog_data_interleaved
            if not interleaved:
                ch_offset = (ch * num_samples) * n_bytes_per_sample
            else:
                # for interleaved case, one "sample" is actually one scan of all N channels
                n_bytes_per_sample = 2 * n_ch

        n_clips = len(clip_starts)
        out = np.zeros((len(clip_starts), clip_dur), dtype='<h')
        if interleaved:
            with open(src_path, 'rb') as src:
                clip_idx = 0
                while clip_idx < n_clips:
                    # seek to start of next clip to be processed, then read in THAT CLIP ONLY
                    sample_idx = clip_starts[clip_idx]
                    src.seek(ch_offset + sample_idx * n_bytes_per_sample)
                    curr_clip = np.frombuffer(src.read(clip_dur*n_bytes_per_sample), dtype='<h')

                    # for interleaved source: deinterleave channels and extract clip for target channel
                    curr_clip = curr_clip.reshape(-1, n_ch).transpose()
                    out[clip_idx, :] = curr_clip[ch]

                    clip_idx += 1
                    if cancel.is_set():
                        return np.array([], dtype='<h')
        else:
            with open(src_path, 'rb') as src:
                clip_idx = 0
                while clip_idx < n_clips:
                    # seek to start of next clip to be processed
                    chunk_s = clip_starts[clip_idx]
                    src.seek(ch_offset + chunk_s * n_bytes_per_sample)

                    # read a chunk of size M <= _READ_CHUNK_SIZE that fully contains 1 or more clips
                    n = 1
                    while ((clip_idx + n < n_clips) and
                           ((clip_starts[clip_idx+n-1] - chunk_s + clip_dur) * n_bytes_per_sample < _READ_CHUNK_SIZE)):
                        n += 1
                    chunk_sz = (clip_starts[clip_idx+n-1] - chunk_s + clip_dur) * n_bytes_per_sample
                    chunk = np.frombuffer(src.read(chunk_sz), dtype='<h')

                    # process all clips in the chunk
                    for k in range(n):
                        start = clip_starts[clip_idx + k] - chunk_s
                        out[clip_idx+k, :] = chunk[start:start + clip_dur]
                    clip_idx += n
                    if cancel.is_set():
                        return np.array([], dtype='<h')

        return out
    except Exception as e:
        return f"Error retrieving clips on channel {ch}: {str(e)}"
