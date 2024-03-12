import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from enum import Enum
# noinspection PyProtectedMember
from multiprocessing.pool import AsyncResult
from pathlib import Path
from queue import Queue, Empty
from threading import Event
from typing import List, Optional, Dict, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal, QRunnable, Slot, QThreadPool
from PySide6.QtWidgets import QProgressDialog

from xsort.data.files import WorkingDirectory, CHANNEL_CACHE_FILE_PREFIX, UNIT_CACHE_FILE_PREFIX
from xsort.data.neuron import Neuron, DataType, ChannelTraceSegment
import xsort.data.taskfunc as tfunc

N_MAX_SPKS_FOR_TEMPLATE: int = 100    # -1 to use all spikes
N_MAX_TEMPLATES_PER_UNIT: int = 16     # we only keep up to this many per-channel templates per neural unit


class TaskType(Enum):
    """ Worker task types. """
    BUILDCACHE = 1,
    """ 
    Process required data sources in XSort working directory and build internal cache of analog channel data 
    streams and neural unit metrics. 
    """
    GETCHANNELS = 2,
    """ Retrieve selected analog channel traces for a specified time period [t0..t1]. """
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


class TaskManager(QObject):
    """
    Manages all running background worker tasks in XSort.
    """

    progress = Signal(str, int)
    """ 
    Signal emitted by a worker task to deliver a progress message and integer completion percentage. Ignore completion
    percentage if not in [0..100].
    """
    data_available = Signal(DataType, object)
    """
    Signal emitted to deliver a data object to the receiver. First argument indicates the type of data retrieved
    (or computed), and the second argument is a container for the data. For the :class:`TaskType`.COMPUTESTATS task,
    the data object is actually the :class:`Neuron` instance in which the computed statistics are cached.
    """
    error = Signal(str)
    """ Signal emitted when the worker task has failed. Argument is an error description. """
    _task_finished = Signal()
    """ Signal emitted when a worker task has finished, succesfully or otherwise. For interal use only. """
    ready = Signal()
    """ Signal emitted whenever a work task has finished and no running background tasks remain. """

    def __init__(self):
        super().__init__()
        self._manager: mp.Manager = mp.Manager()
        """ Interprocess manager context to allow sharing a queue and a cancel event across processes. """
        self._process_pool: mp.Pool = mp.Pool()
        """ A pool of processes for running tasks that are significantly CPU-bound. """
        self._mt_pool_exec = ThreadPoolExecutor(max_workers=32)
        """ Manages a pool of threads for running tasks that are primarily IO bound. """
        self._qthread_pool = QThreadPool()
        """ Qt-managed thread pool for running slow background tasks as QRunnables. """
        self._running_tasks: List[TaskManager._Task] = list()
        """ List of currently running worker tasks. """

        n = os.cpu_count()
        self._num_cpus = 1 if n is None else n
        """ The number of CPUs available according to os.cpu_count(). If that fails, we just assume 1. """

        self.progress.connect(self._on_progress_update)
        self._task_finished.connect(self._on_task_finished)

    def shutdown(self, progress_dlg: QProgressDialog) -> None:
        """
        Cancel any background tasks currently running, then shutdown and release all resources that were used to
        run tasks. This method will block until all tasks have stopped. Upon return, this :class:`TaskManager` is no
        longer usable.

        :param progress_dlg: A modal progress dialog to raise while waiting on cancelled background tasks to finish.
        """
        self.cancel_all_tasks(progress_dlg)
        self._qthread_pool.clear()
        self._mt_pool_exec.shutdown(wait=True, cancel_futures=True)
        self._process_pool.close()
        self._process_pool.join()
        self._manager.shutdown()

    @property
    def busy(self) -> bool:
        """ True if any tasks are currently running in the background. """
        return len(self._running_tasks) > 0

    @property
    def num_cpus(self) -> int:
        """ Number of available CPUs in system according to os.cpu_count(). If that fails, 1 CPU is assumed. """
        return self._num_cpus

    def cancel_all_tasks(self, progress_dlg: QProgressDialog) -> None:
        """
        Cancel all background tasks and BLOCKS waiting for the task to finish.

        Since the task manager lives on the main UI thread but receives signals emitted from background tasks, the main
        UI thread must get CPU time in order to process those signals -- in particular the signal indicating that the
        task has finished. If the UI thread is blocked, then those signals are never received and we'll wait forever
        for the task list to empty. Furthermore, we really don't want any user interactions while waiting for all
        background tasks to stop -- which may take a while.

        To address this issue, this method requires a modal progress dialog that serves both purposes: blocking user
        input while waiting, and spawning a new event loop so that the main GUI thread will still get CPU time to
        process signals from the background tasks.

        :param progress_dlg: The modal progress dialog to raise while waiting on cancelled background tasks to finish.
            This dialog is raised while waiting and closed upon return.
        :return:
        """
        if not self.busy:
            return

        for task in self._running_tasks:
            task.cancel()

        try:
            i = 0
            while self.busy:
                progress_dlg.setValue(i)
                time.sleep(0.05)
                # in the unlikely event it takes more than 5s for task to stop, reset progress to 90%
                i = 90 if i == 99 else i + 1
        finally:
            progress_dlg.close()

    def build_internal_cache_if_necessary(self, work_dir: WorkingDirectory, uids: List[str]) -> bool:
        """
        If necessary, launch a background task to process the analog and unit data source files in the XSort working
        directory specified and generate all missing analog data and neural unit metric cache files within that
        directory.

        Building the internal cache is a time-consuming operation that may take on the order of minutes when the
        analog source includes hundreds of channels.

        :param work_dir: The XSort working directory.
        :param uids: The UIDs of the neural units that should be cached in the directory.
        :return: True if a background task was launched to build out the cache, else False if directory's internal
            cache is already complete.
        """
        need_caching = False
        if work_dir.need_analog_cache:
            for idx in work_dir.analog_channel_indices:
                if not Path(work_dir.path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(idx)}").is_file():
                    need_caching = True
                    break
        if not need_caching:
            for uid in uids:
                if not Path(work_dir.path, f"{UNIT_CACHE_FILE_PREFIX}{uid}").is_file():
                    need_caching = True
                    break

        if not need_caching:
            return False

        self._launch_task(work_dir, TaskType.BUILDCACHE)
        return True

    def get_channel_traces(self, work_dir: WorkingDirectory, first: int, n_ch: int, start: int, n_samples: int) -> None:
        """
        Launch a background task to retrieve analog traces for a specified range of analog data channels over a
        specified time period. The background task delivers each trace as a :class:`ChannelTraceSegment` object via
        the :class:`TaskManager`'s data_available signal.

        :param work_dir: The XSort working directory.
        :param first: Index of first channel to retrieve
        :param n_ch: The number of channels to retrieve, starting at the index specified.
        :param start: Sample index at which traces start
        :param n_samples: Number of samples in each trace.
        """
        task = TaskManager._Task(self, work_dir, TaskType.GETCHANNELS, (first, n_ch, start, n_samples))
        self._running_tasks.append(task)
        self._qthread_pool.start(task)

    def _launch_task(self, work_dir: WorkingDirectory, task_type: TaskType, params: Optional[Tuple] = None) -> None:
        """
        Launch a background task.

        :param work_dir: The XSort working directory on which the task operates.
        :param task_type: The type of background task to perform.
        :param params: A tuple of task-specific arguments. See :class:`TaskManager._Task`.
        """
        task = TaskManager._Task(self, work_dir, task_type, params)
        self._running_tasks.append(task)
        self._qthread_pool.start(task)

    @Slot(str, int)
    def _on_progress_update(self, msg: str, pct: int) -> None:
        pass

    @Slot()
    def _on_task_finished(self) -> None:
        """ Whenever any background task is done, remove ANY finished tasks from the running task list. """
        i = 0
        while i < len(self._running_tasks):
            if self._running_tasks[i].done:
                self._running_tasks.pop(i)
            else:
                i += 1
        if len(self._running_tasks) == 0:
            self.ready.emit()

    class _Task(QRunnable):
        def __init__(self, mgr: 'TaskManager', working_dir: WorkingDirectory, task_type: TaskType,
                     params: Optional[Tuple]):
            """
            Initialize, but do not start, a background task runnoble. The 'params' argument is an optional tuple, the
            contents of which vary with the task.
             - TaskType.BUILDCACHE: None (ignored).
             - TaskType.GETCHANNELS: A 4-tuple of ints (first, n, start, count), where [first, first+n-1] is a
               contiguous range of analog channel indices and [start, start+count-1] is the span of the desired trace
               segments as a contiguous range of sample indices.

            :param mgr: The background task manager.
            :param working_dir: The XSort working directory on which to operate.
            :param task_type: The type of background task to perform
            :param params: A tuple of additional keyword arguments that depend on the type of task performed, as
                described above. Default is None.
            """
            super().__init__()
            self.mgr = mgr
            """ 
            The task will use the task manager's resources to perform the task, using a multiprocessing strategy for a
            CPU-bound task or a multithreading approach for an IO-bound task.
            """
            self.work_dir = working_dir
            """ The XSort working directory in which required source files and internal cache files are located. """
            self.task_type = task_type
            """ The type of background task executed. """
            self.first_ch: int = int(params[0]) if task_type == TaskType.GETCHANNELS else -1
            """ For GETCHANNELS task only, index of the first analog channel to retrieve. """
            self.n_channels: int = int(params[1]) if task_type == TaskType.GETCHANNELS else 0
            """ For GETCHANNELS task only, the numbe of analog channels to retrieve. """
            self.start: int = int(params[2]) if task_type == TaskType.GETCHANNELS else -1
            """ For GETCHANNELS task only, the index of the first analog sample to retrieve. """
            self.count: int = int(params[3]) if task_type == TaskType.GETCHANNELS else 0
            """ For GETCHANNELS task only, the number of analog samples to retrieve. """
            self.cancel_event: Optional[Event] = None
            """ An optional event object used to cancel the task. If None, task is not cancellable. """
            self._done: bool = False
            """ Flag set once task has finished, successfully or otherwise. """

        @Slot()
        def run(self):
            """ Perform the specified task. """
            try:
                if self.task_type == TaskType.BUILDCACHE:
                    self.cache_all_analog_channels()
                    if not self.cancelled:
                        self.cache_all_neural_units()
                elif self.task_type == TaskType.GETCHANNELS:
                    self.get_channel_traces()
                # elif self.task_type == TaskType.COMPUTESTATS:
                #     self.compute_statistics()
                else:
                    raise Exception("Unrecognized request")
            except Exception as e:
                if not self.cancelled:
                    traceback.print_exception(e)
                    self.mgr.error.emit(str(e))
            finally:
                self._done = True
                self.mgr._task_finished.emit()

        def cancel(self) -> None:
            """
            Cancel this task, if possible. **NOTE that the task will not stop until the run() method detects the
            cancellation.**
            """
            if self.cancel_event is not None:
                self.cancel_event.set()

        @property
        def cancelled(self) -> bool:
            """ True if task was cancelled. """
            return (self.cancel_event is not None) and self.cancel_event.is_set()

        @property
        def done(self) -> bool:
            """ True if task has finished, successfully or otherwise. """
            return self._done

        def cache_all_analog_channels(self) -> None:
            """
            Extract the entire data stream for each analog channel in the file designated as the analog data source for
            the current XSort working directory, bandpass filter it if necessary, and store it in a separate cache file
            (flat binary file of 16-bit samples) in the directory.

            Analog stream caching is required if the analog data source is an Omniplex PL2 file, or if it's a flat
            binary file containing raw, unfiltered data.

            The cache files are named ".xs.ch.<idx>", where <idx> is the integer channel index. If the cache file for
            any channel already exists, that channel is NOT cached again.
            """
            if not self.work_dir.need_analog_cache:
                return

            if self.work_dir.uses_omniplex_as_analog_source:
                self.cache_all_analog_channels_pl2()
            elif self.work_dir.is_analog_data_interleaved:
                self.cache_all_analog_channels_interleaved()
            else:
                self.cache_all_analog_channels_noninterleaved()

        def cache_all_analog_channels_pl2(self) -> None:
            """
            Extract, bandpass filter if necessary, and separately cache all wideband (WB) and narrowband (SPKC) analog
            data channels recorded in the Omniplex PL2 file designated as the analog data source for the current XSort
            working directory.

            NOTE: Current MT approach spawns a separate thread for each analog channel to be cached. Given the structure
            of the PL2 file, each thread only reads what it needs to extract the channel data (unlike a flat binary
            interleaved file!). However, if we need to tackle Omniplex files with hundreds of channels recorded, we
            may have to rethink this solution.
            """
            # list of analog channel indices that need to be cached
            ch_indices = [k for k in self.work_dir.analog_channel_indices]

            progress_per_ch: Dict[int, int] = {idx: 0 for idx in ch_indices}
            next_progress_update_pct = 0

            self.cancel_event = Event()
            progress_q = Queue()
            futures: List[Future] = [self.mgr._mt_pool_exec.submit(tfunc.cache_pl2_analog_channel, self.work_dir, idx,
                                                                   progress_q, self.cancel_event)
                                     for idx in ch_indices]

            first_error: Optional[str] = None
            while True:
                try:
                    # progress updates are: (ch_idx, pct_complete, error_msg)
                    update = progress_q.get(timeout=0.2)
                    if len(update[2]) > 0:
                        # an error has occurred. If not bc task was cancelled, signal cancel now so that remaining
                        # threads stop
                        if not self.cancel_event.is_set():
                            self.cancel_event.set()
                            first_error = f"Error caching channel {update[0]}: {update[2]}"
                    else:
                        progress_per_ch[update[0]] = update[1]
                        total_progress = sum(progress_per_ch.values()) / len(progress_per_ch)
                        if total_progress >= next_progress_update_pct:
                            self.mgr.progress.emit(f"Caching {len(ch_indices)} analog channels", int(total_progress))
                            next_progress_update_pct += 5
                except Empty:
                    pass

                if all([future.done() for future in futures]):
                    break

            # on error or cancellation, delete all analog cache files
            if self.cancel_event.is_set():
                tfunc.delete_internal_cache_files(self.work_dir, del_analog=True, del_units=False)

            # report first error encountered, if any
            if first_error is not None:
                self.mgr.error(first_error)

        def cache_all_analog_channels_noninterleaved(self) -> None:
            """
            Extract, bandpass filter, and separately cache all analog data streams in the noninterleaved, flat binary
            file designated as the analog data source for the current XSort working directory.

            NOTE: Current MT approach spawns a separate thread for each analog channel to be cached. For a
            noninterleaved flat binary file, each thread only reads what it needs to extract the channel data. Using a
            32-count thread pool, it took ~68-75s to cache a 385-channel, 37GB Neuropixel binary file with
            noninterleaved channels. With a 7-count pool, ~100s. With a 10-count pool (the default for a machine with
            6 cores), ~75-85s. With a 64-count pool, ~60s (several times my machine crashed while running this test,
            launching from the PyCharm IDE; the kernel panic occurred in the Python process, but I have no idea what the
            root cause is. My machine has been experiencing a lot of kernel panics in other processes as well...)

            Also tried variations on the one-process-per-channel-bank strategy that is currently the best solution for
            the interleaved case. Chose channel bank size so that the number of spawned processes matched the number of
            cores in the system. Variations included: 1 thread per proc, processing each of the channels in the bank
            sequentially); or N threads per proc, where N = # channels in the bank and each thread processed one
            channel. The former was slower than the MT-only approach: 83-88s typical.

            The latter variation, essentially the MT approach distributed across multiple processes, was the fastest
            solution, averaging 55-57s if the thread pool size was 32, and 60-63s if the pool size was 10. However, when
            I repeatedly cleared the cache and repeated the test, subsequent attempts would fail on a "Connection
            refused" IO error -- EVEN after exiting and relaunching the test fixture! I think that underlying file
            IO/socket resources were getting used up and there was a delay in cleaning up old file handles. Eventually a
            subsequent attempt would succeed after several tries. In any case, I decided to stick with the pure MT
            approach for the noninterleaved scenario, as you get good performance without the resource issue.

            NOTE2: This method is the same as cache_all_analog_channels_pl2(), except for the task function used. We
            decided to keep it separate because we may fine-tune the solution for this use case.
            """
            # list of analog channel indices that need to be cached
            ch_indices = [k for k in self.work_dir.analog_channel_indices]

            progress_per_ch: Dict[int, int] = {idx: 0 for idx in ch_indices}
            next_progress_update_pct = 0
            
            self.cancel_event = Event()
            progress_q = Queue()
            futures: List[Future] = \
                [self.mgr._mt_pool_exec.submit(tfunc.cache_noninterleaved_analog_channel, self.work_dir, idx,
                                               progress_q, self.cancel_event) for idx in ch_indices]

            first_error: Optional[str] = None
            while True:
                try:
                    # progress updates are: (ch_idx, pct_complete, error_msg)
                    update = progress_q.get(timeout=0.2)
                    if len(update[2]) > 0:
                        # an error has occurred. If not bc task was cancelled, signal cancel now so that remaining
                        # threads stop
                        if not self.cancel_event.is_set():
                            self.cancel_event.set()
                            first_error = f"Error caching channel {update[0]}: {update[2]}"
                    else:
                        progress_per_ch[update[0]] = update[1]
                        total_progress = sum(progress_per_ch.values()) / len(progress_per_ch)
                        if total_progress >= next_progress_update_pct:
                            self.mgr.progress.emit(f"Caching {len(ch_indices)} analog channels", int(total_progress))
                            next_progress_update_pct += 5
                except Empty:
                    pass

                if all([future.done() for future in futures]):
                    break

            # on error or cancellation, delete all analog cache files
            if self.cancel_event.is_set():
                tfunc.delete_internal_cache_files(self.work_dir, del_analog=True, del_units=False)

            # report first error encountered, if any
            if first_error is not None:
                self.mgr.error(first_error)

        def cache_all_analog_channels_interleaved(self) -> None:
            """
            Extract, bandpass filter, and separately cache all analog data streams in the INTERLEAVED, flat binary file
            designated as the analog data source for the current XSort working directory.

            Because the individual channel streams are interleaved, caching one channel requires reading the entire
            file. If the file contains hundreds of channels, the one-thread-per-channel approach utilized for
            noninterleaved files is extremely wasteful, as every thread reads through the entire source file to
            extract the channel stream. And since every channel stream is filtered (else caching would be unnecessary),
            each thread has both CPU-bound and IO-bound components.

            Approaches that have not worked well thus far (tested on a 37GB binary file with 385 interleaved channels):
             - One-thread-per-channel strategy (32 threads running in the XSort process). Took 130s to cache 32 of 385
               channels; estimate about 1600s to cache all 385.
             - MT pipeline strategy: One "reader thread" consumes the source file, reading reading one second's worth of
               channel data at a time and pushing each channel's one-second block onto one of 10 thread-safe queues.
               Each of ten "writer threads" service one of these queues, popping the next block off the queue and
               writing it to the corresponding channel's cache file. Execution time was highly variable, between 240 and
               660s.
             - One-process-per-channel-bank strategy: Spawn N processes, each of which caches a bank of 20 channels.
               Each process reads the source from beginning to end, extracting, filtering and caching the 20 channels
               for which it is responsible. Execution time was more consistent, typ 200-225s. But this still seems
               wasteful: the 37GB, 385-channel source file is read in its entirety 20x. Also, the test machine had 6
               cores, so that jobs were queued in the process pool until previously started jobs finished -- serializing
               the work to some extent. However, if we choose the bank size so that the # of processes matches the
               # of available CPUs, we get the execution time down to 140-160s typ.
             - MP pipeline strategy: Similar to the MT pipeline, but using one separate "reader process" to digest
               the source and push de-interleaved, filtered channel blocks onto a process-safe queue serviced by 10
               writer threads (spawned by the main XSort process). The idea here was to only read and process the source
               file once, in a separate process running on a different core. In reality, this did not work well: 80-115s
               to process only 20% of the 37GB file, which extrapolate to roughly 400-575s for the entire file. This
               was NOT a good approach because the reader process did all of the CPU-bound work, so that was the
               bottleneck. It would be better to have as many processes as possible digesting raw blocks from one
               reader process.
             - Variants on the one-process-per-channel-bank strategy: The work of the main thread in each process is
               "serialized": read a raw buffer, process that buffer (deinterleave into N streams and filter each), and
               append the N channel blocks to the corresponding cache files. Tried introducing a block queue serviced
               by a pool of M writer threads so that main thread could keep working while the writer threads did the
               appends. Even tried a separate reader thread to read raw blocks sequentially from the source file into
               a "read queue". The queues were size-limited to ensure they did not grow too large. Nevertheless, the
               overall performance of these variants was somewhat worse than the basic serialized approach.

            Thus far, the best strategy has been the one-process-per-channel-bank approach, selecting the number of
            channels per banks so that the number of processes spawned matches the number of available cores. This
            method currently implements that solution.
            """
            if not (self.work_dir.need_analog_cache and self.work_dir.is_analog_data_interleaved):
                return

            n_ch = self.work_dir.num_analog_channels()

            # need a process-safe event to signal cancellation of task, as well as a queue for IP comm
            self.cancel_event = self.mgr._manager.Event()
            progress_q: Queue = self.mgr._manager.Queue()

            # assign banks of N channels to each process, with one process per core
            n_banks = self.mgr.num_cpus
            ch_per_bank = int(0.5 + n_ch / n_banks)

            progress_per_bank: Dict[int, int] = {idx: 0 for idx in range(n_banks)}
            next_progress_update_pct = 0
            dir_path = str(self.work_dir.path.absolute())
            task_args = list()
            for i in range(n_banks):
                task_args.append((dir_path, i*ch_per_bank, ch_per_bank, progress_q, self.cancel_event))

            first_error: Optional[str] = None
            result: AsyncResult = self.mgr._process_pool.starmap_async(
                tfunc.cache_interleaved_analog_channel_bank, task_args, chunksize=1)
            while not result.ready():
                try:
                    update = progress_q.get(timeout=0.5)  # (pct_complete, error_msg)
                    if len(update[2]) > 0:
                        # an error has occurred. If not bc task was cancelled, signal cancel now so that remaining
                        # processes stop, then break out
                        if not self.cancel_event.is_set():
                            self.cancel_event.set()
                            first_error = (f"Error processing interleaved {ch_per_bank}-channel bank "
                                           f"starting at {update[0]}: {update[2]}")
                        break
                    else:
                        bank_idx = int(update[0] / ch_per_bank)
                        progress_per_bank[bank_idx] = update[1]
                        total_progress = sum(progress_per_bank.values()) / len(progress_per_bank)
                        if total_progress >= next_progress_update_pct:
                            self.mgr.progress.emit(f"Caching {n_ch} analog channels", int(total_progress))
                            next_progress_update_pct += 5
                except Empty:
                    pass

            # in case of cancellation or premature termination on an error, make sure all processes have stopped
            while not result.ready():
                time.sleep(0.1)

            # on error or cancellation, delete all analog cache files
            if self.cancel_event.is_set():
                tfunc.delete_internal_cache_files(self.work_dir, del_analog=True, del_units=False)

            if first_error is not None:
                self.mgr.error.emit(first_error)
                return

        def cache_all_neural_units(self) -> None:
            """
            NOTE: THIS SOLUTION IS UNTENABLE WHEN THE NUMBER OF CHANNELS AND THE NUMBER OF UNITS ARE IN THE HUNDREDS.

            Calculate and cache metrics for all neural units found in the XSort working directory's unit data source
            file.

            Calculating the mean spike waveform -- aka, "spike template" -- on each recorded analog data channel
            requires reading and processing each analog data channel cache file (or the original analog source if
            caching is not required) for each unit. The task is both IO- and CPU-bound, and prior testing found that a
            multiprocessing strategy improved performance by roughly the number of cores available. A separate task
            digests each analog channel stream once, calculating the template for each unit on that channel, as well as
            the channel's noise level. The task results are accumulated by this method as they come in, then the
            individual units are updated and the unit cache files written.

            For each unit, up to N_MAX_TEMPLATES_PER_UNIT per-channel templates are kept -- those with the best
            observed SNR. For all other channels, the templates are assumed to be "flat lines".
            """
            emsg, neurons = self.work_dir.load_neural_units()
            if len(emsg) > 0:
                raise Exception(f"{emsg}")

            n_units = len(neurons)
            self.mgr.progress.emit(f"Calculating and caching metrics for {n_units} neural units across "
                                   f"{self.work_dir.num_analog_channels()} analog channels...", 0)

            # accumulate results as they come in. For recordings with hundreds of analog channels, we only keep a
            # limited number of per-channel templates for each unit -- those exhibiting the highest SNRs
            template_snrs_per_unit: List[Dict[int, float]] = list()
            template_dict_per_unit: List[Dict[int, np.ndarray]] = list()
            for i in range(n_units):
                template_dict_per_unit.append(dict())
                template_snrs_per_unit.append(dict())

            # need a process-safe event to signal cancellation of task, as well as a queue for IP comm
            self.cancel_event = self.mgr._manager.Event()
            progress_q: Queue = self.mgr._manager.Queue()

            progress_per_ch: Dict[int, int] = {idx: 0 for idx in self.work_dir.analog_channel_indices}
            next_progress_update_pct = 0
            dir_path = str(self.work_dir.path.absolute())
            task_args = [(dir_path, ch_idx, progress_q, self.cancel_event, N_MAX_SPKS_FOR_TEMPLATE)
                         for ch_idx in self.work_dir.analog_channel_indices]

            first_error: Optional[str] = None
            result: AsyncResult = self.mgr._process_pool.starmap_async(
                tfunc.mp_compute_unit_templates_on_channel, task_args, chunksize=1)
            while not result.ready():
                try:
                    update = progress_q.get(timeout=0.5)  # (ch_idx, pct_complete, error_msg)
                    if len(update[2]) > 0:
                        # an error has occurred. If not bc task was cancelled, signal cancel now so that remaining
                        # processes stop, then break out
                        if not self.cancel_event.is_set():
                            self.cancel_event.set()
                            first_error = f"Error processing channel {update[0]}: {update[2]}"
                        break
                    else:
                        progress_per_ch[update[0]] = update[1]
                        total_progress = sum(progress_per_ch.values()) / len(progress_per_ch)
                        if total_progress >= next_progress_update_pct:
                            self.mgr.progress.emit(f"Caching {n_units} neural units", int(total_progress))
                            next_progress_update_pct += 5
                except Empty:
                    pass

            # in case of cancellation or premature termination on an error, make sure all processes have stopped
            while not result.ready():
                time.sleep(0.1)
            if first_error is not None:
                self.mgr.error.emit(first_error)
                return

            for res in result.get():
                emsg, ch_idx, noise, template_dict = res
                if len(emsg) == 0:
                    for i in range(n_units):
                        template = template_dict[neurons[i].uid]
                        snr = (np.max(template) - np.min(template)) / (1.96 * noise)
                        # we only keep a limited number of templates -- with the highest observed SNRs
                        snr_dict = template_snrs_per_unit[i]
                        snr_dict[ch_idx] = snr
                        template_dict_per_unit[i][ch_idx] = template * self.work_dir.analog_channel_sample_to_uv(ch_idx)
                        if len(snr_dict) > N_MAX_TEMPLATES_PER_UNIT:
                            discard_ch_idx = min(snr_dict.keys(), key=snr_dict.get)
                            snr_dict.pop(discard_ch_idx)
                            template_dict_per_unit[i].pop(discard_ch_idx)
                else:
                    self.mgr.error.emit(emsg)
                    return

            # finally, for each unit, update metrics and save to cache file
            for i in range(n_units):
                u: Neuron = neurons[i]
                snr_dict = template_snrs_per_unit[i]
                primary_ch = max(snr_dict.keys(), key=snr_dict.get)
                u.update_metrics(primary_ch, snr_dict[primary_ch], template_dict_per_unit[i])
                if not self.work_dir.save_neural_unit_to_cache(u):
                    raise Exception(f"Error occurred while writing unit metrics to internal cache: uid={u.uid}")
                # TODO: TESTING
                print(f"Unit {u.uid}: {sorted(snr_dict, key=snr_dict.get, reverse=True)}")

        def get_channel_traces(self) -> None:
            """
            Retrieve the specified trace segment for selected analog data channel streams cached in the working
            directory.
                This method handles the GETCHANNELS task. When channel caching is NOT required (prefiltered flat binary
            source file, the method retrieves the channel trace segments sequentially on the task runnable, because
            we're reading small parts of the same file. When caching is required, the cache files must exist, and
            the method spins up a separate thread to retrieve the trace segment for each channel (this MT strategy was
            about 70x faster than retrieving the segments sequentially on the task runnable).

            :raises Exception: If a channel cache file is missing, or if a file IO error occurs.
            """

            # check/correct task arguments
            total_ch = self.work_dir.num_analog_channels()
            if (self.first_ch < 0) or (self.first_ch >= total_ch):
                raise Exception("Invalid analog channel range!")
            self.n_channels = min(self.n_channels, total_ch - self.first_ch)
            total_samples = self.work_dir.analog_channel_recording_duration_samples
            if (self.start < 0) or (self.start >= total_samples):
                raise Exception("Invalid starting index for channel traces!")
            self.count = min(self.count, total_samples - self.start)

            self.mgr.progress.emit(f"Retrieving {self.n_channels} channel trace segments ...", 0)

            self.cancel_event = Event()

            # special case: If analog data caching isn't required, it's overkill to spin up threads to read very small
            # parts of the analog data source file.
            if not self.work_dir.need_analog_cache:
                for i in range(self.n_channels):
                    idx = self.first_ch + i
                    segment = self.work_dir.retrieve_cached_channel_trace(idx, self.start, self.count)
                    if self.cancel_event.is_set():
                        raise Exception("Operation cancelled")
                    self.mgr.data_available.emit(DataType.CHANNELTRACE, segment)
                return

            # otherwise: spin up a thread to retrieve the trace segment for each of the channels specified. The
            # individual tasks are not cancellable.
            futures: List[Future] = \
                [self.mgr._mt_pool_exec.submit(tfunc.retrieve_trace_from_channel_cache_file, self.work_dir,
                                               i + self.first_ch, self.start, self.count)
                 for i in range(self.n_channels)]
            for future in as_completed(futures):
                res = future.result()
                if isinstance(res, ChannelTraceSegment):
                    self.mgr.data_available.emit(DataType.CHANNELTRACE, res)
                else:  # an error occurred. Cancel any pending tasks and hope any running task finish quickly!
                    for f in futures:
                        f.cancel()
                    raise Exception(res)
