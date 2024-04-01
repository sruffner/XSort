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

from PySide6.QtCore import QObject, Signal, QRunnable, Slot, QThreadPool
from PySide6.QtWidgets import QProgressDialog

from xsort.data.files import WorkingDirectory, CHANNEL_CACHE_FILE_PREFIX, UNIT_CACHE_FILE_PREFIX
from xsort.data.neuron import Neuron, DataType, ChannelTraceSegment
import xsort.data.taskfunc as tfunc


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
    TESTFIXTURE = 4
    """ Run a performance test on the XSort working directory (TEMPORARY DURING DEV/TEST). """


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
        self._launch_task(work_dir, TaskType.GETCHANNELS, params=(first, n_ch, start, n_samples))

    def run_performance_test(self, work_dir: WorkingDirectory) -> None:
        """
        Launch a background task to run a performance test on the specified XSort working directory. This is for
        developer use only.
        :param work_dir: The XSort working directory.
        """
        self._launch_task(work_dir, TaskType.TESTFIXTURE)

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
                    self.cache_analog_channels()
                    if not self.cancelled:
                        self.cache_neural_units()
                elif self.task_type == TaskType.GETCHANNELS:
                    self.get_channel_traces()
                # elif self.task_type == TaskType.COMPUTESTATS:
                #     self.compute_statistics()
                elif self.task_type == TaskType.TESTFIXTURE:
                    self.test_fixture()
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

        def cache_analog_channels(self) -> None:
            """
            Extract the entire data stream for each analog channel in the file designated as the analog data source for
            the current XSort working directory, bandpass filter it if necessary, and store it in a separate cache file
            (flat binary file of 16-bit samples) in the directory.

            Analog stream caching is required if the analog data source is an Omniplex PL2 file, or if it's a flat
            binary file containing raw, unfiltered data. No action is taken otherwise.

            The cache files are named ".xs.ch.<idx>", where <idx> is the integer channel index. If the cache file for
            any channel already exists, that channel is NOT cached again.

            Strategy: Implicitly assuming that the host machine has multiple CPUs, this method splits the number of
            analog channels in the source file into N ranges, where N is the number of available CPUS. A separate
            process handles each channel range, hopefully parallelizing the work to some extent.
            """
            if not self.work_dir.need_analog_cache:
                return

            n_ch = self.work_dir.num_analog_channels()

            # need a process-safe event to signal cancellation of task, as well as a queue for IP comm
            self.cancel_event = self.mgr._manager.Event()
            progress_q: Queue = self.mgr._manager.Queue()

            # assign banks of N channels to each process, with one process per core.
            n_banks = self.mgr.num_cpus
            ch_per_bank = int(n_ch/n_banks)
            if n_ch % n_banks != 0:
                ch_per_bank += 1

            progress_per_bank: Dict[int, int] = {i*ch_per_bank: 0 for i in range(n_banks)}
            next_progress_update_pct = 0
            dir_path = str(self.work_dir.path.absolute())
            task_args = list()
            for i in range(n_banks):
                task_args.append((dir_path, i*ch_per_bank, ch_per_bank, progress_q, self.cancel_event))

            first_error: Optional[str] = None
            result: AsyncResult = self.mgr._process_pool.starmap_async(tfunc.cache_analog_channels_in_range, task_args,
                                                                       chunksize=1)
            while not result.ready():
                try:
                    update = progress_q.get(timeout=0.5)  # (first_ch, pct_complete, error_msg)
                    if len(update[2]) > 0:
                        # an error has occurred. If not bc task was cancelled, signal cancel now so that remaining
                        # processes stop
                        if not self.cancel_event.is_set():
                            self.cancel_event.set()
                            first_error = (f"Error caching {ch_per_bank}-channel bank "
                                           f"starting at {update[0]}: {update[2]}")
                    else:
                        progress_per_bank[update[0]] = update[1]
                        total_progress = sum(progress_per_bank.values()) / len(progress_per_bank)
                        if total_progress >= next_progress_update_pct:
                            self.mgr.progress.emit(f"Caching {n_ch} analog channels", int(total_progress))
                            next_progress_update_pct += 5
                except Empty:
                    pass

            # on error or cancellation, delete all analog cache files
            if self.cancel_event.is_set():
                tfunc.delete_internal_cache_files(self.work_dir, del_analog=True, del_units=False)

            if first_error is not None:
                self.mgr.error.emit(first_error)
                return

        # TODO: CONTNUE HERE 3/26 --
        #  Testing: 2021_07_20_Edgar - Done. 30-33 seconds to build internal cache from scratch.
        #  Testing: neuropix_session (interleaved, prefiltered) - ARGH!!! 676 seconds, where the primary channel
        #  identification phase was typ 80s. So still need to improve this! We are reading 100x as many bytes as in
        #  the first phase, but is that why? Should compare with doing 10000 clips per unit and keeping top 16 right
        #  off the bat, rather than 2 phases. Do we really save any time???
        #  Testing: neuropix_session (noninterleaved, prefiltered) - NEXT  BUT need to modify
        #  tfunc._compute_templates_and_cache_metrics_for_unit to read in larger chunks when using a channel cache file
        #  or a non-interleaved analog source, similar to tfunc.identify_unit_primary_channels_in_range....

        def cache_neural_units(self) -> None:
            """
            Calculate and cache metrics for all neural units found in the XSort working directory's unit data source
            file.

            By design, XSort computes and displays the mean spike waveform -- aka, "spike  template" -- on each of up to
            16 analog data channels. Calculating each unit's template on each recorded channel requires reading and
            processing each channel cache file (or the original analog source if caching is not required) for each unit.
            Noise level is estimated on each channel in order to calculate each unit's SNR on that channel. Spike
            templates are maintained for a given unit on the 16 data channels (or fewer if the total number of channels
            recorded is < 16) exhibiting the highest SNRs for that unit; the unit's SNR is the highest observed SNR, and
            its "primary channel" is the data channel with the highest SNR.

            The task is both IO- and CPU-bound, and prior testing found that a multiprocessing strategy improved
            performance by roughly the number of cores available. A separate task digests each analog channel stream
            once, calculating the template for each unit on that channel, as well as the channel's noise level. The task
            results are accumulated across all channels, then the individual units are updated with calculated metrics
            (per-channel templates, primary channel index, and SNR on that channel) and the unit cache files written.
            Calculating a template involves accumulating "clips" at spike times in the unit's spike train, then dividing
            by the number of clips. To save time, only a random sampling of 10000 clips is used, as this gives a decent
            estimate of the spike template; if the unit has N<10000 spikes, then all N clips are used.

            This approach works reasonably well on a multi-CPU machine when the total number of recorded analog channels
            is <= 16. But for a recording session with hundreds of channels and hundreds of neural units, it's way too
            slow. The amount of file IO is roughly #channels * #units * 10000 * (number of bytes in one clip). For this
            use case, we use a 2-step strategy:

            - Phase 1: Do a "quick" estimate of templates across all units and channels using a random sampling of only
              100 clips per unit in order to identify each unit's "primary channel".
            - Phase 2: Do the slower, more accurate estimate of templates for each unit only on the 16 channels "in the
              neighborhood" of the unit's primary channel. Since XSort does not yet support the notion of "probe
              geometry", we use the range of channel indices numerically "around" the primary channel index.
            """
            # load all neural units
            emsg, neurons = self.work_dir.load_neural_units()
            if len(emsg) > 0:
                raise Exception(f"{emsg}")
            if len(neurons) == 0:
                return

            # CASE 1: Not too many analog channels
            if self.work_dir.num_analog_channels() <= tfunc.N_MAX_TEMPLATES_PER_UNIT:
                self._cache_neural_units_all_channels(neurons)
                return

            # otherwise: STAGE 1: Identify each unit's primary channel
            unit_to_primary = self._identify_primary_channels()
            if unit_to_primary is None:
                return
            if not all([unit_to_primary.get(u.uid, -1) >= 0 for u in neurons]):
                raise Exception("Missing primary channel for at least one neural unit in recording session.")
            # STAGE 2: Compute and cache metrics for all units on 16 channels "near" each unit's primary channel
            self._cache_neural_units_select_channels(unit_to_primary)

        def _cache_neural_units_all_channels(self, neurons: List[Neuron]) -> None:
            """
            Helper method for cache_neural_units() handles recording sessions where the total number of analog data
            channels is <= 16.
            :param neurons: The list of all neural units to be cached.
            """
            uids = [u.uid for u in neurons]
            n_units = len(uids)
            self.mgr.progress.emit(f"Calculating and caching metrics for {n_units} neural units across "
                                   f"{self.work_dir.num_analog_channels()} analog channels...", 0)

            # need a process-safe event to signal cancellation of task, as well as a queue for IP comm
            self.cancel_event = self.mgr._manager.Event()
            progress_q: Queue = self.mgr._manager.Queue()

            # divide the units into N banks, where N is number of available CPUS
            n_banks = self.mgr.num_cpus
            if n_units <= n_banks:
                units_per_bank = 1
            else:
                units_per_bank = int(n_units / n_banks)
                if n_units % n_banks != 0:
                    units_per_bank += 1

            progress_per_bank: Dict[int, int] = dict()
            next_progress_update_pct = 0
            dir_path = str(self.work_dir.path.absolute())
            task_args: List[tuple] = list()
            bank_idx = 0
            while bank_idx * units_per_bank < n_units:
                bank_uids = uids[bank_idx*units_per_bank:(bank_idx+1)*units_per_bank]
                task_args.append((dir_path, bank_idx, bank_uids, progress_q, self.cancel_event))
                bank_idx += 1

            first_error: Optional[str] = None
            result: AsyncResult = self.mgr._process_pool.starmap_async(
                tfunc.cache_neural_units_all_channels, task_args, chunksize=1)
            while not result.ready():
                try:
                    update = progress_q.get(timeout=0.2)  # (task_id, pct_complete, error_msg)
                    if len(update[2]) > 0:
                        # an error has occurred. If not bc task was cancelled, signal cancel now so that remaining
                        # processes stop, then break out
                        if not self.cancel_event.is_set():
                            self.cancel_event.set()
                            first_error = f"Error caching neural units in subtask {update[0]}: {update[2]}"
                    else:
                        progress_per_bank[update[0]] = update[1]
                        total_progress = sum(progress_per_bank.values()) / len(progress_per_bank)
                        if total_progress >= next_progress_update_pct:
                            self.mgr.progress.emit(f"Caching {n_units} neural units", int(total_progress))
                            next_progress_update_pct += 5
                except Empty:
                    pass

            if first_error is not None:
                tfunc.delete_internal_cache_files(self.work_dir, del_analog=False, del_units=True)
                self.mgr.error.emit(first_error)
                return

        def _identify_primary_channels(self) -> Optional[Dict[str, int]]:
            """
            Helper method for cache_neural_units() performs a "quick-n-dirty" computation of per-channel spike templates
            and SNRs across all neural units and all analog data channels to identify the "primary channel" (channel
            exhibiting highest SNR) for each unit.

            :return: Dictionary mapping each unit's UID to the index of that unit's primary analog channel. On failure,
                returns None.
            """
            n_ch = self.work_dir.num_analog_channels()
            s = 'interleaved' if self.work_dir.is_analog_data_interleaved else 'non-interleaved'
            self.mgr.progress.emit(f"Idenifying primary channels for all units across {n_ch} analog channels from "
                                   f"{s} source", 0)

            # need a process-safe event to signal cancellation of task, as well as a queue for IP comm
            self.cancel_event = self.mgr._manager.Event()
            progress_q: Queue = self.mgr._manager.Queue()

            # divide the channels into N banks, where N is number of available CPUS
            n_banks = self.mgr.num_cpus
            if n_ch <= n_banks:
                ch_per_bank = 1
                n_banks = n_ch
            else:
                ch_per_bank = int(n_ch / n_banks)
                if n_ch % n_banks != 0:
                    ch_per_bank += 1
                n_banks = int(n_ch / ch_per_bank) + 1

            progress_per_bank: Dict[int, int] = dict()
            next_progress_update_pct = 0
            dir_path = str(self.work_dir.path.absolute())
            task_args: List[tuple] = list()
            for i in range(n_banks):
                task_args.append((dir_path, i * ch_per_bank, ch_per_bank, progress_q, self.cancel_event))
                progress_per_bank[i * ch_per_bank] = 0

            first_error: Optional[str] = None
            result: AsyncResult = self.mgr._process_pool.starmap_async(
                tfunc.identify_unit_primary_channels_in_range, task_args, chunksize=1)
            while not result.ready():
                try:
                    update = progress_q.get(timeout=0.2)  # (start_index, pct_complete, error_msg)
                    if len(update[2]) > 0:
                        # an error has occurred. If not bc task was cancelled, signal cancel now so that remaining
                        # processes stop, then break out
                        if not self.cancel_event.is_set():
                            self.cancel_event.set()
                            first_error = f"Error identifying primary channels in subtask {update[0]}: {update[2]}"
                    else:
                        progress_per_bank[update[0]] = update[1]
                        total_progress = sum(progress_per_bank.values()) / len(progress_per_bank)
                        if total_progress >= next_progress_update_pct:
                            self.mgr.progress.emit(f"Identifying primary channels", int(total_progress))
                            next_progress_update_pct += 5
                except Empty:
                    pass

            if first_error is not None:
                self.mgr.error.emit(first_error)
                return None

            unit_to_primary: Dict[str, int] = dict()
            unit_to_best_snr: Dict[str, float] = dict()
            for res in result.get():
                bank_res: Dict[str, Tuple[int, float]] = res
                for uid, t in bank_res.items():
                    best_snr_so_far: Optional[float] = unit_to_best_snr.get(uid)
                    if (best_snr_so_far is None) or (best_snr_so_far < t[1]):
                        unit_to_primary[uid] = t[0]
                        unit_to_best_snr[uid] = t[1]
            return unit_to_primary

        def _cache_neural_units_select_channels(self, unit_to_primary: Dict[str, int]) -> None:
            """
            Helper method for cache_neural_units() performs computes per-channel spike templates on a select range of
            16 analog data channels near the primary channel for each neural unit, then caches the templates and other
            metrics for each unit. The channel range for a given unit with primary channel P is [P-8 .. P+7], or [0..15]
            if P < 8, or [N-16, N-1] if P >= N-8, where N is the total number of analog channels recorded. The number
            of clips used to compute each unit's spike template on a given channel is max(10000, M), where M is the
            total number of spikes in the unit's spike train.

            This is the second stage in caching neural unit metrics when the number of recorded analog channels exceeds
            16. The first stage does a quick determination of each unit's primary channel using only 100 clips to
            estimate a unit's template on each analog channel. Note that the primary channel ID for a given unit could
            change as the result of using 10000 clips instead of 100 to calculate per-channel templates.

            :param unit_to_primary: Dictionary mapping each unit's UID to the index of that unit's primary analog
                channel.
            """
            uids = [uid for uid in unit_to_primary.keys()]
            n_units = len(uids)
            self.mgr.progress.emit(f"Calculating and caching metrics for {n_units} neural units across "
                                   f"{self.work_dir.num_analog_channels()} analog channels...", 0)

            # need a process-safe event to signal cancellation of task, as well as a queue for IP comm
            self.cancel_event = self.mgr._manager.Event()
            progress_q: Queue = self.mgr._manager.Queue()

            # divide the units into N banks, where N is number of available CPUS
            n_banks = self.mgr.num_cpus
            if n_units <= n_banks:
                units_per_bank = 1
            else:
                units_per_bank = int(n_units / n_banks)
                if n_units % n_banks != 0:
                    units_per_bank += 1

            progress_per_bank: Dict[int, int] = dict()
            next_progress_update_pct = 0
            dir_path = str(self.work_dir.path.absolute())
            task_args: List[tuple] = list()
            bank_idx = 0
            while bank_idx * units_per_bank < n_units:
                bank_uids = uids[bank_idx * units_per_bank:(bank_idx + 1) * units_per_bank]
                uid_2_pc: Dict[str, int] = dict()
                for uid in bank_uids:
                    uid_2_pc[uid] = unit_to_primary[uid]
                task_args.append((dir_path, bank_idx, uid_2_pc, progress_q, self.cancel_event))
                bank_idx += 1

            first_error: Optional[str] = None
            result: AsyncResult = self.mgr._process_pool.starmap_async(
                tfunc.cache_neural_units_select_channels, task_args, chunksize=1)
            while not result.ready():
                try:
                    update = progress_q.get(timeout=0.2)  # (task_id, pct_complete, error_msg)
                    if len(update[2]) > 0:
                        # an error has occurred. If not bc task was cancelled, signal cancel now so that remaining
                        # processes stop, then break out
                        if not self.cancel_event.is_set():
                            self.cancel_event.set()
                            first_error = f"Error caching neural units in subtask {update[0]}: {update[2]}"
                    else:
                        progress_per_bank[update[0]] = update[1]
                        total_progress = sum(progress_per_bank.values()) / len(progress_per_bank)
                        if total_progress >= next_progress_update_pct:
                            self.mgr.progress.emit(f"Caching {n_units} neural units", int(total_progress))
                            next_progress_update_pct += 5
                except Empty:
                    pass

            if first_error is not None:
                tfunc.delete_internal_cache_files(self.work_dir, del_analog=False, del_units=True)
                self.mgr.error.emit(first_error)
                return

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

        def test_fixture(self) -> None:
            """
            Test fixture for assessing performance of various approaches to handling bkg work...
            """
            do_noise = True
            n_clips = 100
            n_units = 482
            n_ch_proc = 385
            n_max_read_chunk_sz = 0

            # need a process-safe event to signal cancellation of task, as well as a queue for IP comm
            self.cancel_event = Event()
            progress_q: Queue = Queue()

            future: Future
            if do_noise:
                future = self.mgr._mt_pool_exec.submit(tfunc.estimate_noise_interleaved_analog_src, self.work_dir,
                                                       progress_q, self.cancel_event)
                self.mgr.progress.emit(f"Estimating noise levels on {self.work_dir.num_analog_channels()} channels", 0)
            elif n_max_read_chunk_sz > 0:
                future = self.mgr._mt_pool_exec.submit(tfunc.read_interleaved_analog_source, self.work_dir,
                                                       n_max_read_chunk_sz, progress_q, self.cancel_event)
                self.mgr.progress.emit(f"Starting read test with max chunk sz = {n_max_read_chunk_sz}", 0)
            else:
                future = self.mgr._mt_pool_exec.submit(tfunc.extract_template_clips_interleaved_analog_src,
                                                       self.work_dir, n_clips, n_units, n_ch_proc, progress_q,
                                                       self.cancel_event)
                self.mgr.progress.emit(f"Starting extract test with n_clips={n_clips}, n_units={n_units}, "
                                       f"n_ch_proc={n_ch_proc}", 0)

            next_progress_update_pct = 0
            first_error: Optional[str] = None
            while not future.done():
                try:
                    update = progress_q.get(timeout=0.2)   # (pct_complete, error_msg)
                    if len(update[1]) > 0:
                        # an error has occurred, which means the single threaded task is stopping/has stopped. If not
                        # because task was cancelled, remember error message
                        if not self.cancel_event.is_set():
                            first_error = f"Test fixture failed: {update[1]}"
                    else:
                        pct = int(update[0])
                        if pct >= next_progress_update_pct:
                            self.mgr.progress.emit(f"Text fixture progress", pct)
                            next_progress_update_pct += 5
                except Empty:
                    pass

            if first_error is not None:
                self.mgr.error.emit(first_error)
