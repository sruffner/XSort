import multiprocessing as mp
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum
# noinspection PyProtectedMember
from multiprocessing.pool import AsyncResult
from pathlib import Path
from queue import Queue, Empty
from threading import Event
from typing import List, Optional, Dict

import numpy as np
from PySide6.QtCore import QObject, Signal, QRunnable, Slot, QThreadPool
from PySide6.QtWidgets import QProgressDialog

from xsort.data.files import WorkingDirectory, CHANNEL_CACHE_FILE_PREFIX, UNIT_CACHE_FILE_PREFIX
from xsort.data.neuron import Neuron, ChannelTraceSegment
from xsort.data.taskfunc import _cache_analog_channel, _mp_compute_unit_templates_on_channel, \
    read_and_filer_interleaved_analog_source, cache_analog_channels_from_queue, mp_cache_interleaved_analog_source, \
    mp2_cache_interleaved_analog_source, service_analog_channel_blocks_queue

N_MAX_SPKS_FOR_TEMPLATE: int = 10000   # -1 to use all spikes


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


class TaskManager(QObject):
    """
    Manages all running background worker tasks in XSort.
    """

    progress = Signal(str, int)
    """ 
    Signal emitted by a worker task to deliver a progress message and integer completion percentage. Ignore completion
    percentage if not in [0..100].
    """
    # data_available = Signal(DataType, object)
    # """
    # Signal emitted to deliver a data object to the receiver. First argument indicates the type of data retrieved
    # (or computed), and the second argument is a container for the data. For the :class:`TaskType`.COMPUTESTATS task,
    # the data object is actually the :class:`Neuron` instance in which the computed statistics are cached.
    # """
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
        self._proc_progress_q: mp.Queue = self._manager.Queue()
        """ A synchronized queue by which tasks running in child processes provide progress updates. """
        self._mt_pool_exec = ThreadPoolExecutor(max_workers=32)
        """ Manages a pool of threads for running tasks that are primarily IO bound. """
        self._thrd_progress_q = Queue()
        """ A thread-safe queue by which tasks running in the thread pool provide progress updates. """
        self._qthread_pool = QThreadPool()
        """ Qt-managed thread pool for running slow background tasks as QRunnables. """
        self._running_tasks: List[TaskManager._Task] = list()
        """ List of currently running worker tasks. """

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
        :return: True if a background task was launched to build out the cache, else False.
        """
        need_caching = False
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

    def _launch_task(self, work_dir: WorkingDirectory, task_type: TaskType, **kwargs) -> None:
        """
        Launch a background task.

        :param work_dir: The XSort working directory on which the task operates.
        :param task_type: The type of background task to perform.
        :param kwargs: Task-specific arguments. See :class:`TaskManager._Task`.
        :return:
        """
        task = TaskManager._Task(self, work_dir, task_type, **kwargs)
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
        def __init__(self, mgr: 'TaskManager', working_dir: WorkingDirectory, task_type: TaskType, **kwargs):
            """
            Initialize, but do not start, a background task runnoble. Additional keywords:

            :param mgr: The background task manager.
            :param working_dir: The XSort working directory on which to operate.
            :param task_type: The type of background task to perform
            :param kwargs: Additional keyword arguments that depend on the type of task performed, as described.
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
            self.start: int = kwargs.get('start', -1)
            """ For the GETCHANNELS task only, the index of the first analog sample to retrieve. """
            self.count: int = kwargs.get('count', 0)
            """ For the GETCHANNELS task only, the number of analog samples to retrieve. """
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
                    # if not self.cancelled:
                    #     self.cache_all_neural_units()
                # elif self.task_type == TaskType.GETCHANNELS:
                #     self.get_channel_traces()
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
            Extract the entire data stream for each analog channel in the XSort working directory's analog data source,
            bandpass filter it if necessary, and store it in a separate cache file (flat binary file of 16-bit samples)
            in the directory.

            Analog stream caching is required if the analog data source is an Omniplex PL2 file, or if it's a flat
            binary file containing raw, unfiltered data. Because this task is primarly IO-bound, it is handled by a pool
            of up to 32 independent threads, with each thread caching one analog channel.

            The cache files are named ".xs.ch.<idx>", where <idx> is the integer channel index. If the cache file for
            any channel already exists, that channel is NOT cached again.
            """
            if not self.work_dir.need_analog_cache:
                return

            # handle interleaved flat binary files differently
            if self.work_dir.is_analog_data_interleaved:
                self.cache_all_analog_channels_interleaved()  # TODO: TESTING
                return

            # list of analog channel indices that need to be cached
            ch_indices = [k for k in self.work_dir.analog_channel_indices if
                          not Path(self.work_dir.path, f"{CHANNEL_CACHE_FILE_PREFIX}{str(k)}").is_file()]
            if len(ch_indices) == 0:
                return

            progress_per_ch: Dict[int, int] = {idx: 0 for idx in ch_indices}
            next_progress_update_pct = 10

            self.cancel_event = Event()
            futures: List[Future] = [self.mgr._mt_pool_exec.submit(_cache_analog_channel, self.work_dir, idx,
                                                                   self.mgr._thrd_progress_q, self.cancel_event)
                                     for idx in ch_indices]

            first_error: Optional[str] = None
            while True:
                try:
                    # progress updates are: (ch_idx, first_sec_trace) or (ch_idx, pct_complete, error_msg)
                    update = self.mgr._thrd_progress_q.get(timeout=0.2)
                    if len(update) == 2:
                        # TODO: Deliver channel trace segment to GUI via data_available signal
                        self.mgr.progress.emit(f"Received channel trace segment on channel {update[0]}: "
                                               f"{isinstance(update[1], ChannelTraceSegment)}", 0)
                    elif len(update[2]) > 0:
                        # an error has occurred. If not bc task was cancelled, signal cancel now so that remaining
                        # threads stop, then break out
                        if not self.cancel_event.is_set():
                            self.cancel_event.set()
                            first_error = f"Error caching channel {update[0]}: {update[2]}"
                    else:
                        progress_per_ch[update[0]] = update[1]
                        total_progress = sum(progress_per_ch.values()) / len(progress_per_ch)
                        if total_progress >= next_progress_update_pct:
                            self.mgr.progress.emit(f"Caching {len(ch_indices)} analog channels", int(total_progress))
                            next_progress_update_pct += 10
                except Empty:
                    pass

                if all([future.done() for future in futures]):
                    break

            # report first error encountered, if any
            if first_error is not None:
                self.mgr.error(first_error)

        def cache_all_analog_channels_interleaved(self) -> None:
            """
            TODO: WORKING HERE - Modifying to use a separate process to read the source and deliver blocks to ONE
                process-safe queue that is serviced by 10 writer threads...

            Thie method handles the case of an unfiltered flat binary analog source file in which the channel streams
            are interleaved. Because it is unfiltered, we bandpass filter each channel's data stream and cache it to a
            separate file. But if there are hundreds of channels, the source file will be very large. In this scenario,
            a MT approach in which we assign a separate thread to cache each channel is untenable: since the channels
            are interleaved, every thread has to read through the entire file -- extremely wasteful.

            Instead, we employ a "pipeline" MT strategy: one thread consumes the source file, reading one second's
            worth of channel data at a time and pushing each channel's one-second block onto one of 10 thread-safe
            queues. Each of ten "writer threads" service one of these queues, popping the next block off the queue
            and writing it to the corresponding channel's cache file.

            Each object in the queues are 2-tuples (ch_idx, block), where ch_idx is the channel index and block is
            the byte buffer containing the next second's worth of that channel's data stream, ready to be appended to
            the corresponding cache file. To divvy up the work, queue K, K=0..9, holds channel data blocks for which
            ch_idx % 10 == K. Each writer thread simply appends each block extracted from its assigned queue to the
            corresponding channel cache file. One writer thread will be responsible for N / 10 cache files, where N
            is the total number of analog data channels.

            NOTE: Method does not check for any existing analog channel cache files. Should remove all such files
            from working directory before running this task, as the writer threads will simply append to an already
            existing file!
            """
            if not (self.work_dir.need_analog_cache and self.work_dir.is_analog_data_interleaved):
                return

            n_ch = self.work_dir.num_analog_channels()
            next_progress_update_pct = 0

            self.cancel_event = self.mgr._manager.Event()
            block_queue = self.mgr._manager.Queue()
            task_args = [(self.work_dir.path, block_queue, self.mgr._proc_progress_q, self.cancel_event)]
            result: AsyncResult = self.mgr._process_pool.starmap_async(mp2_cache_interleaved_analog_source, task_args)
            futures: List[Future] = \
                [self.mgr._mt_pool_exec.submit(service_analog_channel_blocks_queue, self.work_dir, i, block_queue,
                                               self.mgr._proc_progress_q, self.cancel_event) for i in range(10)]

            first_error: Optional[str] = None
            while not result.ready():
                try:
                    # progress updates are: (pct_complete, error_msg)
                    update = self.mgr._proc_progress_q.get(timeout=0.2)
                    if len(update[1]) > 0:
                        # an error has occurred. If not bc task was cancelled, signal cancel now so that remaining
                        # threads stop, then break out
                        if not self.cancel_event.is_set():
                            self.cancel_event.set()
                            first_error = f"Error caching interleaved analog data: {update[1]}"
                    elif update[0] >= next_progress_update_pct:
                        self.mgr.progress.emit(f"Caching {n_ch} interleaved analog channels", int(update[0]))
                        next_progress_update_pct += 5
                except Empty:
                    pass
                except Exception as e:
                    if not self.cancel_event.is_set():
                        self.cancel_event.set()
                        first_error = f"Unexpected error while caching interleaved analog data: {str(e)}"

            # wait for writer threads to finish
            while not all([future.done() for future in futures]):
                time.sleep(0.1)

            # report first error encountered, if any
            if first_error is not None:
                self.mgr.error(first_error)

        def mp_cache_all_analog_channels_interleaved(self) -> None:
            """
            This version of cache_all_analog_channels_interleaved() spawns a separate process for every 20 channels in
            the source file.
            """
            if not (self.work_dir.need_analog_cache and self.work_dir.is_analog_data_interleaved):
                return

            n_ch = self.work_dir.num_analog_channels()

            # need a process-safe event to signal cancellation of task
            self.cancel_event = self.mgr._manager.Event()

            # assign banks of 20 channels to each process
            n_banks = int(n_ch / 20)
            if n_banks * 20 < n_ch:
                n_banks += 1
            progress_per_bank: Dict[int, int] = {idx: 0 for idx in range(n_banks)}
            next_progress_update_pct = 5
            dir_path = str(self.work_dir.path.absolute())
            task_args = list()
            for i in range(n_banks):
                task_args.append((dir_path, i*20, 20, self.mgr._proc_progress_q, self.cancel_event))

            first_error: Optional[str] = None
            result: AsyncResult = self.mgr._process_pool.starmap_async(
                mp_cache_interleaved_analog_source, task_args, chunksize=1)
            while not result.ready():
                try:
                    update = self.mgr._proc_progress_q.get(timeout=0.5)  # (pct_complete, error_msg)
                    if len(update[2]) > 0:
                        # an error has occurred. If not bc task was cancelled, signal cancel now so that remaining
                        # processes stop, then break out
                        if not self.cancel_event.is_set():
                            self.cancel_event.set()
                            first_error = f"Error processing channels {update[0]}-{update[0]+19}: {update[2]}"
                        break
                    else:
                        bank_idx = int(update[0] / 20)
                        progress_per_bank[bank_idx] = update[1]
                        total_progress = sum(progress_per_bank.values()) / len(progress_per_bank)
                        if total_progress >= next_progress_update_pct:
                            self.mgr.progress.emit(f"Caching {n_ch} interleaved analog channels", int(total_progress))
                            next_progress_update_pct += 5
                except Empty:
                    pass

            # in case of cancellation or premature termination on an error, make sure all processes have stopped
            while not result.ready():
                time.sleep(0.1)
            if first_error is not None:
                self.mgr.error.emit(first_error)
                return

        def cache_all_neural_units(self) -> None:
            """
            Calculate and cache metrics for all neural units found in the XSort working directory's unit data source
            file.

            Calculating the mean spike waveform -- aka, "spike template" -- on each recorded analog data channel
            requires reading and processing each analog data channel cache file (or the original analog source if
            caching is not required) for each unit. The task is both IO- and CPU-bound, and prior testing found that a
            multiprocessing strategy improved performance by roughly the number of cores available. A separate task
            digests each analog channel stream once, calculating the template for each unit on that channel, as well as
            the channel's noise level. The task results are accumulated by this method as they come in, then the
            individual units are updated and the unit cache files written.
            """
            emsg, neurons = self.work_dir.load_neural_units()
            if len(emsg) > 0:
                raise Exception(f"{emsg}")

            n_units = len(neurons)
            self.mgr.progress.emit(f"Calculating and caching metrics for {n_units} neural units across "
                                   f"{self.work_dir.num_analog_channels()} analog channels...", 0)

            # accumulate results as they come in
            best_snr_per_unit: List[float] = [0] * n_units
            primary_ch_per_unit: List[int] = [-1] * n_units
            template_dict_per_unit: List[Dict[int, np.ndarray]] = list()
            for i in range(n_units):
                template_dict_per_unit.append(dict())

            # need a process-safe event to signal cancellation of task
            self.cancel_event = self.mgr._manager.Event()

            progress_per_ch: Dict[int, int] = {idx: 0 for idx in self.work_dir.analog_channel_indices}
            next_progress_update_pct = 10
            dir_path = str(self.work_dir.path.absolute())
            task_args = [(dir_path, ch_idx, self.mgr._proc_progress_q, self.cancel_event, N_MAX_SPKS_FOR_TEMPLATE)
                         for ch_idx in self.work_dir.analog_channel_indices]

            first_error: Optional[str] = None
            result: AsyncResult = self.mgr._process_pool.starmap_async(
                _mp_compute_unit_templates_on_channel, task_args, chunksize=1)
            while not result.ready():
                try:
                    update = self.mgr._proc_progress_q.get(timeout=0.5)  # (ch_idx, pct_complete, error_msg)
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
                            next_progress_update_pct += 10
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
                        if snr > best_snr_per_unit[i]:
                            best_snr_per_unit[i] = snr
                            primary_ch_per_unit[i] = ch_idx
                        # convert template units from raw ADC samples to microvolts
                        template_dict_per_unit[i][ch_idx] = template * self.work_dir.analog_channel_sample_to_uv(ch_idx)
                else:
                    self.mgr.error.emit(emsg)
                    return

            # finally, for each unit, update metrics and save to cache file
            for i in range(n_units):
                u: Neuron = neurons[i]
                u.update_metrics(primary_ch_per_unit[i], best_snr_per_unit[i], template_dict_per_unit[i])
                if not self.work_dir.save_neural_unit_to_cache(u):
                    raise Exception(f"Error occurred while writing unit metrics to internal cache: uid={u.uid}")
