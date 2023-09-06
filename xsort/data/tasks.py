import time
from enum import Enum
from pathlib import Path
from typing import Dict

from PySide6.QtCore import QObject, Slot, Signal, QRunnable


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
        TaskType.DUMMY: 'Test task', TaskType.CACHECHANNELS: 'Extracting and cacheing Omniplex recorded channels...',
        TaskType.CACHEUNIT: 'Cacheing metadata and templates for neural unit ',
        TaskType.GETCHANNELS: 'Retrieving excerpts of Omniplex recorded channel streams...'
    }

    def __init__(self, task_type: TaskType, working_dir: Path):
        super().__init__()

        self._task_type = task_type
        """ The type of background task executed. """
        self._working_dir = working_dir
        """ The working directory in which required data files and internal XSort cache files are located. """
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
