import pickle
from pathlib import Path
from typing import Union, Optional, Dict, Any, List

from PySide6.QtCore import QObject, Signal, Slot, QThreadPool

from xsort.data import PL2
from xsort.data.neuron import Neuron
from xsort.data.tasks import Task, TaskType, get_required_data_files, channel_cache_files_exist, unit_cache_file_exists


class Analyzer(QObject):
    """
    TODO: UNDER DEV
    """

    working_directory_changed: Signal = Signal()
    """ Signals that working directory has changed. All views should refresh accordingly. """
    background_task_updated: Signal = Signal(str)
    """ 
    Signals a status update from a IO/analysis job running in the background. Arg (str): status message. If message is 
    empty, then background task has finished. 
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._working_directory: Optional[Path] = None
        """ The current working directory. """
        self._pl2_file: Optional[Path] = None
        """ The Omniplex mulit-channel electrode recording file (PL2). """
        self._pl2_info: Optional[Dict[str, Any]] = None
        """ Metadata extracted from the Omniplex data file. """
        self._channel_map: Optional[Dict[str, int]] = None
        """ 
        Maps Omniplex wideband ("WBnnn") or narrowband channel ("SPKCnnn") to channel index in PL2 file. Only includes 
        channels that were actually recorded.
        """
        self._pkl_file: Optional[Path] = None
        """ The original spike sorter results file (for now, must be a Python Pickle file). """
        self._neurons: List[Neuron] = list()
        """ 
        List of defined neural units. When a valid working directory is set, this will contain information on the neural
        units identified in the original spiker sorter results file located in that directory.
        """
        self._thread_pool = QThreadPool()
        """ Managed thread pool for running slow background tasks. """
    @property
    def working_directory(self) -> Optional[Path]:
        """ The analyzer's current working directory. """
        return self._working_directory

    @property
    def is_valid_working_directory(self) -> bool:
        """ True if analyzer's working directory is set and contains the data files XSort requires. """
        return isinstance(self._working_directory, Path)

    @property
    def neurons(self) -> List[Neuron]:
        """
        A **shallow** copy of the current list of neurons. If the working directory is undefined or otherwise invalid,
        this will be an empty list.
        """
        return self._neurons.copy()

    def change_working_directory(self, p: Union[str, Path]) -> Optional[str]:
        """
        Change the analyzer's current working directory. If the specified directory exists and contains the requisite
        data files, the analyzer will launch a background task to process these files -- and any internal XSort cache
        files already present in the directory -- to prepare the information and data needed for the various XSort
        analysis views. If the candidate directory matches the current working directory, no action is taken.

        :param p: The file system path for the candidate directory.
        :return: An error description if the cancdidate directory does not exist or does not contain the expected data
        files; else None
        """
        _p = Path(p) if isinstance(p, str) else p
        if not isinstance(_p, Path):
            return "Invalid directory path"
        elif _p == self._working_directory:
            return None
        elif not _p.is_dir():
            return "Directory not found"

        # check for required data files. For now, we expect exactly one PL2 and one PKL file
        pl2_file, pkl_file, emsg = get_required_data_files(_p)
        if len(emsg) > 0:
            return emsg

        # load metadata from the PL2 file.
        pl2_info: Dict[str, Any]
        channel_map: Dict[str, int]
        try:
            with open(pl2_file, 'rb') as fp:
                pl2_info = PL2.load_file_information(fp)
                channel_map = dict()
                channel_list = pl2_info['analog_channels']
                for i in range(len(channel_list)):
                    if channel_list[i]['num_values'] > 0:
                        if channel_list[i]['source'] == PL2.PL2_ANALOG_TYPE_WB:
                            channel_map[f"WB{channel_list[i]['channel']}"] = i
                        elif channel_list[i]['source'] == PL2.PL2_ANALOG_TYPE_SPKC:
                            channel_map[f"SPKC{channel_list[i]['channel']}"] = i
        except Exception as e:
            return f"Unable to read Ommniplex (PL2) file: {str(e)}"

        # load neural units (spike train timestamps) from the spike sorter results file (PKL)
        neurons: List[Neuron] = list()
        purkinje_neurons: List[Neuron] = list()  # sublist of Purkinje complex-spike neurons
        try:
            with open(pkl_file, 'rb') as f:
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
            return f"Unable to read spike sorter results from PKL file: {str(e)}"

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

        # success
        self._working_directory = _p
        self._pl2_file = pl2_file
        self._pl2_info = pl2_info
        self._channel_map = channel_map
        self._pkl_file = pkl_file
        self._neurons = neurons

        # signal views
        self.working_directory_changed.emit()

        # if any channel cache files are missing, spawn task to build them and return
        if not channel_cache_files_exist(self._working_directory, self._channel_map):
            task = Task(TaskType.CACHECHANNELS, self._working_directory)
            task.signals.progress.connect(self.on_task_progress)
            task.signals.finished.connect(self.on_task_done)
            self._thread_pool.start(task)
            return None

        # TODO: Otherwise, check for neural unit cache files.
        for unit in neurons:
            if not unit_cache_file_exists(self._working_directory, unit.label):
                task = Task(TaskType.CACHEUNIT, self._working_directory, unit)
                task.signals.progress.connect(self.on_task_progress)
                task.signals.finished.connect(self.on_task_done)
                self._thread_pool.start(task)
                return None

        # TODO: Otherwise, refresh all views
        return None

    @Slot(str, int)
    def on_task_progress(self, desc: str, pct: int) -> None:
        self.background_task_updated.emit(f"{desc} - {pct}%")

    @Slot(bool, object)
    def on_task_done(self, ok: bool, result: object) -> None:
        if ok:
            self.background_task_updated.emit("")
        else:
            self.background_task_updated.emit(str(result))
