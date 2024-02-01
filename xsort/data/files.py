import csv
import pickle
import struct
import sys
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QComboBox, QLineEdit, QCheckBox, QVBoxLayout, \
    QGroupBox, QGridLayout, QMainWindow, QPushButton, QApplication, QSpacerItem

from xsort.data import PL2
from xsort.data.neuron import Neuron

DIRINFO_FILE: str = '.xs.directory.txt'
"""
Internal directory configuration file persists the filenames of the original analog data and unit data source files,
as well as required parameters if analog data is stored in a flat binary file (.bin or .dat) rather than an Omnniplex
PL2 file. 
"""
CHANNEL_CACHE_FILE_PREFIX: str = '.xs.ch.'
""" Prefix for analog channel data stream cache file -- followed by the channel index. """
UNIT_CACHE_FILE_PREFIX: str = '.xs.unit.'
""" Prefix for a neural unit cache file -- followed by the unit's UID. """


class _QueryDialog(QDialog):
    """
    A customized dialog that queries user for information needed when loading an XSort working directory with
    ambiguous content.
    """
    def __init__(self, folder: Path, analog_input_files: List[Path], unit_input_files: List[Path], parent=None):
        super().__init__(parent)

        self._work_dir_path = folder
        self._analog_input_paths = analog_input_files
        self._unit_input_paths = unit_input_files

        self.setWindowTitle(f"Select input files: {folder.name}")

        msg_label = QLabel("Please select the Omniplex PL2 or flat binary file (.bin, .dat) containing the original \n"
                           "analog data channel streams and the file containing spike trains for each neural unit \n"
                           "'spike-sorted' from the analog data. For a flat binary file source, specify the \n"
                           "configuration parameters needed to parse that file.")

        # the "Input Files" group
        ai_label = QLabel("Analog data")
        self._analog_input_combo = QComboBox()
        self._analog_input_combo.addItems([p.name for p in analog_input_files])
        self._analog_input_combo.setCurrentIndex(0)
        self._analog_input_combo.setEnabled(len(analog_input_files) > 1)
        self._analog_input_combo.currentTextChanged.connect(lambda _: self._refresh())

        ui_label = QLabel("Spike data")
        self._unit_input_combo = QComboBox()
        self._unit_input_combo.addItems([p.name for p in unit_input_files])
        self._unit_input_combo.setCurrentIndex(0)
        self._unit_input_combo.setEnabled(len(unit_input_files) > 1)

        input_file_grp = QGroupBox("Input Files")
        input_file_grp_layout = QGridLayout()
        input_file_grp_layout.addWidget(ai_label, 0, 0)
        input_file_grp_layout.addWidget(self._analog_input_combo, 0, 1)
        input_file_grp_layout.addWidget(ui_label, 1, 0)
        input_file_grp_layout.addWidget(self._unit_input_combo, 1, 1)
        input_file_grp_layout.setColumnStretch(1, 1)
        input_file_grp.setLayout(input_file_grp_layout)

        # the "Flat Binary File Configuration" group
        nchan_label = QLabel("#Analog channels")
        self._nchannels_edit = QLineEdit()
        self._nchannels_edit.setInputMask('09')
        self._nchannels_edit.setText('4')

        rate_label = QLabel("Sampling rate (Hz)")
        self._rate_edit = QLineEdit()
        self._rate_edit.setInputMask('00999')
        self._rate_edit.setText('40000')

        self._interleaved_cb = QCheckBox('Interleaved?')
        self._interleaved_cb.setChecked(False)

        self._prefiltered_cb = QCheckBox('Prefiltered?')
        self._prefiltered_cb.setChecked(False)

        self._warning_label = QLabel("Binary file size in bytes must be a multiple of 2 x #channels!")

        config_grp = QGroupBox("Flat Binary File Configuration")
        config_grp_layout = QGridLayout()
        config_grp_layout.addWidget(nchan_label, 0, 0)
        config_grp_layout.addWidget(self._nchannels_edit, 0, 1)
        config_grp_layout.addWidget(self._interleaved_cb, 0, 2)
        config_grp_layout.addWidget(rate_label, 1, 0)
        config_grp_layout.addWidget(self._rate_edit, 1, 1)
        config_grp_layout.addWidget(self._prefiltered_cb, 1, 2)
        config_grp_layout.addWidget(self._warning_label, 2, 0, 1, 3)
        config_grp.setLayout(config_grp_layout)

        # set initial state of the various widgets
        self._refresh()

        # Ok/Cancel buttons
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttonBox.accepted.connect(self._validate_before_accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        self.layout.addWidget(msg_label)
        self.layout.addSpacing(10)
        self.layout.addWidget(input_file_grp)
        self.layout.addSpacing(10)
        self.layout.addWidget(config_grp)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def _refresh(self) -> None:
        """
        Refresh dialog state whenever user changes the analog data source or the number of analog channels. When the
        analog input source is a flat binary file, the file configuration parameter widgets must be enabled, and the
        warning label made visible IF the number of analog channels is not consistent with the file's size. Otherwise,
        the parameter widges are disabled and the warning label hidden.
        """
        sel_path = self._analog_input_paths[self._analog_input_combo.currentIndex()]
        n_channels = int(self._nchannels_edit.text())
        enable_params = (sel_path.suffix.lower() != '.pl2')
        warn = enable_params and (n_channels > 0) and ((sel_path.stat().st_size % (2 * n_channels)) != 0)

        self._nchannels_edit.setEnabled(enable_params)
        self._rate_edit.setEnabled(enable_params)
        self._interleaved_cb.setEnabled(enable_params)
        self._prefiltered_cb.setEnabled(enable_params)
        if warn:
            if n_channels <= 0:
                self._warning_label.setText("Number of analog channels must be > 0!")
            else:
                self._warning_label.setText(f"Binary file size ({sel_path.stat().st_size} bytes) "
                                            f"must be a multiple of {2 * n_channels}!")
            self._warning_label.setVisible(True)
        else:
            self._warning_label.setVisible(False)

    def _validate_before_accept(self) -> None:
        """
        If a flat binary file is chosen as the analog data source, verify that the file size is consistent with
        the number of analog channels specifed. If so, or if a PL2 file is the source, then extinguish dialog,
        accepting the user entries.
        """
        self._refresh()
        if not self._warning_label.isVisible():
            self.accept()

    @property
    def analog_data_source(self) -> Path:
        """ Path of analog data source selected by user. """
        return self._analog_input_paths[self._analog_input_combo.currentIndex()]

    @property
    def unit_data_source(self) -> Path:
        """ Path of neural unit data source selected by user. """
        return self._unit_input_paths[self._unit_input_combo.currentIndex()]

    @property
    def num_analog_channels(self) -> int:
        """ Number of analog data channels. Applicable only if analog data source is a flat binary file. """
        return int(self._nchannels_edit.text())

    @property
    def sampling_rate(self) -> int:
        """ Analog data sampling rate in Hz. Applicable only if analog data source is a flat binary file. """
        return int(self._rate_edit.text())

    @property
    def interleaved(self) -> bool:
        """ Is analog data interleaved? Applicable only if analog data source is a flat binary file. """
        return self._interleaved_cb.isChecked()

    @property
    def prefiltered(self) -> bool:
        """ Is analog data prefiltered? Applicable only if analog data source is a flat binary file. """
        return self._prefiltered_cb.isChecked()


class WorkingDirectory:
    """
    Encapsulation of an XSort "working directory", which includes the original source files containing recorded analog
    data channel streams and spike-sorted neural units, as well as internal configuration and cache files generated by
    XSort itself.
    """
    def __init__(self, folder: Path, analog_src: str, unit_src: str, rate: int = 0, num_channels: int = 0,
                 interleaved: bool = False, prefiltered: bool = False):
        self._folder = folder
        """ The working directory file system path. """
        self._analog_src = analog_src
        """ Name of analog data channel stream source file within the working directory. """
        self._unit_src = unit_src
        """ Name of spike-sorted neural unit data source file within the working directory. """
        self._sampling_rate: int = rate
        """ Sampling rate in Hz for analog data streams in flat binary file. Ignored for PL2 source file. """
        self._num_channels: int = num_channels
        """ Number of analog data channels stored in flat binary file. Ignored for Pl2 source file. """
        self._interleaved: bool = interleaved
        """ Are analog data channel samples interleaved in flat binary file? Ignored for PL2 source file. """
        self._prefiltered: bool = prefiltered
        """ Is analog data in flat binary file prefiltered? Ignored for PL2 source file."""
        self._pl2_info: Optional[Dict[str, Any]] = None
        """ 
        Metadata extracted from the Omniplex PL2 file if that is the analog data source. Ignored if the analog data 
        source is a flat binary file.
        """
        self._neurons: List[Neuron] = list()
        """ List of neural units extracted from the unit data source file. Cached temporarily at construction time. """
        self._error: Optional[str] = self._validate()
        """ None if directory content is valid; else a brief description of first error encountered. """

    def _validate(self) -> Optional[str]:
        """
        Validate the working directory's configuration, as follows:
         - The analog data source file must exist. If it is an Omniplex PL2 file, metadata is read from the file as a
           check that the file is valid. If it is a flat binary file, the file size must be consistent with the number
           of analog channels and the sampling rate specified by user (all samples assumed to be int16).
         - The spike-sorted neural units source file must exist. The list of neural units is loaded from the file to
           verify its correctness.

        :return: A brief error description if a problem is detected, else None.
        """
        if not (isinstance(self._folder, Path) and self._folder.is_dir()):
            return "Invalid or nonexistent working directory"
        if not (isinstance(self._analog_src, str) and self.analog_source.is_file()):
            return "Missing analog data source file"
        if not (isinstance(self._unit_src, str) and self.unit_source.is_file()):
            return "Missing neural unit data source file"

        if self.analog_source.suffix.lower() == '.pl2':
            try:
                with open(self.analog_source, 'rb') as fp:
                    self._pl2_info = PL2.load_file_information(fp)
            except Exception as e:
                return f"Unable to read Ommniplex (PL2) file: {str(e)}"
        elif (self._num_channels <= 0) or ((self.analog_source.stat().st_size % (self._num_channels * 2)) != 0):
            return "Flat binary analog data source file is not consistent with # of channels specified"

        emsg, units = self.load_neural_units()
        if len(emsg) > 0:
            return emsg
        self._neurons = units

        return None

    @staticmethod
    def load_working_directory(folder: Path, window: Optional[QMainWindow] = None) \
            -> Tuple[str, Optional['WorkingDirectory']]:
        """
        Load the specified folder as an XSort working directory, verifying that it contains the required data source
        files and, if applicable, configuration information for a flat binary analog data source.

        If the specified working directory contains multiple possible analog and/or neural unit data source files, or
        if the analog data source is a flat binary file, a modal dialog is raised to collect this information from
        the user. The user may cancel out of this dialog.

        :param folder: File system path for a putative XSort working directory.
        :param window: Main application window (to serve as parent for modal dialog, if needed).
        :return: A 2-tuple **(emsg, W)**, where **W** is a :class:`WorkingDirectory` object encapsulating the working
            directory content. On failure, **W** is None and **emsg** is a brief description of the error encountered.
            If the user cancels out of the modal dialog, **W** is None and **emsg** is an empty string.
        """
        if not (isinstance(folder, Path) and folder.is_dir()):
            return "Invalid or non-existent working directory", None
        cfg, analog_sources, unit_sources = WorkingDirectory._list_working_directory_files(folder)
        if len(analog_sources) == 0:
            return "No analog data source files (.pl2, .bin, .dat) found", None
        if len(unit_sources) == 0:
            return "No neural unit data source files (.pkl, .pickle) found", None

        # if internal directory configuration file exists, use that unless its content is no longer valid, in which
        # case we delete it.
        if isinstance(cfg, Path):
            work_dir = WorkingDirectory._from_config_file(cfg)
            if (work_dir is None) or not work_dir.is_valid:
                cfg.unlink(missing_ok=True)
            else:
                return "", work_dir

        if len(analog_sources) == 1 and (analog_sources[0].suffix.lower() == '.pl2') and (len(unit_sources) == 1):
            work_dir = WorkingDirectory(folder, analog_sources[0].name, unit_sources[0].name)
            if work_dir.is_valid:
                work_dir._to_cfg_file()
                return "", work_dir
            else:
                return work_dir.error_description, None

        # if we get here, we need user input
        dlg = _QueryDialog(folder, analog_sources, unit_sources, parent=window)
        dlg.exec()
        if dlg.result() == QDialog.DialogCode.Accepted:
            work_dir = WorkingDirectory(folder, dlg.analog_data_source.name, dlg.unit_data_source.name,
                                        dlg.sampling_rate, dlg.num_analog_channels, dlg.interleaved, dlg.prefiltered)
            if work_dir.is_valid:
                work_dir._to_cfg_file()
                return "", work_dir
            else:
                return work_dir.error_description, None
        else:
            return "", None   # user cancelled

    @staticmethod
    def _list_working_directory_files(folder: Path) -> Tuple[Optional[Path], List[Path], List[Path]]:
        """
        Scan the specified folder for any and all data source files recognized by XSort:
         - Omniplex PL2 files (.pl2) containing multi-channel electrode recordings in a proprietary format.
         - Flat binary files (.bin or .dat extensions) containing analog channel data streams (int16 samples).
         - Python Pickle files containing "neural units" extracted from analog data via spike-sorting applications.
         - An internal XSort configuration file listing the **single** analog data source file and **single** unit data
           source file, along with additional configuration parameters needed to parse individual analog data channel
           streams from a flat binary file (if applicable). This file is written after XSort has successfully "opened"
           a working directory and identified the required source files.

        :param folder: File system path for a putative XSort working directory.
        :return: A 3-tuple **(cfg_file, analog_src_files, unit_src_files)**, where **cfg_file** is the XSort
            configuration file described (or None if not found), **analog_src_files** is the list of all PL2 or flat
            binary files found that could serve as the analog data source, and **unit_src_files** is the list of all
            pickle files found that could serve as the source of spike-sorted neural units.
        """
        if not (isinstance(folder, Path) and folder.is_dir()):
            return None, [], []
        cfg_path: Optional[Path] = None
        analog_data: List[Path] = list()
        unit_data: List[Path] = list()
        for child in folder.iterdir():
            if child.is_file():
                ext = child.suffix.lower()
                if ext in ['.pkl', '.pickle']:
                    unit_data.append(child)
                elif ext in ['.pl2', '.bin', '.dat']:
                    analog_data.append(child)
                elif child.name == DIRINFO_FILE:
                    cfg_path = child
        return cfg_path, analog_data, unit_data

    @staticmethod
    def _from_config_file(cfg_path: Path) -> Optional['WorkingDirectory']:
        """
        Load working directory information from the internal configuration file specified. The configuration file is
        a single-line text file with at least 2 comma-separated string token, in order:
         - The source file name for the recorded analog data channel streams (.pl2, .bin, or .dat).
         - The source file name for the spike-sorted neural units (.pkl or .pickle).
        If the analog source is a flat binary file (.bin or .dat), then there are an additional 4 string tokens:
         - Sampling rate in Hz (integer string).
         - Number of analog channels (integer string).
         - Whether or not analog channel data is interleaved ("true"/"false").
         - Whether or not analog channel data is prefiltered ("true"/"false").

        :param cfg_path: File system path for the internal XSort working directory configuration file.
        :return: Working directory configuration or None if operation fails
        """
        try:
            with open(cfg_path, 'r', newline='') as f:
                reader = csv.reader(f)
                for line in reader:
                    if len(line) == 2:
                        return WorkingDirectory(cfg_path.parent, line[0], line[1])
                    elif len(line) == 6:
                        rate, n_channels = int(line[2]), int(line[3])
                        interleaved, prefiltered = (line[4] == "true"), (line[5] == "true")
                        return WorkingDirectory(cfg_path.parent, line[0], line[1], rate, n_channels,
                                                interleaved, prefiltered)
                    else:
                        return None
        except Exception:
            return None

    def _to_cfg_file(self) -> None:
        """
        If this is a valid XSort working directory, store analog and unit data source file names and any
        additional configuration parameters in the internal configuration file within the directory.
        """
        if self.is_valid:
            cfg = Path(self._folder, DIRINFO_FILE)
            try:
                with open(cfg, 'w', newline='') as f:
                    writer = csv.writer(f)
                    if self.uses_omniplex_as_analog_source:
                        writer.writerow((self._analog_src, self._unit_src))
                    else:
                        writer.writerow((self._analog_src, self._unit_src, str(self._sampling_rate),
                                         str(self._num_channels), "true" if self._interleaved else "false",
                                         "true" if self._prefiltered else "false"))
            except Exception:
                cfg.unlink(missing_ok=True)

    @property
    def error_description(self) -> Optional[str]:
        """ Description of error encountered if working directory is not valid, else None. """
        return self._error

    @property
    def is_valid(self) -> bool:
        """ Is the working directory content/configuration valid? """
        return self._error is None

    @property
    def path(self) -> Path:
        """ File system path for the working directory. """
        return self._folder

    @property
    def uses_omniplex_as_analog_source(self) -> bool:
        """ True if the analog data source is an Omniplex PL2 recording. """
        return self.is_valid and (self.analog_source.suffix.lower() == '.pl2')

    @property
    def analog_source(self) -> Path:
        """ File system path for the original analog channel data source file in working directory. """
        return Path(self._folder, self._analog_src)

    @property
    def analog_channel_indices(self) -> List[int]:
        """ List of analog channel indices on which electrophysiological data was recorded. """
        if self._pl2_info is None:
            return [k for k in range(self._num_channels)]
        else:
            out: List[int] = list()
            channel_list = self._pl2_info['analog_channels']
            for i in range(len(channel_list)):
                if channel_list[i]['num_values'] > 0:
                    if channel_list[i]['source'] in [PL2.PL2_ANALOG_TYPE_WB, PL2.PL2_ANALOG_TYPE_SPKC]:
                        out.append(i)
            return out

    @property
    def unit_source(self) -> Path:
        """ File system path for the original spike-sorted neural unit data source file in working directory. """
        return Path(self._folder, self._unit_src)

    def load_neural_units(self) -> Tuple[str, Optional[List[Neuron]]]:
        """
        Load all neural units stored in the unit data source file in this working directory. This Python pickle file
        contains the results of spike sorting as a list of Python dictionaries, where each dictionary represents one
        neural unit.
        :return: On success, a tuple ("", L), where L is a list of :class:`Neuron` objects encapsulating the neural
            units found in the file. Certain derived unit metrics -- mean spike waveforms, SNR, primary analog channel
            -- will be undefined. On failure, returns ('error description', None).
        """
        # when the working directory content is validated, the pickle file is read and units are loaded and cached
        # until requested. Once retrieved, further invocations of the method will reload the pickle file...
        if len(self._neurons) > 0:
            out = [n for n in self._neurons]
            self._neurons.clear()
            return "", out

        neurons: List[Neuron] = list()
        purkinje_neurons: List[Neuron] = list()  # sublist of Purkinje complex-spike neurons
        try:
            with open(self.unit_source, 'rb') as f:
                res = pickle.load(f)
                ok = isinstance(res, list) and all([isinstance(k, dict) for k in res])
                if not ok:
                    raise Exception("Unexpected content found")
                for i, u in enumerate(res):
                    if u['type__'] == 'PurkinjeCell':
                        neurons.append(Neuron(i + 1, u['spike_indices__'] / u['sampling_rate__'], suffix='s'))
                        neurons.append(Neuron(i + 1, u['cs_spike_indices__'] / u['sampling_rate__'], suffix='c'))
                        purkinje_neurons.append(neurons[-1])
                    else:
                        neurons.append(Neuron(i + 1, u['spike_indices__'] / u['sampling_rate__']))
        except Exception as e:
            return f"Unable to read neural units from PKL file: {str(e)}", None

        # the spike sorter algorithm generating the PKL file copies the 'cs_spike_indices__' of a 'PurkinjeCell' type
        # into the 'spike_indices__' of a separate 'Neuron' type. Above, we split the simple and complex spike trains of
        # the sorter's 'PurkinjeCell' into two neural units. We need to remove the 'Neuron' records that duplicate any
        # 'PurkinjeCell' complex spike trains...
        removal_list: List[int] = list()
        for purkinje in purkinje_neurons:
            n: Neuron
            for i, n in enumerate(neurons):
                if (not n.is_purkinje) and purkinje.matching_spike_trains(n):
                    removal_list.append(i)
                    break
        for idx in sorted(removal_list, reverse=True):
            neurons.pop(idx)

        return "", neurons


def get_required_data_files(folder: Path) -> Tuple[Optional[Path], Optional[Path], str]:
    """
    Scan the specified folder for the data source files required by XSort: an Omniplex PL2 file containing the
    multi-channel electrode recording, and a Python pickle file containing the original spike sorter results. Enforce
    the requirement that the folder contain only ONE file of each type.

    :param folder: File system path for the current XSort working directory.
    :return: On success, returns (pl2_file, pkl_file, ""), where the first two elements are the paths of the PL2 and
    pickle files, respectively. On failure, returns (None, None, emsg) -- where the last element is a brief description
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


def load_spike_sorter_results(sorter_file: Path) -> Tuple[str, Optional[List[Neuron]]]:
    """
    Load the contents of the spike sorter results file specified. **This method strictly applies to the spike sorter
    program utilized in the Lisberger lab, which outputs the sorting algorithm's results to a Python pickle file.**

    :param sorter_file: File system path for the spike sorter results file.
    :return: On success, a tuple ("", L), where L is a list of **Neuron** objects encapsulating the neural units found
    in the file. Certain derived unit metrics -- mean spike waveforms, SNR, primary analog channel -- will be undefined.
    On failure, returns ('error description', None).
    """
    neurons: List[Neuron] = list()
    purkinje_neurons: List[Neuron] = list()  # sublist of Purkinje complex-spike neurons
    try:
        with open(sorter_file, 'rb') as f:
            res = pickle.load(f)
            ok = isinstance(res, list) and all([isinstance(k, dict) for k in res])
            if not ok:
                raise Exception("Unexpected content found")
            for i, u in enumerate(res):
                if u['type__'] == 'PurkinjeCell':
                    neurons.append(Neuron(i + 1, u['spike_indices__'] / u['sampling_rate__'], suffix='s'))
                    neurons.append(Neuron(i + 1, u['cs_spike_indices__'] / u['sampling_rate__'], suffix='c'))
                    purkinje_neurons.append(neurons[-1])
                else:
                    neurons.append(Neuron(i + 1, u['spike_indices__'] / u['sampling_rate__']))
    except Exception as e:
        return f"Unable to read spike sorter results from PKL file: {str(e)}", None

    # the spike sorter algorithm generating the PKL file copies the 'cs_spike_indices__' of a 'PurkinjeCell' type
    # into the 'spike_indices__' of a separate 'Neuron' type. Above, we split the simple and complex spike trains of
    # the sorter's 'PurkinjeCell' into two neural units. We need to remove the 'Neuron' records that duplicate any
    # 'PurkinjeCell' complex spike trains...
    removal_list: List[int] = list()
    for purkinje in purkinje_neurons:
        n: Neuron
        for i, n in enumerate(neurons):
            if (not n.is_purkinje) and purkinje.matching_spike_trains(n):
                removal_list.append(i)
                break
    for idx in sorted(removal_list, reverse=True):
        neurons.pop(idx)

    return "", neurons


def channel_cache_files_exist(folder: Path, channel_indices: List[int]) -> bool:
    """
    Scan specified folder for existing Omniplex analog channel data cache files. The cache file name has the format
    f'{CHANNEL_CACHE_FILE_PREFIX}N', where N is the channel index. The method does not validate the contents of the
    files, which are typically quite large.
    :param folder: File system path for the current XSort working directory.
    :param channel_indices: Unordered list of indices of the analog channels that should be cached.
    :return: True if a cache file is found for each channel specified; False if at least one is missing.
    """
    for i in channel_indices:
        f = Path(folder, f"{CHANNEL_CACHE_FILE_PREFIX}{str(i)}")
        if not f.is_file():
            return False
    return True


def unit_cache_file_exists(folder: Path, uid: str) -> bool:
    """
    Does an internal cache file exist for the specified neural unit in the specified working directory? The unit cache
    file name has the format f'{UNIT_CACHE_FILE_PREFIX}{label}', where {label} is a label string uniquely identifying
    the unit. The method does not validate the contents of the file.
    :param folder: File system path for the current XSort working directory.
    :param uid: A label uniquely identifying the unit.
    :return: True if the specified unit cache file is found; else False.
    """
    return Path(folder, f"{UNIT_CACHE_FILE_PREFIX}{uid}").is_file()


def delete_unit_cache_file(folder: Path, uid: str) -> None:
    """
    Delete the internal cache file for the specified neural unit, if it exists in the specified working directory.
    :param folder: File system path for the current XSort working directory.
    :param uid: A label uniquely identifying the unit.
    """
    p = Path(folder, f"{UNIT_CACHE_FILE_PREFIX}{uid}")
    p.unlink(missing_ok=True)


def delete_all_derived_unit_cache_files_from(folder: Path) -> None:
    """
    Remove all internal cache files for "derived units" (created via merge or split operations) from the specified
    working directory. Unit cache files end with the unit UID, and the UID of every derived unit ends with the letter
    'x' -- so it is a simple task to selectively delete derived unit cache files.
    :param folder: File system path for the current XSort working directory. No action taken if directory invalid.
    """
    if not (isinstance(folder, Path) and folder.is_dir()):
        return
    for child in folder.iterdir():
        if child.is_file() and child.name.endswith('x'):
            child.unlink(missing_ok=True)


def save_neural_unit_to_cache(folder: Path, unit: Neuron, suppress: bool = True) -> bool:
    """
    Save the spike train and other computed metrics for the specified unit to an internal cache file in the specified
    XSort working directory. If a cache file already exists for the unit, it will be overwritten.

    All unit cache files in the working directory start with the same prefix (UNIT_CACHE_FILE_PREFIX) and end with the
    UID of the neural unit. The unit spike times and metrics are written to the binary file as follows:
     - A 20-byte header: [best SNR (f32), primary channel index (U32), total number of spikes in unit's spike times
       array (U32), number of per-channel spike templates (U32), template length (U32)].
     - The byte sequence encoding the spike times array, as generated by np.ndarray.tobytes().
     - For each template: Source channel index (U32) followed by the byte sequence from np.ndarray.tobytes().

    Computing the best SNR (and thereby identifying the unit's "primary channel") and the per-channel mean spike
    template waveforms takes a considerable amount of time. When a derived unit is created by the user via a "merge" or
    "split" operation, it is important to cache the spike times for the new unit immediately (the spike train is not
    persisted anywhere else!); we cannot wait for the metrics to be computed. To this end, the unit cache file can come
    in either of two forms:
     - The "complete" version as described above.
     - The "incomplete" version which stores only the spike train. In this version, the header is [-1.0, 0, N, 0, 0],
       where N is the number of spikes.


    :param folder: File system path for the XSort working directory.
    :param unit: The neural unit object.
    :param suppress: If True, any exception (bad working directory, file IO error) is suppressed. Default is True.
    :return: True if successful, else False.
    :raises Exception: If an error occurs and exceptions are not suppressed.
    """
    unit_cache_file = Path(folder, f"{UNIT_CACHE_FILE_PREFIX}{unit.uid}")
    try:
        # TODO: If unit cache file exists, really should write to temp file first...
        with open(unit_cache_file, 'wb') as dst:
            best_snr = -1.0 if (unit.primary_channel is None) else unit.snr
            primary_channel_idx = 0 if (unit.primary_channel is None) else unit.primary_channel
            dst.write(struct.pack('<f4I', best_snr, primary_channel_idx, unit.num_spikes,
                                  unit.num_templates, unit.template_length))

            dst.write(unit.spike_times.tobytes())

            for k in unit.template_channel_indices:
                template = unit.get_template_for_channel(k)
                dst.write(struct.pack('<I', k))
                dst.write(template.tobytes())
        return True
    except Exception as e:
        unit_cache_file.unlink(missing_ok=True)
        if not suppress:
            raise e
        else:
            return False


def load_neural_unit_from_cache(folder: Path, uid: str, suppress: bool = True) -> Optional[Neuron]:
    """
    Load the specified neural unit from the corresponding internal cache file in the specified working directory. The
    cache file may contain only the unit's spike train (the "incomplete" version) or the spike train along with SNR,
    primary channel index and per-channel mean spike template waveforms (the "complete" version). See
    :method:`save_neural_unit_to_cache` for file format details.

    :param folder: File system path for the XSort working directory.
    :param uid: The neural unit's UID.
    :param suppress: If True, any exception (file not found, file IO error) is suppressed. Default is True.
    :return: A **Neuron** object encapsulating the spike train and any cached metrics for the specified neural unit, or
        None if an error occurred and exceptions are suppressed.
    :raises Exception: If an error occurs and exceptions are not suppressed. However, an exception is thrown
        regardless if the unit label is invalid.
    """
    # validate unit label and extract unit index. Exception thrown if invalid
    unit_idx, unit_suffix = Neuron.dissect_uid(uid)

    try:
        unit_cache_file = Path(folder, f"{UNIT_CACHE_FILE_PREFIX}{uid}")
        if not unit_cache_file.is_file():
            raise Exception(f"Unit metrics cache file missing for neural unit {uid}")

        with open(unit_cache_file, 'rb') as fp:
            hdr = struct.unpack_from('<f4I', fp.read(struct.calcsize('<f4I')))
            if len(hdr) != 5:
                raise Exception(f"Invalid header in unit metrics cache file for neural unit {uid}")
            incomplete = hdr[0] < 0
            n_bytes = struct.calcsize("<{:d}f".format(hdr[2]))
            spike_times = np.frombuffer(fp.read(n_bytes), dtype='<f')
            template_dict: Dict[int, np.ndarray] = dict()
            if not incomplete:
                template_len = struct.calcsize("<{:d}f".format(hdr[4]))
                for i in range(hdr[3]):
                    channel_index: int = struct.unpack_from('<I', fp.read(struct.calcsize('<I')))[0]
                    template = np.frombuffer(fp.read(template_len), dtype='<f')
                    template_dict[channel_index] = template

            unit = Neuron(unit_idx, spike_times, suffix=unit_suffix)
            if not incomplete:
                unit.update_metrics(hdr[1], hdr[0], template_dict)
            return unit
    except Exception:
        if not suppress:
            raise
        else:
            return None


if __name__ == "__main__":
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()

            self.setWindowTitle("Testing WorkingDirectory")

            dir_label = QLabel("Directory:")
            self._dir_edit = QLineEdit()
            button = QPushButton("Load")
            button.clicked.connect(self.button_clicked)
            self._result_label = QLabel("<enter directory and press button to test>")

            control_grp = QGroupBox("Hello")
            control_grp_layout = QGridLayout()
            control_grp_layout.addWidget(dir_label, 0, 0)
            control_grp_layout.addWidget(self._dir_edit, 0, 1)
            control_grp_layout.addWidget(button, 1, 1)
            control_grp_layout.addWidget(self._result_label, 2, 0, 1, 2)
            control_grp.setLayout(control_grp_layout)
            self.setCentralWidget(control_grp)
            self.setMinimumWidth(600)

        def button_clicked(self, _: bool):
            folder = Path(self._dir_edit.text())
            if not folder.exists():
                self._result_label.setText('Specified path does not exist')
            elif not folder.is_dir():
                self._result_label.setText('Specified path is not a directory')
            else:
                emsg, work_dir = WorkingDirectory.load_working_directory(folder)
                if len(emsg) > 0:
                    self._result_label.setText(f"Error: {emsg}")
                elif work_dir is None:
                    self._result_label.setText(f"User cancelled")
                else:
                    self._result_label.setText(f"Success! Working directory is valid.")

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    app.exec()
