import csv
import pickle
import struct
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Union

import numpy as np
from PySide6.QtGui import QIntValidator, QDoubleValidator, QFont
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QComboBox, QLineEdit, QCheckBox, QVBoxLayout, \
    QGroupBox, QGridLayout, QMainWindow

from xsort.data import PL2
from xsort.data.neuron import Neuron, ChannelTraceSegment

DIRINFO_FILE: str = '.xs.directory.txt'
"""
Internal directory configuration file persists the filenames of the original analog data and unit data source files,
as well as required parameters if analog data is stored in a flat binary file (.bin or .dat) rather than an Omnniplex
PL2 file. 
"""
CHANNEL_CACHE_FILE_PREFIX: str = '.xs.ch.'
""" Prefix for analog channel data stream cache file -- followed by the channel index. """
NOISE_CACHE_FILE: str = '.xs.noise'
""" 
Internal cache file holds the estimated noise level (in raw ADC units, not voltage) on each recorded analog data 
channel. Stored as a list of N single-precision floating-point values, where N is the number of analog channels.
"""
UNIT_CACHE_FILE_PREFIX: str = '.xs.unit.'
""" Prefix for a neural unit cache file -- followed by the unit's UID. """
EDIT_HISTORY_FILE: str = '.xs.edits.txt'
""" Internal cache file holds the edit history for the working directory's neural unit list. """


class UserEdit:
    """
    A simple container encapsulating an individual user-initiated change to the list of neural units defined in the
    original spike sorter results file in the XSort working directory. The following edit operations are supported:
     - Label a neural unit (typically to specify the putative neuron type).
     - Delete a neural unit. Often an "uncurated" spike sorter results file contains many "garbage" units, and the
       user's first task in XSort is to simply delete all such units.
     - Create a new unit by merging two selected units.
     - Split an existing unit. This is done by specifying a subset of the existing unit's spikes to assign to one unit,
       while the remaining are assigned to another. In practice, this is done on the XSort UI by "lassoing" a population
       of spikes in the principal component analysis (PCA) view.

    The edit history for an XSort working directory is simply a list of :class:`UserEdit` objects, in the order in
    which the edits were applied.
    """
    LABEL: str = 'LABEL'
    """ Edit a unit label: 'LABEL,unit_uid,old_label,new_label'."""
    DELETE: str = 'DELETE'
    """ Delete a unit: 'DELETE,unit_uid'. """
    MERGE: str = 'MERGE'
    """ Merge two selected units into one: 'MERGE,uid1,uid2,uid_merged'. """
    SPLIT: str = 'SPLIT'
    """ Split a unit into two units: 'SPLIT,uid_before,uid_split1,uid_split2'. """
    _EDIT_OPS = {LABEL: 'Change unit label', DELETE: 'Delete unit', MERGE: 'Merge units', SPLIT: 'Split unit'}
    """ All recognized edit operations. """

    def __init__(self, op: str, params: List[str]):
        """
        Construct an edit record. The parameter list depend on the edit operation, as follows:
         - op = 'LABEL': params=[<UID of affected neural unit>, <previous label>, <new label>].
         - op = 'DELETE': params=[<UID of deleted neural unit>].
         - op = 'MERGE': params=[uid1, uid2, uid_merged].
         - op = 'SPLIT': params=[uid_split, uid1, uid2].

        :param op: The edit operation.
        :param params: The operation parameter list, as described above. All parameter values are strings.
        :raises IndexError: If the parameter list is not long enough (extra parameter values are ignored).
        :raises TypeError: If any parameter value is not a string.
        :raises ValueError: If any parameter value contains a comma. Edit operations are stored in the edit history
            file as a comma-separated list of strings.
        """
        if not (op in UserEdit._EDIT_OPS):
            raise ValueError(f"Unrecognized edit op: {op}")
        self._op = op
        """ The edit operation type. """
        self._params: List[str] = list()
        """ 
        The operation parameters.
        """
        n_params = 1 if (op == UserEdit.DELETE) else 3
        for i in range(n_params):
            if not isinstance(params[i], str):
                raise TypeError('Expected string-valued parameter')
            elif params[i].find(',') > -1:
                raise ValueError('Parameter values may not contain a comma')
            self._params.append(params[i])

    def __str__(self):
        return ','.join((self._op, *self._params))

    def __eq__(self, other: 'UserEdit'):
        return isinstance(other, UserEdit) and (self._op == other._op) and (len(self._params) == len(other._params)) \
            and all([self._params[i] == other._params[i] for i in range(len(self._params))])

    @property
    def operation(self) -> str:
        """ The edit operation type: LABEL, DELETE, MERGE or SPLIT. """
        return self._op

    @property
    def short_description(self) -> str:
        """ A short generic description of the edit operation type. """
        return self._EDIT_OPS[self._op]

    @property
    def longer_description(self) -> str:
        """ A longer description of the edit record that includes the UIDs of the affected/created units. """
        if self._op == UserEdit.LABEL:
            desc = f"Changed unit {self.affected_uids} label: '{self.previous_unit_label}' --> '{self.unit_label}'"
        elif self._op == UserEdit.DELETE:
            desc = f"Deleted unit '{self.affected_uids}'"
        elif self._op == UserEdit.MERGE:
            desc = f"Merged units {self.affected_uids} into unit '{self.result_uids}'"
        else:   # SPLIT
            desc = f"Split unit '{self.affected_uids}' into units {self.result_uids}"
        return desc

    @property
    def affected_uids(self) -> Union[str, Tuple[str, str]]:
        """
        The UID of the unit that is re-labeled, deleted, or split. For a "merge", a 2-tuple containing the UIDs of
        the two merged units.
        """
        return (self._params[0], self._params[1]) if self._op == UserEdit.MERGE else self._params[0]

    @property
    def result_uids(self) -> Union[None, str, Tuple[str, str]]:
        """ The UID of the new merged unit, or the UIDs of the two units resulting from a split. Else None. """
        if self._op == UserEdit.MERGE:
            return self._params[2]
        elif self._op == UserEdit.SPLIT:
            return self._params[1], self._params[2]
        else:
            return None

    @property
    def previous_unit_label(self) -> Optional[str]:
        """ For the "label" op only, the old label assigned to the unit; else None. """
        return self._params[1] if self._op == UserEdit.LABEL else None

    @property
    def unit_label(self) -> Optional[str]:
        """ For the "label" op only, the new label assigned to the unit; else None. """
        return self._params[2] if self._op == UserEdit.LABEL else None

    @staticmethod
    def save_edit_history(working_dir: Path, history: List['UserEdit']) -> Tuple[bool, str]:
        """
        Save the list of user edits to the edit history file within the specifed XSort working directory.

        :param working_dir: The XSort working directory.
        :param history: The edit history to be saved. **If this list is empty, the edit history file (if it exists)
            will be removed from the working directory.**
        :return: A 2-tuple: (True, '') if successful, else (False, <error description>).
        """
        if not working_dir.is_dir():
            return False, "Invalid working directory"

        # if the edit history is empty, obliterate the edit history file
        p = Path(working_dir, EDIT_HISTORY_FILE)
        if len(history) == 0:
            p.unlink(missing_ok=True)
            return True, ""

        try:
            with open(p, 'w', newline='') as f:
                writer = csv.writer(f)
                for ue in history:
                    writer.writerow((ue._op, *ue._params))
        except Exception as e:
            msg = f"Failed to save edit history: {str(e)}"
            return False, msg

        return True, ""

    @staticmethod
    def load_edit_history(working_dir: Path) -> Tuple[str, List['UserEdit']]:
        """
        Load the contents of the edit history file from the specified XSort working directory.

        :param working_dir: The XSort working directory.
        :return: A 2-typle: ('', H) on success or (<error description>, []) on failure, where H is the edit history
            read from the file. If the file is missing from the working directory, it is assumed the edit history is
            empty.
        """
        if not working_dir.is_dir():
            return "Invalid working directory", []

        # if edit history file does not exist, the edit history is empty
        p = Path(working_dir, EDIT_HISTORY_FILE)
        if not p.is_file():
            return "", []

        try:
            history: List[UserEdit] = list()
            with open(p, 'r', newline='') as f:
                reader = csv.reader(f)
                for line in reader:
                    ue = UserEdit(line[0], [line[i] for i in range(1, len(line))])
                    history.append(ue)
            return "", history
        except Exception as e:
            msg = f"Failed to load edit history: {str(e)}"
            return msg, []


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
        self._valid_config = False

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
        self._nchannels_edit.setValidator(QIntValidator(1, 999, self._nchannels_edit))
        self._nchannels_edit.setText('16')
        self._nchannels_edit.setToolTip("Enter number of recorded channels in [1..999].")
        self._nchannels_edit.textEdited.connect(lambda _: self._refresh())

        rate_label = QLabel("Sampling rate (Hz)")
        self._rate_edit = QLineEdit()
        self._rate_edit.setValidator(QIntValidator(1000, 99999, self._rate_edit))
        self._rate_edit.setText('40000')
        self._rate_edit.textEdited.connect(lambda _: self._refresh())
        self._rate_edit.setToolTip("Enter sampling rate in [1000 .. 99999] Hz; same for all channels.")

        scale_label = QLabel("Voltage scaling (*1e-7):")
        self._scale_edit = QLineEdit()
        self._scale_edit.setValidator(QDoubleValidator(0.1, 99, 5, self._scale_edit))
        self._scale_edit.setText('1.52588')
        self._scale_edit.setToolTip('Multiply each 16-bit signed integer sample by '
                                    'this factor to convert it to volts. Range: [0.1 .. 99] x 1e-7')
        self._scale_edit.textEdited.connect(lambda _: self._refresh())

        self._interleaved_cb = QCheckBox('Interleaved?')
        self._interleaved_cb.setToolTip('Check this box if the analog data channel samples are interleaved in the file')
        self._interleaved_cb.setChecked(False)

        self._prefiltered_cb = QCheckBox('Prefiltered?')
        self._prefiltered_cb.setToolTip("Check this box if the analog data has already been bandpass-filtered")
        self._prefiltered_cb.setChecked(True)

        self._warning_label = QLabel("Binary file consistent with number of channels specifed.")
        self._warning_label.setFont(QFont(self._warning_label.font().family(), weight=QFont.Weight.Bold))

        config_grp = QGroupBox("Flat Binary File Configuration")
        config_grp_layout = QGridLayout()
        config_grp_layout.addWidget(nchan_label, 0, 0)
        config_grp_layout.addWidget(self._nchannels_edit, 0, 1)
        config_grp_layout.addWidget(self._interleaved_cb, 0, 2)
        config_grp_layout.addWidget(rate_label, 1, 0)
        config_grp_layout.addWidget(self._rate_edit, 1, 1)
        config_grp_layout.addWidget(self._prefiltered_cb, 1, 2)
        config_grp_layout.addWidget(scale_label, 2, 0)
        config_grp_layout.addWidget(self._scale_edit, 2, 1)
        config_grp_layout.addWidget(self._warning_label, 3, 0, 1, 3)
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
        enable_params = (sel_path.suffix.lower() != '.pl2')
        self._nchannels_edit.setEnabled(enable_params)
        self._rate_edit.setEnabled(enable_params)
        self._scale_edit.setEnabled(enable_params)
        self._interleaved_cb.setEnabled(enable_params)
        self._prefiltered_cb.setEnabled(enable_params)
        if not enable_params:
            self._warning_label.setVisible(False)
            self._valid_config = True
            return

        self._valid_config = False
        msg: Optional[str] = None
        n_channels, rate, scale = 0, 0, 0
        try:
            n_channels = int(self._nchannels_edit.text())
            rate = int(self._rate_edit.text())
            scale = float(self._scale_edit.text())
            if not ((MIN_CHANNELS <= n_channels <= MAX_CHANNELS) and (MIN_VSCALE <= scale <= MAX_VSCALE) and
                    (MIN_SAMPLING_RATE <= rate <= MAX_SAMPLING_RATE)):
                raise Exception()
        except Exception:
            msg = "!!! Incomplete or invalid binary file configuration"

        if msg is None:
            file_size = sel_path.stat().st_size
            if file_size % (2 * n_channels) != 0:
                msg = f"!!! File size ({file_size} bytes) must be a multiple of {2 * n_channels}!"
            else:
                dur_sec = file_size / (2 * n_channels * rate)
                dur_min = int(dur_sec / 60)
                msg = f"Ok. Estimated recording duration is {dur_min} min, {dur_sec - (dur_min * 60):.3f} sec."
                self._valid_config = True

        self._warning_label.setText(msg)
        self._warning_label.setVisible(True)

    def _validate_before_accept(self) -> None:
        """
        If a flat binary file is chosen as the analog data source, verify that the file size is consistent with
        the number of analog channels specifed. If so, or if a PL2 file is the source, then extinguish dialog,
        accepting the user entries.
        """
        self._refresh()
        if self._valid_config:
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
    def voltage_scale_factor(self) -> float:
        """
        Multiplicative scale factor converts 16-bit analog data sample to volts. Applicable only if the analog
        data source is a flat binary file.
        """
        return float(self._scale_edit.text()) * 1e-7

    @property
    def interleaved(self) -> bool:
        """ Is analog data interleaved? Applicable only if analog data source is a flat binary file. """
        return self._interleaved_cb.isChecked()

    @property
    def prefiltered(self) -> bool:
        """ Is analog data prefiltered? Applicable only if analog data source is a flat binary file. """
        return self._prefiltered_cb.isChecked()


MIN_SAMPLING_RATE: int = 1000
""" Minimum supported analog channel sampling rate in Hz for a flat binary analog source. """
MAX_SAMPLING_RATE: int = 99999
""" Maximum supported analog channel sampling rate in Hz for a flat binary analog source. """
MIN_CHANNELS: int = 1
""" Minimum number of supported analog channel streams in a flat binary analog source. """
MAX_CHANNELS: int = 999
""" Maximum number of supported analog channel streams in a flat binary analog source. """
MIN_VSCALE: float = 0.1
""" Minimum scale factor V such that S*V*1e-7 converts raw 16-bit sample S to uV for a flat binary analog source. """
MAX_VSCALE: float = 99
""" Maximum scale factor V such that S*V*1e-7 converts raw 16-bit sample S to uV for a flat binary analog source. """


class WorkingDirectory:
    """
    Encapsulation of an XSort "working directory", which includes the original source files containing recorded analog
    data channel streams and spike-sorted neural units, as well as internal configuration and cache files generated by
    XSort itself, including an edit history file.
    """
    def __init__(self, folder: Path, analog_src: str, unit_src: str, rate: int = 0, num_channels: int = 0,
                 scale: float = 1.52588e-7, interleaved: bool = False, prefiltered: bool = False):
        """
        Construct an XSort working directory and validate its contents.

        :param folder: Directory path
        :param analog_src: Name of the file within directory that serves as the analog data source. Must be an Omniplex
            PL2 file or a flat binary file (.bin or .dat).
        :param unit_src: Name of Python pickle file (.pkl or .pickle) that is the unit data source.
        :param rate: The analog channel sampling rate in Hz. Ignored unless analog source is a flat binary file.
        :param num_channels: # Analog channels recorded. Ignored unless analog source is a flat binary file.
        :param scale: Scale factor V such that raw sample S * V = sample voltage **in microvolts**. Ignored unless
            analog source is a flat binary file.
        :param interleaved: True if analog channel streams are interleaved in flat binary source file.
        :param prefiltered: True if analog data is already bandpass-filtered in flat binary source file.
        """
        self._folder = folder
        """ The working directory file system path. """
        self._analog_src = analog_src
        """ Name of analog data channel stream source file within the working directory. """
        self._unit_src = unit_src
        """ Name of spike-sorted neural unit data source file within the working directory. """
        self._sampling_rate: int = rate
        """ 
        Sampling rate in Hz for analog data streams. Specified by user if analog source is a flat binary file; extracted
        from Omniplex PL2 file.
        """
        self._num_channels: int = num_channels
        """ 
        Number of analog data channels recorded. Specified by user if analog source is a flat binary file; extracted
        from Omniplex Pl2 file. 
        """
        self._to_volts: float = scale
        """ 
        Multiplicative scale factor converts a raw 16-bit analog data sample to volts. Applies to all available
        analog channels in the flat binary file. Ignored for PL2 source file, which includes per-channel scaling
        factors.
        """
        self._analog_channel_indices: List[int] = list()
        """ 
        List of analog channel indices. For a flat binary file, this is always [0..N-1], where N is the number of
        channels specified by user. For an Omniplex source, it depends on which wideband and/or narrowband channels
        were used during the recording session.
        """
        self._interleaved: bool = interleaved and (num_channels > 1)
        """ Are analog data channel samples interleaved in flat binary file? Ignored for PL2 source file. """
        self._prefiltered: bool = prefiltered
        """ Is analog data in flat binary file prefiltered? Ignored for PL2 source file."""
        self._recording_dur_seconds: float = 0
        """ 
        Analog channel recording duration in seconds. For a flat binary file, this is based on file size, the number of 
        analog channels recorded, and the sampling rate specified by user. For a PL2 file, the duration is extracted
        from metadata stored in the file.
        """
        self._recording_dur_samples: int = 0
        """ Analog channel recording duration in total number of samples. """
        self._pl2_info: Optional[Dict[str, Any]] = None
        """ 
        Metadata extracted from the Omniplex PL2 file if that is the analog data source. Ignored if the analog data 
        source is a flat binary file.
        """
        self._original_units: List[str] = list()
        """ List of UIDs identifying the units extracted from the original unit data source file. """
        self._neurons: List[Neuron] = list()
        """ List of neural units extracted from the unit data source file. Cached temporarily at construction time. """
        self._edit_history: List[UserEdit] = list()
        """ 
        The edit history for this working directory, a record of all user-initiated changes to the list of neural units,
        in chronological order.
        """

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
         - Load edit history file if present. Directory is considered invalid if edit history is present but cannot be
           read.

        This method is called at construction time and computes or retrieves several parameters including analog
        sampling rate, analog recording duration, and number of analog channels.

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
                with (open(self.analog_source, 'rb') as fp):
                    self._pl2_info = PL2.load_file_information(fp)
                    channel_list = self._pl2_info['analog_channels']
                    for i in range(len(channel_list)):
                        if channel_list[i]['num_values'] > 0:
                            if channel_list[i]['source'] in [PL2.PL2_ANALOG_TYPE_WB, PL2.PL2_ANALOG_TYPE_SPKC]:
                                self._analog_channel_indices.append(i)
                    ch_indices = self._analog_channel_indices
                    self._num_channels = len(ch_indices)
                    self._sampling_rate = \
                        int(self._pl2_info['analog_channels'][ch_indices[0]]['samples_per_second'])
                    if self._sampling_rate > 0:
                        dur = max([self._pl2_info['analog_channels'][idx]['num_values'] for idx in ch_indices])
                        self._recording_dur_samples = dur
                        self._recording_dur_seconds = dur / self._sampling_rate
                        # verify all channels are sampled at the same rate
                        for k in ch_indices:
                            if self._sampling_rate != int(self._pl2_info['analog_channels'][k]['samples_per_second']):
                                raise Exception("Found different sampling rates among the analog channels!")

            except Exception as e:
                return f"Unable to read Ommniplex (PL2) file: {str(e)}"
        elif (self._num_channels <= 0) or ((self.analog_source.stat().st_size % (self._num_channels * 2)) != 0):
            return "Flat binary analog data source file is not consistent with # of channels specified"
        else:
            # compute analog recording duration based on flat binary file's size and user-specifed sampling rate and
            # number of channels
            num_scans = self.analog_source.stat().st_size / (2 * self._num_channels)
            self._recording_dur_samples = num_scans
            self._recording_dur_seconds = num_scans / float(self._sampling_rate)
            self._analog_channel_indices = [k for k in range(self._num_channels)]

        emsg, history = UserEdit.load_edit_history(self._folder)
        if len(emsg) > 0:
            return emsg
        self._edit_history = history

        emsg, units = self.load_original_neural_units()
        if len(emsg) > 0:
            return emsg
        self._original_units = [u.uid for u in units]
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
        :param window: Main application window (to serve as parent for modal dialog, if needed). If None and user
            input is required, the method fails.
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

        # if we get here, we need user input -- fail if main application window is None
        if window is None:
            return "Unable to load working directory without user input", None
        dlg = _QueryDialog(folder, analog_sources, unit_sources, parent=window)
        dlg.exec()
        if dlg.result() == QDialog.DialogCode.Accepted:
            work_dir = WorkingDirectory(
                folder, dlg.analog_data_source.name, dlg.unit_data_source.name, dlg.sampling_rate,
                dlg.num_analog_channels, dlg.voltage_scale_factor, dlg.interleaved, dlg.prefiltered)
            if work_dir.is_valid:
                work_dir._to_cfg_file()
                return "", work_dir
            else:
                return work_dir.error_description, None
        else:
            return "", None   # user cancelled

    @staticmethod
    def load_working_directory_headless(
            folder: Path, file1: Optional[str] = None, file2: Optional[str] = None, rate: Optional[int] = None,
            n_ch: Optional[int] = None, v_scale: Optional[float] = None, interleaved: Optional[bool] = None,
            prefiltered: Optional[bool] = None) -> Tuple[str, Optional['WorkingDirectory']]:
        """
        Load an XSort working directory without user interaction.

        If only the directory is specified, it must contain a single Omniplex PL2 file as the analog source and a single
        Python pickle file as the unit data source. If the directory contains more than one possible analog source, then
        **file1** or **file2** must specify the name of the file to use. Similarly if the directory contains more than
        one pickle file as the unit data source. Finally, the remaining parameters are required when the analog data
        source is a flat binary file.

        :param folder: File system path for a putative XSort working directory.
        :param file1: Name of analog or unit data source file in directory.
        :param file2: Name of analog or unit data source file in directory. If file1 identifies the analog source file,
            then this identifies the unit source, and vice versa.
        :param rate: Analog channel sampling rate in Hz (required when analog source is flat binary file). Must lie in
            [MIN_SAMPLING_RATE .. MAX_SAMPLING_RATE].
        :param n_ch: Number of analog channels recorded (required when analog source is flat binary file). Must lie in
            [MIN_CHANNELS, MAX_CHANNELS].
        :param v_scale: Scale factor V such that raw sample * V * 1e-7 = sample in microvolts (flat binary file only).
            Must lie in [MIN_VSCALE, MAX_VSCALE].
        :param interleaved: True if analog channel streams are interleaved in flat binary source file.
        :param prefiltered: True if analog data is already bandpass-filtered in flat binary source file.
        :return: A 2-tuple **(emsg, W)**, where **W** is a :class:`WorkingDirectory` object encapsulating the working
            directory content. On failure, **W** is None and **emsg** is a brief description of the error encountered.
        """
        if not (isinstance(folder, Path) and folder.is_dir()):
            return "Invalid or non-existent working directory", None
        _, analog_sources, unit_sources = WorkingDirectory._list_working_directory_files(folder)
        if len(analog_sources) == 0:
            return "No analog data source files (.pl2, .bin, .dat) found", None
        if len(unit_sources) == 0:
            return "No neural unit data source files (.pkl, .pickle) found", None

        # if there are multiple analog and/or unit data sources -- need to figure out which to use
        a_src: Optional[str] = analog_sources[0].name if len(analog_sources) == 1 else None
        u_src: Optional[str] = unit_sources[0].name if len(unit_sources) == 1 else None
        if a_src is None:
            for f in [file1, file2]:
                if isinstance(f, str) and f.lower() in ['.pl2', '.bin', '.dat']:
                    a_src = f
                    break
            if a_src is None:
                return "Ambiguous - multiple analog data source files found in working directory", None
        if u_src is None:
            for f in [file1, file2]:
                if isinstance(f, str) and f.lower() in ['.pkl', '.pickle']:
                    u_src = f
                    break
            if u_src is None:
                return "Ambiguous - multiple unit data source files found in working directory", None

        # if the analog source is an Omniplex PL2 file, we don't need the params specific to a flat binary file
        if a_src.lower().endswith('.pl2'):
            work_dir = WorkingDirectory(folder, a_src, u_src)
            if work_dir.is_valid:
                work_dir._to_cfg_file()
                return "", work_dir
            else:
                return work_dir.error_description, None

        # otherwise, we're dealing with a flat binary file as the analog source
        if (rate is None) or not (MIN_SAMPLING_RATE <= rate <= MAX_SAMPLING_RATE):
            return "Invalid or unspecified sampling rate for a flat binary analog data file", None
        if (n_ch is None) or not (MIN_CHANNELS <= n_ch <= MAX_CHANNELS):
            return "Invalid or unspecified number of channels for a flat binary analog data file", None
        if (v_scale is None) or not (MIN_VSCALE <= v_scale <= MAX_VSCALE):
            return "Invalid or unspecified voltage scale factor for a flat binary analog data file", None
        if (interleaved is None) or (prefiltered is None):
            return "Must specify whether or not flat binary analog data file is interleaved/prefiltered", None
        work_dir = WorkingDirectory(folder, a_src, u_src, rate, n_ch, v_scale*1e-7, interleaved, prefiltered)
        if work_dir.is_valid:
            work_dir._to_cfg_file()
            return "", work_dir
        else:
            return work_dir.error_description, None

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
        If the analog source is a flat binary file (.bin or .dat), then there are an additional 5 string tokens:
         - Sampling rate in Hz (integer string).
         - Number of analog channels (integer string).
         - Multiplicative factor N such that S * N * 1e-7 converts raw 16-bit sample S to microvolts (float string).
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
                    elif len(line) == 7:
                        rate, n_channels, scale = int(line[2]), int(line[3]), float(line[4]) * 1e-7
                        interleaved, prefiltered = (line[5] == "true"), (line[6] == "true")
                        return WorkingDirectory(cfg_path.parent, line[0], line[1], rate, n_channels, scale,
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
                                         str(self._num_channels), f"{self._to_volts * 1.0e7:.5f}",
                                         "true" if self._interleaved else "false",
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
    def is_analog_data_prefiltered(self) -> bool:
        """
        True if the analog data source is a flat binary file AND the analog data therein has been prefiltered.
        Returns False if the data source is an Omniplex PL2 file.
        """
        return (not self.uses_omniplex_as_analog_source) and self._prefiltered

    @property
    def is_analog_data_interleaved(self) -> bool:
        """
        True if the analog data source is a flat binary file AND the analog data channel streams are stored in an
        interleaved fashion: ch0 at t0, ch1 at t0, ..., chN at t0, ch0 at t1, ... Returns False if the data source is
        an Omniplex PL2 file.
        """
        return (not self.uses_omniplex_as_analog_source) and self._interleaved

    @property
    def analog_source(self) -> Path:
        """ File system path for the original analog channel data source file in working directory. """
        return Path(self._folder, self._analog_src)

    def num_analog_channels(self) -> int:
        """ Number of analog channels stored in the original analog source file in this XSort working directory. """
        return self._num_channels

    @property
    def analog_channel_indices(self) -> List[int]:
        """ List of analog channel indices on which electrophysiological data was recorded. """
        return self._analog_channel_indices.copy()

    def label_for_analog_channel(self, idx: int) -> str:
        """
        A short label for the specified analog channel. If the analog source is Omniplex, then the label has the
        form "WB<N>" for a wideband channel and "SPKC<N>" for a narrowband channel, where N is the channel's ordinal
        position in the bank of available wideband ar narrowband channels on the Omniplex system (it is NOT the
        channel index). If the analog source is a flat binary file, then the label is simply "Ch<X>", where X is the
        channel index.
        :param idx: The channel index.
        :return: The channel label, or an empty string if the index is invalid.
        """
        if self._pl2_info is None:
            return f"Ch{idx}" if 0 <= idx < self._num_channels else ""
        else:
            if idx in self._analog_channel_indices:
                ch_dict = self._pl2_info['analog_channels'][idx]
                src = "WB" if ch_dict['source'] == PL2.PL2_ANALOG_TYPE_WB else "SPKC"
                return f"{src}{str(ch_dict['channel'])}"
            else:
                return ""

    @property
    def analog_sampling_rate(self) -> int:
        """ The analog channel sampling rate in Hz. """
        return self._sampling_rate

    def analog_channel_sample_to_uv(self, idx: int) -> float:
        """
        The multiplicative factor converting a raw int16 analog sample on the specified channel to **microvolts**.
        :param idx: The channel index
        :return: The conversion factor, or 0 if channel index is invalid.
        """
        factor = 0
        if self.is_valid and (idx in self._analog_channel_indices):
            if self.uses_omniplex_as_analog_source:
                factor = self._pl2_info['analog_channels'][idx]['coeff_to_convert_to_units'] * 1.0e6
            else:
                factor = self._to_volts * 1.0e6
        return factor

    @property
    def analog_channel_recording_duration_seconds(self) -> float:
        """
        Duration of analog channel recording in seconds.

        If the analog source is the Omniplex, this method reports the maximum observed recording duration (total number
        of samples) across the relevant channels, but typically the duration is the same for all. If the source is a
        flat binary file, then the duration is determined by the file size and the user-specified sampling rate.
        """
        return self._recording_dur_seconds

    @property
    def analog_channel_recording_duration_samples(self) -> int:
        """ Duration of analog recording in # of samples. Should be the same for every analog channel recorded. """
        return self._recording_dur_samples

    @property
    def omniplex_file_info(self) -> Optional[Dict[str, Any]]:
        """
        Metdata extracted from the Omniplex PL2 file as the analog data source for this XSort working directory.
        Returns None if the analog source is a flat binary file. **NOT A COPY. DO NOT MODIFY.**
        """
        return self._pl2_info if self.is_valid else None

    @property
    def unit_source(self) -> Path:
        """ File system path for the original spike-sorted neural unit data source file in working directory. """
        return Path(self._folder, self._unit_src)

    def load_original_neural_units(self) -> Tuple[str, Optional[List[Neuron]]]:
        """
        Load all neural units stored in the unit data source file in this working directory. This Python pickle file
        contains the results of spike sorting as a list of Python dictionaries, where each dictionary represents one
        neural unit. **This is the original list of defined units prior to the user making any edits in XSort.**

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

    def load_current_neural_units(self) -> Tuple[str, Optional[List[Neuron]]]:
        """
        Load the current list of neural units defined in this working directory. This method first loads the original
        list of units, then updates that list IAW the current edit history. **This method is intended only for loading
        the initial state of the unit list when a working directory is loaded, prior to building/loading internal
        cache files.**

        :return: On success, a tuple ("", L), where L is a list of :class:`Neuron` objects encapsulating the working
            directory's current list of neural units. Since the units are loaded from the original unit data source
            file, then updated IAW the edit history, the unit objects will lack cached unit metrics like spike
            templates, SNR, and primary analog channel designation. On failure, returns ('error description', None).
        """
        emsg, neurons = self.load_original_neural_units()
        if len(emsg) > 0:
            return emsg, neurons
        self._apply_edit_history_to(neurons)
        return "", neurons

    def _apply_edit_history_to(self, units: List[Neuron]) -> None:
        for edit_rec in self._edit_history:
            if edit_rec.operation == UserEdit.LABEL:
                for u in units:
                    if (u.uid == edit_rec.affected_uids) and (u.label == edit_rec.previous_unit_label):
                        try:
                            u.label = edit_rec.unit_label
                        except ValueError:
                            pass
                        break
            elif edit_rec.operation == UserEdit.DELETE:
                found_idx = -1
                for i in range(len(units)):
                    if units[i].uid == edit_rec.affected_uids:
                        found_idx = i
                        break
                if found_idx > -1:
                    units.pop(found_idx)
            elif edit_rec.operation == UserEdit.MERGE:
                components: List[Neuron] = list()
                for u in units:
                    if u.uid in edit_rec.affected_uids:
                        components.append(u)
                if len(components) == 2:
                    for u in components:
                        units.remove(u)
                    new_uid = edit_rec.result_uids
                    new_idx = int(new_uid[0:-1])  # remove the 'x' suffix to get the unit index
                    merged_unit = Neuron.merge(components[0], components[1], idx=new_idx)
                    units.append(merged_unit)
            else:  # SPLIT: remove unit was split and append the two unit resulting from that split.
                found_idx = -1
                for i in range(len(units)):
                    if units[i].uid == edit_rec.affected_uids:
                        found_idx = i
                        break
                if found_idx > -1:
                    units.pop(found_idx)
                    # the units resulting from the split must have corresponding cache files in the working directory
                    result_units: List[Neuron] = list()
                    for uid in edit_rec.result_uids:
                        u = self.load_neural_unit_from_cache(uid)
                        if isinstance(u, Neuron):
                            result_units.append(u)
                    if len(result_units) == 2:
                        units.extend(result_units)

    def load_select_neural_units(self, uids: List[str]) -> Tuple[str, Optional[List[Neuron]]]:
        """
        Load the specified neural units defined in this working directory. All original units are loaded from the
        original unit data source, while derived units (UID ends with suffix 'x') must be loaded from the corresponding
        unit metrics cache file. For such units, XSort will write the unit's spike times to an **incomplete** version
        of the metrics cache file, leaving the computation of metrics (primary channel, SNR, spike templates) and
        "completing" the cache file to a background task.

        This method is intended for collecting a list of units for which metrics (primary channel, spike templates, etc)
        have not yet been calculated and cached.

        :return: On success, a tuple ("", L), where L is a list of :class:`Neuron` objects encapsulating the neural
            units found in the file. Those metrics which are calculated and cached in the backgroun -- mean spike
            waveforms, SNR, primary analog channel -- are undefined. On failure, returns ('error description', None).
        """
        neurons: List[Neuron] = list()

        # load derived units first. It's a fatal error if the (incomplete) cache file for a derived unit is missing.
        for uid in uids:
            if Neuron.is_derived_uid(uid):
                unit = self.load_neural_unit_from_cache(uid)
                if unit is None:
                    return f"No cache file for derived unit {uid}", None
                neurons.append(unit)

        # in case we're only looking for derived units
        if len(neurons) == len(uids):
            return "", neurons

        # load all units defined in the original unit data source
        emsg, original_units = self.load_original_neural_units()
        if len(emsg) > 0:
            return emsg, None

        # augment our list with the original units requested
        for u in original_units:
            if u.uid in uids:
                neurons.append(u)

        return ("", neurons) if (len(neurons) == len(uids)) else (f"At least one UID was invalid/not found", None)

    def unit_metrics_not_cached(self) -> List[str]:
        """
        Scan this XSort working directory and check for any missing or incomplete unit metrics cache files.

        The cache generator task that runs when the working directory is opened will identify the primary channel for
        each defined unit, calculate spike templates for up to 16 channels "near" the primary channel, then save the
        unit spike train, primary channel index, unit SNR on that channel, and spike templates to a "complete" unit
        metrics cache file. Additionally, when XSort creates a derived unit by a merge or split, it will immediately
        write the new unit's spike times to an "incomplete" metrics file.

        :return: List of UIDs identifying all original or derived units for which the unit metrics cache file is
            either missing or incomplete.
        """
        out: List[str] = list()
        if self.is_valid:
            out.extend(self._original_units)
            for child in self._folder.iterdir():
                if child.is_file() and child.name.startswith(UNIT_CACHE_FILE_PREFIX):
                    uid = child.name[len(UNIT_CACHE_FILE_PREFIX):]
                    try:
                        out.remove(uid)
                    except ValueError:
                        # not an original unit cache file -- check for an incomplete cache file for a derived unit
                        unit: Neuron = self.load_neural_unit_from_cache(uid)
                        if (unit is None) or (unit.primary_channel is None):
                            out.append(uid)
        return out

    @property
    def need_analog_cache(self) -> bool:
        """
        Are per-channel internal cache files needed for the analog data source in this XSort working directory?

            For performant retrieval of any portion of an analog data channel's bandpass-filtered stream, that channel's
        samples are extracted from the analog data source file, filtered, and stored in a dedicated internal cache file
        in the working directory. This is the case whenever the analog data source in an Omniplex PL2 file, or a flat
        binary file containing raw **unfiltered** channel data. However, if the analog source is a prefiltered flat
        binary file, then there is no need to cache the analog channel streams.

        :return: True unless the analog data source is a flat binary file containing prefiltered channel data.
        """
        return self.uses_omniplex_as_analog_source or (not self.is_analog_data_prefiltered)

    def analog_channel_cache_files_missing(self) -> bool:
        """
        If per-channel cache files required, scan this XSort working directory to check for any missing files.

        :return: True if analog channel caching is required and at least one channel cache file does not exist.
        """
        if self.is_valid and self.need_analog_cache:
            for i in self._analog_channel_indices:
                if not Path(self._folder, f"{CHANNEL_CACHE_FILE_PREFIX}{str(i)}").is_file():
                    return True
        return False

    def channels_not_cached(self) -> List[int]:
        """
        If this XSort working directory is valid and analog channel caching is required, scan the directory and return
        the list of all analog channels that have not been cached.
        :return: List of channel indices of those analog channels that need to be cached internally, in ascending
            order. Returns an empty list if working directory is invalid, if analog source does not require channel
            caching, or if all channel cache files are present.
        """
        out: List[int] = list()
        if not (self.is_valid and self.need_analog_cache):
            return out
        for i in self._analog_channel_indices:
            if not Path(self._folder, f"{CHANNEL_CACHE_FILE_PREFIX}{str(i)}").is_file():
                out.append(i)
        out.sort()
        return out

    def retrieve_cached_channel_trace(
            self, idx: int, start: int, count: int, suppress: bool = False) -> Optional[ChannelTraceSegment]:
        """
        Retrieve a small portion of a recorded analog data channel trace from the corresponding channel cache file
        in this XSort working directory.
            **NOTE**: When the analog data source is a flat binary file containing prefiltered analog data, there is
        no need to extract and cache the individual analog data channels to separate files. In this case, this method
        reads the channel trace segment from the binary file directly!

        :param idx: The analog channel index.
        :param start: Index of the first sample to retrieve.
        :param count: The number of samples to retrieve.
        :param suppress: If True, any exception (file not found, file IO error) is suppressed. Default is False.
        :return: The requested channel trace segment, or None if an error occurred and exceptions are suppressed.
        """
        try:
            to_microvolts = self.analog_channel_sample_to_uv(idx)
            samples_per_sec = self._sampling_rate

            # SPECIAL CASE: per-channel cache files not needed when the source is a prefiltered flat binary file
            if not self.need_analog_cache:
                n_ch = self.num_analog_channels()
                file_size = self.analog_source.stat().st_size
                if file_size % (2 * n_ch) != 0:
                    raise Exception(f'Flat binary file size inconsistent with specified number of channels ({n_ch})')
                n_samples = int(file_size / (2 * n_ch))
                n_bytes_per_sample = n_ch * 2 if self.is_analog_data_interleaved else 2

                with open(self.analog_source, 'rb') as src:
                    # calc offset to the first sample
                    if not self.is_analog_data_interleaved:
                        ofs = (idx * n_samples + start) * n_bytes_per_sample
                    else:
                        ofs = start * n_bytes_per_sample
                    src.seek(ofs)

                    # read in the samples. If interleaved, we have to subsample the block read.
                    samples = np.frombuffer(src.read(count * n_bytes_per_sample), dtype='<h')
                    if self.is_analog_data_interleaved:
                        samples = samples[idx::n_ch].copy()
                    return ChannelTraceSegment(idx, start, samples_per_sec, to_microvolts, samples)

            cache_file = Path(self._folder, f"{CHANNEL_CACHE_FILE_PREFIX}{str(idx)}")
            if not cache_file.is_file():
                raise Exception(f"Channel cache file missing for analog channel {str(idx)}")

            num_samples = int(cache_file.stat().st_size / 2)
            if (start < 0) or (count <= 0) or (start + count > num_samples):
                raise Exception("Invalid trace segment bounds")

            offset = struct.calcsize("<{:d}h".format(start))
            n_bytes = struct.calcsize("<{:d}h".format(count))
            with open(cache_file, 'rb') as fp:
                fp.seek(offset)
                samples = np.frombuffer(fp.read(n_bytes), dtype='<h')
                return ChannelTraceSegment(idx, start, samples_per_sec, to_microvolts, samples)
        except Exception:
            if not suppress:
                raise
            else:
                return None

    def save_neural_unit_to_cache(self, unit: Neuron) -> bool:
        """
        Save the spike train and other computed metrics for the specified unit to an internal cache file in this XSort
        working directory. If a cache file already exists for the unit, it will be overwritten.

        All unit cache files in the working directory start with the same prefix (UNIT_CACHE_FILE_PREFIX) and end with
        the UID of the neural unit. The unit spike times and metrics are written to the binary file as follows:
         - A 20-byte header: [best SNR (f32), primary channel index (U32), total number of spikes in unit's spike times
           array (U32), number of per-channel spike templates (U32), template length (U32)].
         - The byte sequence encoding the spike times array, as generated by np.ndarray.tobytes().
         - For each template: Source channel index (U32) followed by the byte sequence from np.ndarray.tobytes().

        Computing the best SNR (and thereby identifying the unit's "primary channel") and the per-channel mean spike
        template waveforms takes a considerable amount of time. When a derived unit is created by the user via a "merge"
        or "split" operation, it is important to cache the spike times for the new unit immediately (the spike train is
        not persisted anywhere else!); we cannot wait for the metrics to be computed. To this end, the unit cache file
        can come in either of two forms:
         - The "complete" version as described above.
         - The "incomplete" version which stores only the spike train. In this case, the header is [-1.0, 0, N, 0, 0],
           where N is the number of spikes.

        :param unit: The neural unit object.
        :return: True if successful, else False.
        """
        if not self.is_valid:
            return False
        unit_cache_file = Path(self._folder, f"{UNIT_CACHE_FILE_PREFIX}{unit.uid}")
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
        except Exception:
            unit_cache_file.unlink(missing_ok=True)
            return False

    def load_neural_unit_from_cache(self, uid: str) -> Optional[Neuron]:
        """
        Load the specified neural unit from the corresponding internal cache file in this XSort working directory. The
        cache file may contain only the unit's spike train (the "incomplete" version) or the spike train along with SNR,
        primary channel index and per-channel mean spike template waveforms (the "complete" version). See
        :method:`save_neural_unit_to_cache` for file format details.

        :param uid: The neural unit's UID.
        :return: A **Neuron** object encapsulating the spike train and any cached metrics for the specified neural unit,
            or None if an error occurred (cache file not found, or file IO error).
        """
        if not self.is_valid:
            return None

        # validate unit label and extract unit index. Exception thrown if invalid
        unit_idx, unit_suffix = Neuron.dissect_uid(uid)

        try:
            unit_cache_file = Path(self._folder, f"{UNIT_CACHE_FILE_PREFIX}{uid}")
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
            return None

    def unit_cache_file_exists(self, uid: str) -> bool:
        """
        Does an internal cache file exist for the specified neural unit in this XSort working directory? The unit cache
        file name has the format f'{UNIT_CACHE_FILE_PREFIX}{uid}', where {uid} is the unit's unique identifier. The
        method does not validate the contents of the file.
        :param uid: A label uniquely identifying the unit.
        :return: True if the internal cache file for the specified neural unit is found; else False.
        """
        return self.is_valid and Path(self._folder, f"{UNIT_CACHE_FILE_PREFIX}{uid}").is_file()

    def delete_unit_cache_file(self, uid: str) -> None:
        """
        Delete the internal cache file for the specified neural unit, if it exists in this XSort working directory.
        :param uid: The neural unit's unique identifier.
        """
        if self.is_valid:
            p = Path(self._folder, f"{UNIT_CACHE_FILE_PREFIX}{uid}")
            p.unlink(missing_ok=True)

    def _delete_all_derived_unit_cache_files(self) -> None:
        """
        Remove all internal cache files for "derived units" (created via merge or split operations) from this XSort
        working directory. Unit cache files end with the unit UID, and the UID of every derived unit ends with the
        letter 'x' -- so it is a simple task to selectively delete derived unit cache files.
        """
        if not self.is_valid:
            return
        for child in self._folder.iterdir():
            if child.is_file() and child.name.startswith(UNIT_CACHE_FILE_PREFIX) and child.name.endswith('x'):
                child.unlink(missing_ok=True)

    def channel_noise_cache_file_exists(self) -> bool:
        """
        Does this working directory contain the internal cache file '.xs.noise' holding the estimated noise level on
        each recorded analog channel?
        """
        return self.is_valid and Path(self._folder, NOISE_CACHE_FILE).is_file()

    def load_channel_noise_from_cache(self) -> Optional[np.ndarray]:
        """
        Load estimate analog channel noise levels stored in a dedicated internal cache file ('.xs.noise') within this
        XSort working directory. Analog channel noise is estimated and cached while building out the internal cache
        the first time a working directory is visited.

        :return: Single-precision floating-point 1D array F such that F[k] is the noise level on channel k, k in [0,N),
            where N is the number of analog channels recorded. Returns None if channel noise cache file is missing or
            could not be read.
        """
        try:
            with open(Path(self.path, NOISE_CACHE_FILE), 'rb') as src:
                out = np.frombuffer(src.read(), dtype='<f')
            return out
        except Exception:
            return None

    def save_channel_noise_to_cache(self, noise_levels: List[float]) -> bool:
        """
        Save estimated analog channel noise levels in a dedicated internal cache file ('.xs.noise') within this
        XSort working directory
        :param noise_levels: Array F such that F[k] is the noise level on channel k, k in [0,N), where N is the number
            of analog channels. Noise level should be in raw ADC units rather than converted to volts.
        :return: True if successful, else False. Fails if argument length does not match the number of analog channels
        """
        if len(noise_levels) != self.num_analog_channels():
            return False
        try:
            noise_array = np.array(noise_levels, dtype='<f')
            with open(Path(self.path, NOISE_CACHE_FILE), 'wb') as src:
                src.write(noise_array.tobytes())
            return True
        except Exception:
            return False

    def delete_internal_cache_files(self, analog: bool = True, noise: bool = True, units: bool = True) -> None:
        """
        Remove all or some of the dedicated internal cache files in this XSort working directory.

        :param analog: If True, analog data channel cache files are removed. Default = True.
        :param noise: If True, the analog channel noise cache file is removed. Default = True.
        :param units: If True, neural unit metrics cache files are removed. Default = True.
        """
        if (not self.is_valid) or not (analog or noise or units):
            return
        if noise:
            Path(self._folder, NOISE_CACHE_FILE).unlink(missing_ok=True)
        for p in self._folder.iterdir():
            if ((analog and p.name.startswith(CHANNEL_CACHE_FILE_PREFIX)) or
                    (units and p.name.startswith(UNIT_CACHE_FILE_PREFIX))):
                p.unlink(missing_ok=True)

    def save_edit_history(self) -> None:
        """ Save this working directory's current edit history to a dedicated cache file within the directory. """
        UserEdit.save_edit_history(self._folder, self._edit_history)

    def clear_edit_history(self) -> None:
        """ Clear the edit history for this working directory and remove internal cache files for any derived unit. """
        self._edit_history.clear()
        self._delete_all_derived_unit_cache_files()

    @property
    def is_edited(self) -> bool:
        """ Has the user edited the original neural unit list for this working directory in any way? """
        return len(self._edit_history) > 0

    def remove_most_recent_edit(self) -> Optional[UserEdit]:
        """
        Remove most recent user edit from the working directory's current edit history.
        :return: The edit record, or None if the edit history is empty.
        """
        return self._edit_history.pop() if len(self._edit_history) > 0 else None

    @property
    def most_recent_edit(self) -> Optional[UserEdit]:
        """
        Edit record for the most recent user edit in the working directory's current edit history.
        :return: The edit record, or None if the edit history is empty.
        """
        return self._edit_history[-1] if len(self._edit_history) > 0 else None

    @property
    def edit_history_has_only_unit_label_changes(self) -> bool:
        """ True if edit history only consists of changes in unit labels. """
        return all([rec.operation == UserEdit.LABEL for rec in self._edit_history])

    def on_unit_relabeled(self, uid: str, prev_label: str, curr_label: str) -> None:
        """
        Update the working directory's edit history when a neural unit is re-labeled.
        :param uid: UID of affected unit
        :param prev_label: The old label.
        :param curr_label: The new label.
        """
        edit_rec = UserEdit(op=UserEdit.LABEL, params=[uid, prev_label, curr_label])
        self._edit_history.append(edit_rec)

    def on_unit_deleted(self, uid: str) -> None:
        """
        Update the working directory's edit history when a neural unit is deleted.
        :param uid: UID of deleted unit
        """
        edit_rec = UserEdit(op=UserEdit.DELETE, params=[uid])
        self._edit_history.append(edit_rec)

    def on_units_merged(self, uid1: str, uid2: str, merged_uid: str) -> None:
        """
        Update the working directory's edit history when two neural units are merged into one.
        :param uid1: UID of one of the units merged
        :param uid2: UID of the other unit merged.
        :param merged_uid: UID assigned to the unit resulting from the merge.
        """
        edit_rec = UserEdit(op=UserEdit.MERGE, params=[uid1, uid2, merged_uid])
        self._edit_history.append(edit_rec)

    def on_unit_split(self, split_uid: str, uid1: str, uid2: str) -> None:
        """
        Update the working directory's edit history when a neural unit is split into two distinct units.
        :param split_uid: UID of the unit that was split.
        :param uid1: UID of one unit derived from the split.
        :param uid2: UID of the other unit derived from the split.
        """
        edit_rec = UserEdit(op=UserEdit.SPLIT, params=[split_uid, uid1, uid2])
        self._edit_history.append(edit_rec)

    def find_next_assignable_unit_index(self) -> int:
        """
        Find the next available integer that can be assigned to a new unit created by a merge or split operation. This
        method checks the working directory's edit history for the "derived" unit with the largest index N and returns
        N+1. If the edit history is empty, then it finds the largest index N among the units in the original unit list.
        :return: The next available integer index.
        """
        max_idx: int = 0
        if len(self._original_units) > 0:
            max_idx = max([Neuron.dissect_uid(uid)[0] for uid in self._original_units])
        edit_indices: List[int] = list()
        for edit_rec in self._edit_history:
            if edit_rec.operation == UserEdit.DELETE:
                edit_indices.append(Neuron.dissect_uid(edit_rec.affected_uids)[0])
            elif edit_rec.operation == UserEdit.MERGE:
                edit_indices.append(Neuron.dissect_uid(edit_rec.result_uids)[0])
            elif edit_rec.operation == UserEdit.SPLIT:
                edit_indices.append(Neuron.dissect_uid(edit_rec.result_uids[0])[0])
        if len(edit_indices) > 0:
            max_idx = max(max_idx, max([i for i in edit_indices]))

        return max_idx + 1
