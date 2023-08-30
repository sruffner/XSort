import pickle
from pathlib import Path
from typing import Union, Optional, Dict, Any, List

import numpy as np
from PySide6.QtCore import QObject, Signal

from xsort.data import PL2


class Neuron:
    """
    TODO: UNDER DEV
    """
    def __init__(self, uid: int, spike_times: np.ndarray, sampling_rate_hz: int, which: Optional[str] = ''):
        """
        Initialize a neural unit.

        :param uid: The assigned unit ID. For neural units obtained from the original spike sorter results file, this
        should be the ordinal index (starting from 1) of the unit.
        :param spike_times: Chronological spike timestamps -- in units of 1/sampling_rate_hz, assigned to this unit.
        :param sampling_rate_hz: The sampling rate for the spike_times array.
        :param which: 's' for the simple spikes train of a Purkinje cell, 'c' for the complex spikes train of a
        Purkinje cell; otherwise, the spikes train of a non-Purkinje cell.
        """
        self._uid = uid
        """ Unique ID assigned to this neural unit. """
        self._label = f"{str(uid)}{which}" if which in ['c', 's'] else str(uid)
        """ Neural unit label -- for display purposes. """
        self._sampling_rate = sampling_rate_hz
        """ Sampling rate for unit spike times, in Hz. """
        self._spike_times = spike_times
        """ Array of spike times, ordered chrononologically."""
        self._templates: Dict[str, np.ndarray] = dict()
        """
        Per-channel mean spike waveforms computed for this unit, keyed by the Omniplex analog channel source. Channel 
        IDs have the form WB<N> or SPKC<N>, where: WB indicates a wideband channel, SPKC indicates a narrowband channel,
        and <N> is an integer identifying the channel number **within that channel source type**. Each waveform is 
        computed by averaging 10ms clips [T-1:T+9] from the original recorded analog channel data stream at each spike 
        timestamp T. Stored in internal cache file in the XSort working directory.
        """
        self._primary_channel_id: Optional[str] = None
        """ 
        ID of the Omniplex analog channel on which the unit's computed mean spike waveform has the largest peak-to-peak
        amplitude. If None, then the per-channel mean spike waveforms have not yet been computed, so the notion of a
        primary channel is undefined.
        """
        self._snr: Optional[float] = None
        """
        Estimated signal-to-noise ratio for this unit on the primary recording channel. If None, then the SNR has not
        yet been computed because the unit's per-channel mean spike waveforms have not yet been computed.
        """
    @property
    def label(self) -> str:
        """
        Display label for this neural unit. For non-Purkinje cell units, this is simply the integer index (starting
        from 1) assigned to the unit). For a neural unit representing the simple or complex spike train of a Purkinje
        cell, the character 's' or 'c' is appended to that index.
        """
        return self._label

    @property
    def mean_firing_rate_hz(self) -> float:
        """ This unit's mean firing rate in Hz. """
        if len(self._spike_times) > 2:
            return self._sampling_rate * len(self._spike_times) / (self._spike_times[-1] - self._spike_times[0])
        return 0

    @property
    def num_spikes(self) -> int:
        """ Total number of spikes recorded for this unit. """
        return len(self._spike_times)

    @property
    def fraction_of_isi_violations(self) -> float:
        """
        Fraction of interspike intervals (ISI) in this unit's spike train that are less than 1ms. An ISI less than
        the typical refractory period for a neuron is an indication that some of the spike timestamps attributed to
        this unit are simply noise or should be assigned to another unit.
        """
        return sum((np.diff(self._spike_times) / self._sampling_rate) < 0.001) / (len(self._spike_times) - 1)

    @property
    def primary_channel(self) -> Optional[str]:
        """
        The Omniplex source channel on which the largest (maximum peak-to-peak voltage) mean spike waveform was
        computed for this neural unit. Undefined until the unit's mean spike waveform has been computed for all recorded
        Omniplex wideband or narrowband analog channels.

        :return:  None if the mean spike waveforms have not yet been computed; otherwise, returns the primary channel
            ID. The channel ID has the form f"{src}{ch}", where {src} is "WB" (wideband channel) or "SPKC" (narrowband)
            and {ch} is the channel number for that channel source type.
        """
        return self._primary_channel_id

    def snr(self) -> Optional[float]:
        """
        The signal-to-noise ratio for this neural unit as estimated from the mean spike waveform recorded on the
        primary channel.
        :return: None if the mean spike waveforms have not yet been computed; otherwise, the unit's computed SNR.
        """

    @property
    def amplitude(self) -> Optional[float]:
        """
        Largest observed peak-to-peak amplitude of the unit's mean spike waveform across all recorded Omniplex wideband
        or narrowband analog channels.

        :return: None if the mean spike waveforms have not yet been computed; otherwise, returns the peak-to-peak
            voltage of the largest mean spike waveform for this unit. In microvolts.
        """
        if self._primary_channel_id is None:
            return None
        else:
            wv = self._templates.get(self._primary_channel_id)
            return np.max(wv) - np.min(wv)

    def get_template_for_channel(self, channel: str) -> Optional[np.ndarray]:
        """
        Get this unit's mean spike waveform as computed from the data stream on the specified Omniplex analog data
        channel.

        :param channel: The Omniplex analog channel ID, having the form "WB<N>" for wideband channel with index <N> or
            "SPKC<N>" for narrowband channel with index <N>.
        :return: None if the channel ID is invalid or was not found in the Omniplex recording file, or if the mean
            spike waveform has not yet been computed for that channel (that computation happens on a background task
            whenever the XSort working directory changes or a new neural unit is defined).
        """
        return self._templates.get(channel)


class Analyzer(QObject):
    """
    TODO: UNDER DEV
    """

    working_directory_changed: Signal = Signal()
    """ Signals that working directory has changed. All views should refresh accordingly. """

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
        pl2_file: Optional[Path] = None
        pkl_file: Optional[Path] = None
        for child in _p.iterdir():
            if child.is_file():
                ext = child.suffix.lower()
                if ext in ['.pkl', '.pickle']:
                    if pkl_file is None:
                        pkl_file = child
                    else:
                        return "Multiple spike sorter results files (PKL) found"
                elif ext == '.pl2':
                    if pl2_file is None:
                        pl2_file = child
                    else:
                        return "Multiple Omniplex files (PL2) found"
        if pl2_file is None:
            return "No Omniplex file (PL2) found in directory"
        if pkl_file is None:
            return "No spike sorter results file (PKL) found in directory"

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
        try:
            with open(pkl_file, 'rb') as f:
                res = pickle.load(f)
                ok = isinstance(res, list) and all([isinstance(k, dict) for k in res])
                if not ok:
                    raise Exception("Unexpected content found")
                for i, u in enumerate(res):
                    if u['type__'] == 'PurkinjeCell':
                        neurons.append(Neuron(i + 1, u['spike_indices__'], u['sampling_rate__'], 's'))
                        neurons.append(Neuron(i + 1, u['cs_spike_indices__'], u['sampling_rate__'], 'c'))
                    else:
                        neurons.append(Neuron(i + 1, u['spike_indices__'], u['sampling_rate__']))
                    # TODO: TESTING
                    if u['type__'] == 'PurkinjeCell':
                        print(f"Unit {neurons[-2].label}: #spikes={neurons[-2].num_spikes}, "
                              f"%badISI = {neurons[-2].fraction_of_isi_violations:.1f}, "
                              f"firing rate (Hz) = {neurons[-2].mean_firing_rate_hz:.2f}")
                        print(f"Unit {neurons[-1].label}: #spikes={neurons[-1].num_spikes}, "
                              f"%badISI = {neurons[-1].fraction_of_isi_violations:.1f}, "
                              f"firing rate (Hz) = {neurons[-1].mean_firing_rate_hz:.2f}")
                    else:
                        print(f"Unit {neurons[-1].label}: #spikes={neurons[-1].num_spikes}, "
                              f"%badISI = {neurons[-1].fraction_of_isi_violations:.1f}, "
                              f"firing rate (Hz) = {neurons[-1].mean_firing_rate_hz:.2f}")
        except Exception as e:
            return f"Unable to read spike sorter results from PKL file: {str(e)}"

        # success
        self._working_directory = _p
        self._pl2_file = pl2_file
        self._pl2_info = pl2_info
        self._channel_map = channel_map
        self._pkl_file = pkl_file
        self._neurons = neurons

        # TODO: Spawn background task to update everything!

        # signal views
        self.working_directory_changed.emit()

        return None
