from typing import Optional, Dict, Tuple

import numpy as np


class ChannelTraceSegment:
    """
    A container holding a small segment of a recorded analog channel trace, including information to locate the
    segment in the recorded timeline and convert the raw ADC samples to microvvolts.
    """
    def __init__(self, channel: int, start: int, samples_per_sec: float, to_microvolts: float, segment: np.ndarray):
        """
        Construct an analog channel trace segment.

        :param channel: The analog channel index.
        :param start: Offset, from the start of the recorded timeline, to the first sample in this trace segment.
        :param samples_per_sec: The sampling rate for the channel (Hz).
        :param to_microvolts: Multiply trace segment by this scale foctor to convert raw ADC samples to microvolts.
        :param segment: The trace segment recorded on the specified channel, starting at the specified sample offset.
            The Numpy array data type should be int16.
        """
        self._channel = channel
        """ Index of analog source channel. """
        self._start = start
        """ Offset, from the start of recorded timeline, to the first sample in this trace segment. """
        self._rate = samples_per_sec
        """ Sampling rate for source channel. """
        self._scale = to_microvolts
        """ Multiplicative scale factor converts row ADC sample values to microvolts. """
        self._seg = segment
        """ The raw ADC samples in trace segment. """

    @property
    def channel_index(self) -> int:
        """ Index of analog data channel. """
        return self._channel

    @property
    def length(self) -> int:
        """ Number of data samples in this trace segment. """
        return len(self._seg)

    @property
    def t0(self) -> float:
        """ Elapsed recording time at the start of this trace segment, in seconds. """
        return self._start / self._rate

    @property
    def duration(self) -> float:
        """ Duration of trace segment, in seconds. """
        return len(self._seg) / self._rate

    @property
    def raw_trace(self) -> np.ndarray:
        """ The channel trace segment (raw ADC samples).  """
        return self._seg

    @property
    def trace_in_microvolts(self) -> np.ndarray:
        """ The channel trace segment, converted to microvolts. """
        return self._seg * self._scale


class Neuron:
    """
    A container holding the spike train, computed metrics, and important metadata for a neural unit.
    TODO: UNDER DEV
    """
    @staticmethod
    def omniplex_channel_id(is_wideband: bool, ch: int) -> str:
        """
        Get the channel label for a wideband or narrowband analog Omniplex channel on which extracellular recordings
        are made: 'WB<N>' for wideband and 'SPKC<N>' for narrowband, where N is the channel number within that channel
        type.
        :param is_wideband: True for wideband, False for narrowband.
        :param ch: The wideband on narrowband channel number.
        :return: The channel label composed as described above.
        """
        return f"{'WB' if is_wideband else 'SPKC'}{str(ch)}"

    @staticmethod
    def dissect_unit_label(label: str) -> Tuple[int, Optional[bool]]:
        """
        By design, the label uniquely identifying a neural unit in XSort has three possible forms: a simple integer
        string for non-Purkinje units, 'Nc' for a unit representing the complex spikes from a Purkinje cell assigned
        the integer index N, and 'Ns' for a unit representing the simple spikes from that same Purkinje cell.

        This method dissects the label to extract the integer N and a flag indicating whether the unit is a Purkinje
        complex spike train (True), a Purkinje simple spike train (False), or a non-Purkinje unit (None)
        :return: A 2-tuple (N, flag), as described above.
        :raises Exception: If the label string is invalid.
        """
        if label.endswith(('c', 's')):
            is_complex = label.endswith('c')
            index_str = label[:-1]
            if not index_str.isdigit():
                raise Exception("Invalid unit label")
            index = int(index_str)
        else:
            is_complex = None
            if not label.isdigit():
                raise Exception("Invalid unit label")
            index = int(label)
        return index, is_complex

    def __init__(self, idx: int, spike_times: np.ndarray, is_complex: Optional[bool] = None):
        """
        Initialize a neural unit.

        :param idx: An integer index assigned to unit. For neural units obtained from the original spike sorter results
        file, this should be the ordinal index (starting from 1) of the unit.
        :param spike_times: The unit's spike train, ie, an array of spike occurrence times **in seconds elapsed since
        the start of the electrophysiological recording**.
        :param is_complex: True for the complex spikes train of a Purkinje cell, False for the simple spikes train of a
        Purkinje cell; and None for a non-Purkinje cell.
        """
        self._idx = idx
        """
        The integer index assigned to unit. Note that a Purkinje cell has two associated units, which willl have the
        same index -- so this does not uniquely identify a unit.
        """
        self._label = f"{str(idx)}{'c' if is_complex else 's'}" if isinstance(is_complex, bool) else str(idx)
        """ 
        A label which uniquely identifies the unit and has one of three forms: 'N' for a non-Purkinje cell; 'Ns' for
        the simple spikes of a Purkinje cell, and 'Nc' for the complex spikes of the same Purkinje cell -- where N is
        the integer index assigned to the unit.
        """
        self.spike_times = spike_times.astype(dtype='<f')
        """ The unit's spike train: spike times in seconds elapsed since the start of EPhys recording (f32). """
        self._templates: Dict[int, np.ndarray] = dict()
        """
        Per-channel mean spike waveforms, or 'templates', computed for this unit, keyed by the Omniplex analog channel 
        index. Each waveform is computed by averaging 10ms clips [T-1:T+9] from the original recorded analog channel 
        data stream at each spike time T. Stored in internal cache file in the XSort working directory.
        
        All analog channels are assumed to be sampled at the same rate, so all templates have the same length N. The
        sample interval, then, is 10ms/len(template).
        """
        self._primary_channel: Optional[int] = None
        """ 
        Index of the Omniplex analog channel with the highest estimated signal-to-noise ratio (SNR) for this unit. If 
        None, then the primary channel has not yet been determined.
        """
        self._snr: Optional[float] = None
        """
        Estimated signal-to-noise ratio for this unit on the primary recording channel. If None, then the SNR has not
        yet been computed because the unit's per-channel mean spike waveforms have not yet been computed. The unit's
        SNR for a given analog channel is estimated as the peak-to-peak amplitude of the mean spike waveform as measured
        on that channel, divided by the background noise level on the channel.
        """

    @property
    def label(self) -> str:
        """
        A label uniquely identifying this neural unit. For non-Purkinje cell units, this is simply an integer index
        (starting from 1) assigned to the unit. For a neural unit representing the simple or complex spike train of a
        Purkinje cell, the character 's' or 'c' is appended to that index.
        """
        return self._label

    def is_purkinje(self) -> bool:
        """ True if this neural unit represents the simple or complex spike train of a Purkinje cell. """
        return (self._label.find('c') > 0) or (self._label.find('s') > 0)

    @property
    def mean_firing_rate_hz(self) -> float:
        """ This unit's mean firing rate in Hz. """
        if len(self.spike_times) > 2:
            return len(self.spike_times) / (self.spike_times[-1] - self.spike_times[0])
        return 0

    @property
    def num_spikes(self) -> int:
        """ Total number of spikes recorded for this unit. """
        return len(self.spike_times)

    @property
    def fraction_of_isi_violations(self) -> float:
        """
        Fraction of interspike intervals (ISI) in this unit's spike train that are less than 1ms. An ISI less than
        the typical refractory period for a neuron is an indication that some of the spike timestamps attributed to
        this unit are simply noise or should be assigned to another unit.
        """
        out: float = 0
        if self.num_spikes > 2:
            n_violations = np.sum(np.diff(self.spike_times) < 0.001)
            out = n_violations / self.num_spikes
        return out

    @property
    def primary_channel(self) -> Optional[int]:
        """
        The Omniplex analog channel with the largest estimated signal-to-noise ratio (SNR) for this neural unit. **Since
        SNR is estimated as the peak-to-peak amplitude of the mean spike waveform on a channel divided by the mean noise
        level on the channel, the primary channel is undefined until the unit's mean spike waveform has been computed
        for all recorded Omniplex wideband or narrowband analog channels.

        :return: None if the mean spike waveforms have not yet been computed; else, returns the primary channel index.
        """
        return self._primary_channel

    @property
    def snr(self) -> Optional[float]:
        """
        The highest signal-to-noise ratio (SNR) for this neural unit across all recorded analog channels. SNR is
        estimated as the peak-to-peak amplitude of the mean spike waveform on a given analog channel divided by the mean
        noise level on that channel. The channel on which the best SNR was measured is considered the "primary channel"
        for the unit.
        :return: None if the mean spike waveforms have not yet been computed; otherwise, the unit SNR as described.
        """
        return self._snr

    @property
    def amplitude(self) -> Optional[float]:
        """
        THe peak-to-peak amplitude of the unit's mean spike waveform as measured on the primary channel -- ie, the
        analog channel exhibiting the best signal-to-noise ratio for this unit.

        :return: None if the per-channel mean spike waveforms have not yet been computed; otherwise, returns the
            peak-to-peak voltage of the mean spike waveform recorded on the primary channel. In microvolts.
        """
        if self._primary_channel is None:
            return None
        else:
            wv = self._templates.get(self._primary_channel)
            return float(np.max(wv) - np.min(wv))

    def get_template_for_channel(self, idx: int) -> Optional[np.ndarray]:
        """
        Get this unit's mean spike waveform, or 'template', as computed from the data stream on the specified Omniplex
        analog data channel.

        :param idx: Index of an Omniplex analog channel.
        :return: None if the channel index is invalid or was not found in the Omniplex recording file, or if the mean
            spike waveform has not yet been computed for that channel (that computation happens on a background task
            whenever the XSort working directory changes or a new neural unit is defined). Otherwise, a Numppy array
            containing the 10-ms mean spike waveform as recorded on the specified channel, in microvolts. The number
            of samples in the array will depend on the analog channel sampling rate: R = len(template)/0.01s.
        """
        return self._templates.get(idx)

    def update_metrics(self, primary_ch: int, snr: float, templates: Dict[int, np.ndarray]) -> None:
        """
        Set unit metrics which are not part of the spike sorter results file but must be computed in a background
        task after building the internal cache in the current XSort working directory.
            This method is intended only for updating the neural unit record in the background.

        :param primary_ch: Index of the Omniplex analog channel exhibiting the best SNR for this unit.
        :param snr: The best observed signal-to-noise ratio across all recorded Omniplex analog channels.
        :param templates: Dictionary maps Omniplex analog channel index to this unit's mean spike waveform as measured
        on that channel.
        """
        self._primary_channel = primary_ch
        self._snr = snr
        self._templates = templates

    def matching_spike_trains(self, n: "Neuron") -> bool:
        """ Does the specified neural unit have the same spike train as this unit? """
        return np.array_equal(self.spike_times, n.spike_times, equal_nan=True)
