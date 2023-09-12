from typing import Optional, Dict

import numpy as np


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
        self._templates: Dict[str, np.ndarray] = dict()
        """
        Per-channel mean spike waveforms, or 'templates', computed for this unit, keyed by the Omniplex analog channel 
        source. Channel IDs have the form WB<N> or SPKC<N>, where: WB indicates a wideband channel, SPKC indicates a 
        narrowband channel, and <N> is an integer identifying the channel number **within that channel source type**. 
        Each waveform is computed by averaging 10ms clips [T-1:T+9] from the original recorded analog channel data 
        stream at each spike time T. Stored in internal cache file in the XSort working directory.
        
        All analog channels are assumed to be sampled at the same rate, so all templates have the same length N. The
        sample interval, then, is 10ms/len(template).
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
            print(f"DEBUG: is_float = {isinstance(out, float)}")   # TODO: DEBUG
        return out

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

    @property
    def snr(self) -> Optional[float]:
        """
        The signal-to-noise ratio for this neural unit as estimated from the mean spike waveform recorded on the
        primary channel.
        :return: None if the mean spike waveforms have not yet been computed; otherwise, the unit's computed SNR.
        """
        return self._snr

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
            whenever the XSort working directory changes or a new neural unit is defined). Otherwise, a Numppy array
            containing the 10-ms mean spike waveform as recorded on the specified channel, in microvolts. The number
            of samples in the array will depend on the analog channel sampling rate: R = len(template)/0.01s.
        """
        return self._templates.get(channel)

    def matching_spike_trains(self, n: "Neuron") -> bool:
        """ Does the specified neural unit have the same spike train as this unit? """
        return np.array_equal(self.spike_times, n.spike_times, equal_nan=True)
