from enum import Enum
from typing import Optional, Dict, Tuple

import numpy as np

from xsort.data import stats


class DataType(Enum):
    """ The different types of data objects generated/retrieved by :class:`Analyzer` via background tasks. """
    NEURON = 1,   # neural unit record
    CHANNELTRACE = 2,   # analog channel traces for a 1-second segment of the analog recording
    ISI = 3,   # ISI histogram for a neural unit in the focus list
    ACG = 4,   # Autocorrelogram for a neural unit in the focus list
    CCG = 5,   # Crosscorrelogram for a neual unit vs another unit in the focus list
    PCA = 6    # PCA projection for a neural unit in the focus list


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
    A container holding the spike train, computed metrics, important metadata, and cached statistics for a neural unit.

        When initially constructed from the spike sorter results file in the XSort working directory, the only
    available information about a neural unit is its label and spike train. Metadata and metrics are computed in the
    background and stored, along with the spike train, in an internal cache file in the working directory. Additional
    statistics -- the autocorrelogram, interspike interval histogram, cross correlograms with other units, and principal
    component analysis results -- are computed at application runtime and cached in this object.
        Principal component analysis is performed on the spike trains of the neural units in the current "focus list".
    The result of PCA is a projection of each unit's spikes into a 2D space defined by the 2 principal components
    exhibiting the most variance (aka, the most "information"), and it is this projection which is cached in this
    object. Of course, each time the "focus list" changes, the analysis must be redone, so any previously cached
    projection is reset.
    """
    FIXED_HIST_SPAN_MS: int = 200
    """ The (fixed) time span for all neural unit ISI histograms and correlograms, in milliseconds. """

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
        self._cached_isi: Optional[np.ndarray] = None
        """
        The interspike interval histogram for this unit, or None if the ISI has not been computed. Lazily created on a
        background thread when needed.
        """
        self._cached_acg: Optional[np.ndarray] = None
        """ 
        The autocorrelogram for this unit, or None if the ACG has not yet been computed. Lazily created on a background
        thread when needed.
        """
        self._cached_ccgs: Dict[str, np.ndarray] = dict()
        """ 
        Cache of any cross-correlograms computed for this unit WRT another unit, keyed by the other unit's label. Will
        be empty if no CCGs have been computed. CCGs are computed and cached here on a background thread when needed.
        """
        self._cached_pca_projection: Optional[np.ndarray] = None
        """ 
        Cached projection of this unit's spike clips across all analog source channels onto a 2D plane defined by the
        two highest-variance principal components calculated using the mean spike templates across all recorded analog
        channels for all units currently in the neural unit focus list. Will be None if the focus list is empty, if this
        unit is not currently in the focus list, or if the projection has not yet been computed. Principal component 
        analysis occurs in the background and can take many seconds to complete.
        """
        self._cached_similarity: Dict[str, float] = dict()
        """ Similarity of this neural unit to another, lazily computed and cached. Keyed by label of other unit. """

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
            return len(self.spike_times) / float(self.spike_times[-1] - self.spike_times[0])
        return 0.0

    def firing_rate_histogram(self, bin_size: int, dur: float, normalized: bool = True) -> np.ndarray:
        """
        Generate the histogram of firing rate in this unit over the duration of the electrophysiological recording.

        :param bin_size: Histogram bin size in seconds.
        :param dur: Recording duration in seconds.
        :param normalized: If True, histogram is normalized such that a bin value of 1 corresponds to the unit's overall
            mean firing rate; else each bin value is the observed number of spikes in that bin. Default is True.
        :return: Normalized firing rate histogram for the specified bin size, such that 1 corresponds to the unit's
            overall mean firing rate. The last partial bin, if any, is omitted.
        """
        out, _ = np.histogram(self.spike_times, bins=int(dur / bin_size), density=False)
        if normalized:
            out = out / (bin_size * self.mean_firing_rate_hz)
        return out

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
        for all recorded Omniplex analog channels.

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
        The peak-to-peak amplitude of the unit's mean spike waveform as measured on the primary channel -- ie, the
        analog channel exhibiting the best signal-to-noise ratio for this unit.

        :return: None if the per-channel mean spike waveforms have not yet been computed; otherwise, returns the
            peak-to-peak voltage of the mean spike waveform recorded on the primary channel. In microvolts.
        """
        if self._primary_channel is None:
            return None
        else:
            wv = self._templates.get(self._primary_channel)
            return float(np.max(wv) - np.min(wv))

    def similarity_to(self, other_unit: "Neuron") -> float:
        """
        Calculate a similarity metric comparing this unit to another one. The metric is based on the spike template
        waveforms of the two units. For each unit, a 1D sequence is formed by concatenating the per-channel spike
        template waveforms. The cross correlation coefficient for the two sequences is the similarity metric.

        :param other_unit: The neural unit to which this one is compared.
        :return: The similarity metric, as described. Always returns 1.0 if the `other_unit` refers to this unit. **If
        the per-channel spike template waveforms have not yet been computed for either unit, then this metric is not yet
        determined and 0.0 is returned.**
        """
        if other_unit.label == self.label:
            return 1.0
        elif other_unit.label in self._cached_similarity:
            return self._cached_similarity[other_unit.label]

        similarity = 0.0
        if len(self._templates) > 0 and len(other_unit._templates) > 0:
            x1 = np.array([])
            x2 = np.array([])
            channels = sorted([k for k in self._templates.keys()])
            if not all([(idx in other_unit._templates) for idx in channels]):
                return similarity
            for idx in channels:
                x1 = np.hstack((x1, self._templates[idx]))
                x2 = np.hstack((x2, other_unit._templates[idx]))
            cc = np.corrcoef(x1, x2)
            similarity = float(cc[0, 1])
            self._cached_similarity[other_unit.label] = similarity
            other_unit._cached_similarity[self.label] = similarity
        return similarity

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

    @property
    def cached_isi(self) -> np.ndarray:
        """
        The interspike interval histogram for this unit's spike train. The histogram is a 1D array of normalized bin
        counts for ISIs between 0 and M (in milliseconds) inclusive, where M is the maximum ISI considered. The
        histogram is normalized to unity (all bin counts divided by the maximum observed bin count).
            By design, the ISI is lazily created on a background thread when the unit is selected for display for the
        first time during application runtime. It is computed for a fixed span M=200ms and cached in this unit, where
        it remains until application exit.

        :return: A copy of the cached ISI histogram, or an empty array if not available.
        """
        return np.array([]) if self._cached_isi is None else np.copy(self._cached_isi)

    @property
    def cached_acg(self) -> np.ndarray:
        """
        The autocorrelogram for this unit's spike train, if available. The autocorrelogram is the cross-correlogram of
        the unit's spike train with itself. See :func:`get_cached_ccg()` for information on how CCGs are computed and
        normalized.
            By design, the ACG is lazily created on a background thread when the unit is selected for display for the
        first time during application runtime. It is computed for a fixed correlogram span of 200ms and cached in this
        unit, where it remains until application exit.

        :return: A copy of the cached ACG, or an empty array if not available.
        """
        return np.array([]) if self._cached_acg is None else np.copy(self._cached_acg)

    def get_cached_ccg(self, other_unit: str) -> np.ndarray:
        """
        Get the cross-correlogram for this unit's spike train vs the spike train of the unit specified, if avaiable. The
        correlogram is essentially a histogram of how likely a spike occurs in the second neuron at some time T before
        or after a spike occurred in this neuron at t=0. The histogram has 2S+1 1-ms bins between -S and S, inclusive,
        where S is the correlogram span. Element k contains the number of times a spike in the second unit occurred at
        at time lag/lead of k-S millisecconds WRT the occurendce of a spike in this unit, divided by the totol number of
        spikes from this unit's spike train included in the analysis (spikes within S of the start or end of the
        recording are excluded).
            By design, the CCGs are lazily created on a background thread when the unit is selected for display for the
        first time during application runtime. They are computed for a fixed correlogram span of 200ms and cached in
        this unit, where they remain until application exit.

        :param other_unit: Label uniquely identifying the second unit.
        :return: A copy of the cached CCG for this unit vs the unit specified, or an empty array if not available.
        """
        out = self._cached_ccgs.get(other_unit)
        return np.array([]) if out is None else np.copy(out)

    def cache_isi_if_necessary(self) -> bool:
        """
        Compute and cache the 200-ms ISI histogram for this unit if it has not already been computed.
        :return: True if ISI was computed, False if it was already cached.
        """
        if self._cached_isi is None:
            out = stats.generate_isi_histogram(self.spike_times, self.FIXED_HIST_SPAN_MS)
            max_count = max(out)
            if max_count > 0:
                out = out * (1.0 / max_count)
            self._cached_isi = out
            return True
        return False

    def cache_acg_if_necessary(self) -> bool:
        """
        Compute and cache the 200-ms autocorrelogram for this unit if it has not already been computed. **As the
        computation can take a significant amount of time, only invoke this method on a background thread.**
        :return: True if ACG was computed, False if it was already cached.
        """
        if self._cached_acg is None:
            out, n = stats.generate_cross_correlogram(self.spike_times, self.spike_times, self.FIXED_HIST_SPAN_MS)
            if n > 0:
                out = out * (1.0 / n)
            self._cached_acg = out
            return True
        return False

    def cache_ccg_if_necessary(self, other_unit: 'Neuron') -> bool:
        """
        If it is not already cached, compute and cache the 200-ms cross-correlogram for this unit vs the specified unit.
        **As the computation can take a significant amount of time, only invoke this method on a background thread.**
        :param other_unit: The other neural unit.
        :return: True if the CCG was computed, False if it was already cached.
        """
        ccg = self._cached_ccgs.get(other_unit.label)
        if ccg is None:
            out, n = stats.generate_cross_correlogram(self.spike_times, other_unit.spike_times, self.FIXED_HIST_SPAN_MS)
            if n > 0:
                out = out * (1.0 / n)
            self._cached_ccgs[other_unit.label] = out
            return True
        return False

    def cached_pca_projection(self) -> np.ndarray:
        """
        Get the projection of this unit's spike "clips" onto the 2D space defined by the first two principal components
        computed for the space of all mean spike template waveforms across all recorded analog channels for all neural
        units currently in the "focus list". Principal component analysis occurs in the background and typically takes
        many seconds to complete, so the projection is not immediately available and is reset whenever the focus list
        changes.

            In the context of PCA, a spike "clip" is the cancatentation of M 2-ms clips -- one for each recorded
        Omniplex analog channel -- centered on the spike's time of occurrence. The principal components are computed
        for a KxL matrix of mean spike template clips, where each row is the horizontal concatenation of the first 2ms
        of the spike templates computed on each of the M analog channels for a given unit in the focus list. Here L =
        M*P, where P is the number of analog samples in 2ms. PCA yields a Lx2 matrix containing the first 2 principal
        components (eigenvectors with the two highest eigenvalues). This matrix is used to compute the projection of
        ALL of a unit's spike clips (NxL) onto the 2D plane defined by these principal components (Nx2). Plotting these
        projections as scatter plots provides a visual indication of whether or not the units are truly distinct.

        :return: An Nx2 array representing the PCA projection for this unit, as described. If this unit is NOT in the
            current focus list or if the projection has not yet been computed, returns an empty array.
        """
        return np.empty((0, 2)) if self._cached_pca_projection is None else self._cached_pca_projection

    def set_cached_pca_projection(self, prj: Optional[np.ndarray] = None) -> None:
        """
        Cache the current PCA projection for this neural unit, or clear a previously computed projection.
            This method is intended only for cacheing the result of PC analysis in the background, or clearing a
        a previously cached projection that is no longer valid because the neural unit focus list has changed. Note
        that the background task may populate this array in chunks so that the GUI can be updated on the fly as the
        computation proceeds -- useful when a unit a very long spike train.

        :param prj: An Nx2 array representing the PCA projection for this unit's N spikes, or None to clear a
            previously cached projection.
        """
        self._cached_pca_projection = prj
