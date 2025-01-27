from enum import Enum
from typing import Optional, Dict, Tuple, List, Any

import numpy as np


class DataType(Enum):
    """ The different types of data objects generated/retrieved by :class:`Analyzer` via background tasks. """
    NEURON = 1,   # neural unit record
    CHANNELTRACE = 2,   # analog channel traces for a 1-second segment of the analog recording
    ISI = 3,   # ISI histogram for a neural unit in the focus list
    ACG = 4,   # Autocorrelogram for a neural unit in the focus list
    ACG_VS_RATE = 5,  # 3D autocorrelogram vs firing rate for a neural unit in the focus list
    CCG = 6,   # Crosscorrelogram for a neual unit vs another unit in the focus list
    PCA = 7    # PCA projection for a neural unit in the focus list

    @staticmethod
    def is_unit_stat(dt: 'DataType') -> bool:
        """
        Does this data type represent one of the computed statistics for a neural unit?
        :param dt: The data type.
        :return: True for a unit statitics data type; false for a neural unit record or channel trace.
        """
        return dt in _UNIT_STATS


_UNIT_STATS = (DataType.ISI, DataType.ACG, DataType.ACG_VS_RATE, DataType.CCG, DataType.PCA)

MAX_CHANNEL_TRACES: int = 16
""" 
Maximum number of analog channels displayable in XSort at any one time. When the total number of recorded channels 
exceeds this limit, the set of displayable channels is centered around the primary channel for the first unit in the 
current unit focus list. Otherwise, all recorded channels are displayable. **This is also the maximum number of 
per-channel templates computed for each neural unit.**
"""


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

    @property
    def amplitude_in_microvolts(self) -> float:
        """ Peak-to-peak amplitude of the channel trace segmennt, in microvolts. """
        return (np.max(self._seg) - np.min(self._seg)) * self._scale


class Neuron:
    """
    A container holding the spike train, computed metrics, important metadata, and cached statistics for a neural unit.

        When initially constructed from the spike sorter results file in the XSort working directory, the only
    available information about a neural unit is its label and spike train. Metadata and metrics are computed in the
    background and stored, along with the spike train, in an internal cache file in the working directory. Additional
    statistics -- the autocorrelogram, autocorrelogram as a function of firing rate, interspike interval histogram,
    cross correlograms with other units, and principal component analysis results -- are computed at application runtime
    and cached in this object.
        Principal component analysis is performed on the spike trains of the neural units in the current "focus list".
    The result of PCA is a projection of each unit's spikes into a 2D space defined by the 2 principal components
    exhibiting the most variance (aka, the most "information"), and it is this projection which is cached in this
    object. Of course, each time the "focus list" changes, the analysis must be redone, so any previously cached
    projection is reset.
    """
    FIXED_HIST_SPAN_MS: int = 200
    """ The +/- time span (ms) for most neural unit histograms (ISI, ACG, CCG). """
    ACG_VS_RATE_SPAN_MS: int = 100
    """ The +/- time span (ms) for the neural unit 3D histogram representing ACG as a function of firing rate. """
    MAX_LABEL_LEN: int = 25
    """ Maximum length of any user-defined label attached to a neural unit. """

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
    def dissect_uid(uid: str) -> Tuple[int, str]:
        """
        By design, the UID uniquely identifying a neural unit in XSort has four possible forms: a simple integer
        string for non-Purkinje units, 'Nx' for a unit created as the result of merging two units or splitting another
        unit, 'Nc' for a unit representing the complex spikes from a Purkinje cell assigned the integer index N, and
        'Ns' for a unit representing the simple spikes from that same Purkinje cell.

        This method dissects the label to extract the integer N and a suffix:
         - 'c': Indicates the unit is defined in the original spike sorter file and represents the complex spike train
           of a putative Purkinje cell.
         - 's': Indicates the unit is defined in the original spike sorter file and represents the simple spike train of
           a putative Purkinje cell.
         - 'x': Indicates unit is NOT found in the original spike sorter file but is the result of merging two units or
           splitting a unit.
         - '' (no suffix): Indicates the unit is defined in the original spike sorter file and represent a non-Purkinje
           cell.

        :return: A 2-tuple (N, suffix), as described above.
        :raises Exception: If the UID string is invalid.
        """
        index_str, suffix = uid, ''
        if uid.endswith(('c', 's', 'x')):
            index_str = uid[:-1]
            suffix = uid[-1]
        if not index_str.isdigit():
            raise Exception("Invalid unit UID")
        return int(index_str), suffix

    @staticmethod
    def is_derived_uid(uid: str) -> bool:
        """
        Does the specified string match the UID of a derived neural unit, created by a merge or split? The UID of
        any derived unit has an integer index followed by the letter 'x'.
        """
        try:
            index, suffix = Neuron.dissect_uid(uid)
            return suffix == 'x'
        except Exception:
            return False

    @staticmethod
    def merge(unit1: 'Neuron', unit2: 'Neuron', idx: int) -> 'Neuron':
        """
        Create a new neural unit that merges the spike trains of the two specified units, with the merged spike times
        re-sorted in in chronological order.
        :param unit1: A neural unit.
        :param unit2: A second neural unit.
        :param idx: The index assigned to the merged unit. The unit's UID will have the form f"{idx}x", where the 'x'
            indicates the unit is the result of a merge (or split).
        :return: The merged unit.
        """
        merged_spikes = np.concatenate((unit1.spike_times, unit2.spike_times))
        merged_spikes.sort()
        return Neuron(idx, merged_spikes, suffix='x')

    def __init__(self, idx: int, spike_times: np.ndarray, suffix: str = ''):
        """
        Initialize a neural unit.

        :param idx: An integer index assigned to unit. For neural units obtained from the original spike sorter results
            file, this should be the ordinal index (starting from 1) of the unit. For units created by merging other
            units or splitting one unit into two populations of spikes, this index will be larger than that of the last
            unit extracted from the original spike sorter file.
        :param spike_times: The unit's spike train, ie, an array of spike occurrence times **in seconds elapsed since
            the start of the electrophysiological recording**.
        :param suffix: A single-character suffix appended to unit index to form the unit UID. Must be one of: 'c' (the
            complex spikes train of a Purkinje unit defined in the original spike sorter file); 's' (the simple spikes
            train of a Purkinje unit defined in the original spike sorter file); 'x' (a unit which is the result of
            merging two units or splitting a unit into two populations of spikes); or '' (a non-Purkinje unit from the
            original spike sorter file. Any other value is treated as ''.
        """
        self._idx = idx
        """
        The integer index assigned to unit. Note that a Purkinje cell has two associated units, which willl have the
        same index -- so this does not uniquely identify a unit.
        """
        if not (suffix in ['c', 's', 'x']):
            suffix = ''
        self._uid = f"{str(idx)}{suffix}"
        """ 
        A short string which uniquely identifies the unit and has one of four forms: 'N' for a non-Purkinje unit defined
        in the original spike sorter file; 'Ns' for the simple spikes of a Purkinje cell in the original spike sorter
        file, 'Nc' for the complex spikes of the same Purkinje cell, and 'Nx' for a unit that is the result of merging 
        two units or splitting a unit -- where N is the assigned integer index.
        """
        self._label = ""
        """ 
        A short user-defined label attached to the unit -- typically the putative neuron type. Restricted in 
        length. May contain spaces but no commas. Initially an empty string when unit is "created".
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
        """ Similarity of this neural unit to another, lazily computed and cached. Keyed by UID of other unit. """
        self._cached_acg_vs_rate: Optional[Tuple[np.ndarray, np.ndarray]] = None
        """ 
        Cached 3D histogram representing the autocorrelogram as a function of firing rate for this unit. The 3D
        histogram is represented by a 2-tuple of Numpy arrays. The first is a 10x1 vector specifying the bin centers
        for the firing rate axis, while the second is a 10x201 matrix defining the autocorrelogram in each firing
        rate bin. The second dimension of this matrix is time relative to the spike occurrence at T=0, ranging from -100
        to 100 milliseconds. Lazily created on a background thread when needed.
        """

    @property
    def uid(self) -> str:
        """
        A short string uniquely identifying this neural unit. For non-Purkinje cell units defined in the original spike
        sorter file, this is simply an integer index (starting from 1) assigned to the unit. For a neural unit
        representing the simple or complex spike train of a Purkinje cell defined in the original spike sorter file,
        the character 's' or 'c' is appended to that index. For a unit created by merging two units or splitting a unit,
        the character 'x' is appended.
        """
        return self._uid

    @property
    def index(self) -> int:
        """ The integer index assigned to this neural unit. """
        return self._idx

    @property
    def label(self) -> str:
        """
        An arbitrary, user-defined label attached to the unit, typically the putative neuron type. Will be an empty
        string if no label has been assigned to this unit.
        """
        return self._label

    @label.setter
    def label(self, s: str) -> None:
        """
        Specify a label for the unit.

        :param s: The candidate label. No commas allowed. May not exceed 25 characters in length after leading and
            trailing whitespace removed.
        :raises ValueError: If candiaate label is invalid.
        """
        if not isinstance(s, str):
            raise ValueError('Unit label must be a string')
        s = s.strip()
        if not Neuron.is_valid_label(s):
            raise ValueError('Unit label too long or contains a comma')
        self._label = s

    @staticmethod
    def is_valid_label(s: str) -> bool:
        return isinstance(s, str) and (0 <= len(s) <= Neuron.MAX_LABEL_LEN) and (s.find(',') < 0)

    @property
    def is_purkinje(self) -> bool:
        """
        True if this neural unit represents the simple or complex spike train of a Purkinje cell defined in the
        original spike sorter file.
        """
        return (self._uid.find('c') > 0) or (self._uid.find('s') > 0)

    @property
    def is_mod(self) -> bool:
        """ True if this neural unit was the result of merging two units or spltting a unit. """
        return self._uid.find('x') > 0

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
        out = out / (bin_size * (self.mean_firing_rate_hz if normalized else 1))
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

        By design, when there are more than 16 analog channels, XSort only computes templates for the 16 channels in
        the neighborhood of a unit's "primary channel". In this scenario, the similarity metric between two units is
        only computed using templates computed on the channels comprising the intersection of the units' template
        channel indices.

        :param other_unit: The neural unit to which this one is compared.
        :return: The similarity metric, as described. Always returns 1.0 if the `other_unit` refers to this unit. **If
        the per-channel spike template waveforms have not yet been computed for either unit, then this metric is not yet
        determined and 0.0 is returned.**
        """
        if other_unit.uid == self.uid:
            return 1.0
        elif other_unit.uid in self._cached_similarity:
            return self._cached_similarity[other_unit.uid]

        similarity = 0.0
        if len(self._templates) > 0 and len(other_unit._templates) > 0:
            ch_set = {k for k in self._templates.keys()} & {k for k in other_unit._templates.keys()}
            if len(ch_set) == 0:
                return similarity
            x1 = np.array([])
            x2 = np.array([])
            channels = sorted([k for k in ch_set])
            for idx in channels:
                x1 = np.hstack((x1, self._templates[idx]))
                x2 = np.hstack((x2, other_unit._templates[idx]))
            cc = np.corrcoef(x1, x2)
            similarity = float(cc[0, 1])
            self._cached_similarity[other_unit.uid] = similarity
            other_unit._cached_similarity[self.uid] = similarity
        return similarity

    @property
    def num_templates(self) -> int:
        """
        The number of spike templates computed for this unit. Returns 0 if the templates have not yet been computed for
        this unit (that computation happens on a background task whenever the XSort working directory changes or a new
        neural unit is defined).
        :return: The number of per-channel spike template waveforms
        """
        return len(self._templates)

    @property
    def template_length(self) -> int:
        """
        The length of any of the unit's spike template waveforms (they are all the same length). Returns 0 if the
        templates have not yet been computed for this unit.
        """
        return 0 if (self.primary_channel is None) else len(self._templates[self.primary_channel])

    @property
    def template_channel_indices(self) -> List[int]:
        """
        List of indices identifying Omniplex analog data channels on which spike template waveforms were calculated.
        Will return an empty list if the templates have not yet been computed for this unit.
        """
        return [k for k in self._templates.keys()]

    def get_template_for_channel(self, idx: int) -> Optional[np.ndarray]:
        """
        Get this unit's mean spike waveform, or 'template', as computed from the data stream on the specified Omniplex
        analog data channel.

        :param idx: Index of an Omniplex analog channel.
        :return: None if the channel index is invalid or was not found in the Omniplex recording file, or if the mean
            spike waveform has not yet been computed for that channel (that computation happens on a background task
            whenever the XSort working directory changes or a new neural unit is defined). Otherwise, a Numpy array
            containing the 10-ms mean spike waveform as recorded on the specified channel, in microvolts. The number
            of samples in the array will depend on the analog channel sampling rate: R = len(template)/0.01s.
        """
        out = self._templates.get(idx)
        return out

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

    @property
    def cached_acg_vs_rate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        The autocorrelogram for this unit's spike train as a function of firing rate -- a measure of firing regularity
        when the unit is firing at different rates.
            By design, the ACG-vs-rate "3D histogram" is lazily created on a background thread when the unit is first
        selected for display during application runtime. It is computed for a fixed correlogram span of 200ms and 10
        firing rate bins. Once computed, it is cached in this object, where it remains until application exit.

        :return: A 2-tuple of Numpy arrays: The first is a 10x1 vector specifying the bin centers for the firing rate
            axis (in Hz), while the second is a 10x201 matrix containing the computed autocorrelogram for each firing
            rate bin. The second dimension is the correlogram span [-100..100] milliseconds, that is, time relative to
            the occurence of a spike at T=0. Returns an independent copy of the 2-tuple. Both arrays in the tuple will
            be empty if the ACG-vs-rate histogram is not yet available.
        """
        return (np.array([]), np.array([])) if self._cached_acg_vs_rate is None \
            else (np.copy(self._cached_acg_vs_rate[0]), np.copy(self._cached_acg_vs_rate[1]))

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

    @property
    def is_pca_projection_cached(self) -> bool:
        """
        Is the PCA projection for this neural unit cached in its entirety in this object? Since the PCA projection can
        take a significant amount of time to compute, it may be cached in chunks.

        **NOTE**: The length of the complete PCA projection will not necessarily match the length of the unit's spike
        train; spikes too close to the beginning or end of the analog recording are omitted from the analysis.
        Therefore, this  method indicates the projection is fully cached so long as the projection length is within 5
        of the spike train length.
        :return: True if the PCA projection for this unit is cached in its entirety.
        """
        n = len(self.cached_pca_projection())
        return self.num_spikes - 5 < n <= self.num_spikes

    def clear_cached_pca_projection(self) -> None:
        """ Clear the cached PCA projection for this neural unit, if any. """
        self._cached_pca_projection = None

    def is_statistic_cached(self, dt: DataType, uid_other: Optional[str] = None) -> bool:
        """
        Has the specified neural unit statistic been computed and cached in this :class:`Neuron` object?
        :param dt: Data type identifying one of the unit statistics.
        :param uid_other: For crosscorrelogram only, the UID of the other unit.
        :return: True if requested statistic is cached here; else False.
        """
        if dt == DataType.ISI:
            return not (self._cached_isi is None)
        elif dt == DataType.ACG:
            return not (self._cached_acg is None)
        elif dt == DataType.ACG_VS_RATE:
            return not (self._cached_acg_vs_rate is None)
        elif dt == DataType.CCG:
            return not (self._cached_ccgs.get(uid_other, None) is None)
        elif dt == DataType.PCA:
            return self.is_pca_projection_cached
        else:
            return False

    def cache_statistic(self, dt: DataType, params: Tuple[Any]) -> bool:
        """
        Cache one of the statistics for this neural unit that are generated on a background task in XSort.

        The :class:`Neuron` object serves as a runtime cache for various statistics that may be displayed in XSort:
        the interspike interval histogram (DataType.ISI), autocorrelogram of the unit's spike train (ACG), 3D
        autocorrelogram as a function of firing rate (ACG_VS_RATE), crosscorrelogram of the spike trains of two
        distinct units (CCG), and the projection of a unit's spikes onto the 2D space defined by a principal component
        analysis of the units comprising the current focus list (PCA). These various statistics get computed in the
        background on an as-needed basis, and then cached in the relevent Neuron object by calling this function. With
        the exception of the PCA projection, the various statistics remain cached in the neural unit object until it
        is destroyed. PCA projections are cleared whenever the current focus list changes, since the principal
        component analysis is based on the units comprising that list.

        The data tuple provided has the exact form prepared by the background task:
         - ISI: (UID, S). S is a 1D Numpy array holding the interspike interval histogram for unit UID.
         - ACG: (UID, S). S is a 1D Numpy array holding the autocorrelogram for unit UID.
         - ACG_VS_RATE: (UID, (B, S)). B is a 10x1 Numpy array of firing rate bins, and S is the autocorrelogram of
           unit UID for each firing rate bin in B.
         - CCG: (UID, UID2, S). S is a 1D Numpy array holding the correlogram of unit UID with UID2 (not the same).
         - PCA: (UID, K, P). P is a Nx2 Numpy array holding a "chunk" of the PCA projection for unit UID for spikes
           K:K+N. The PCA projection takes a relatively long time to compute -- especially for 100K+ spikes -- so it
           is computed and delivered in chunks to allow progressive updates in XSort.

        :param dt: The data type -- must be one of DataType.ISI, ACG, ACG_VS_RATE, CCG, or PCA.
        :param params: A 2- or 3-tuple containing the statistic as described. In all cases, the first element should be
            the UID of this neural unit.
        :return: True if statistic was successfully cached; False if **params** argument is invalid or does not
            contain statistics for this unit.
        """
        try:
            assert (2 <= len(params) <= 3)
            assert (params[0] == self._uid)
            assert DataType.is_unit_stat(dt)
            if dt == DataType.ISI:
                assert isinstance(params[1], np.ndarray)
                self._cached_isi = np.copy(params[1])
            elif dt == DataType.ACG:
                assert isinstance(params[1], np.ndarray)
                self._cached_acg = np.copy(params[1])
            elif dt == DataType.ACG_VS_RATE:
                assert isinstance(params[1][0], np.ndarray) and isinstance(params[1][1], np.ndarray)
                self._cached_acg_vs_rate = np.copy(params[1][0]), np.copy(params[1][1])
            elif dt == DataType.CCG:
                assert isinstance(params[1], str) and isinstance(params[2], np.ndarray)
                self._cached_ccgs[params[1]] = np.copy(params[2])
            else:  # PCA
                # IMPORTANT: For PCA, some spikes at beginning or end of recording may be omitted -- so we need to
                # be careful with the asserts here!
                assert isinstance(params[1], int) and isinstance(params[2], np.ndarray)
                arr: np.ndarray = np.copy(params[2])
                if self._cached_pca_projection is None:
                    assert len(arr) <= self.num_spikes
                    self._cached_pca_projection = np.copy(arr)
                else:
                    assert params[1] == len(self._cached_pca_projection) and (params[1] + len(arr) <= self.num_spikes)
                    self._cached_pca_projection = np.vstack((self._cached_pca_projection, arr))
            return True
        except Exception:
            return False
