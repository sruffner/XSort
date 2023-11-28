"""
stats.py: A collection of functions implementing various statistical calculations

    NOTE: Copied from sglportalapi package on 10/2/2023.
"""
from typing import Tuple, Optional
import numpy as np


def generate_isi_histogram(spike_times: np.ndarray, max_isi_ms: int = 100) -> np.ndarray:
    """
    Generate an inter-spike interval (ISI) histogram for a specified series of spike times.

    Args:
        spike_times: Array of spike times in seconds. Need not be in chronological order.
        max_isi_ms: The maximum inter-spike interval M included in the histogram, in milliseconds. Default = 100. This
            parameter defines an array of evenly spaced times between 0 and M inclusive, representing the centers of the
            1ms histogram bins over which the ISIs are counted. Minimum value of 20 ms.

    Returns:
        The computed histogram as an array of counts per bin (same length as 'bin_centers').
    """
    max_isi_ms = max(20, max_isi_ms)
    # noinspection PyUnresolvedReferences
    bin_centers = np.linspace(0, max_isi_ms*1e-3, max_isi_ms + 1)
    half_width = 1e-3 / 2.0

    diff_times = np.diff(spike_times)
    diff_times = diff_times[~np.isnan(diff_times)]
    bin_values = np.zeros(len(bin_centers))
    if len(diff_times) > 0:
        for i in range(len(bin_centers)):
            select = np.logical_and(diff_times > (bin_centers[i] - half_width),
                                    diff_times <= (bin_centers[i] + half_width))
            bin_values[i] = len(select.nonzero()[0])
    return bin_values


def generate_cross_correlogram(
        spike_times_1: np.ndarray, spike_times_2: np.ndarray, span_ms: int = 100) -> Tuple[np.ndarray, int]:
    """
    Generate a cross correlogram for two series of spike times.

        **NOTE**: If the two spike trains are identical, the result is an auto-correlogram. In this case, by convention,
    the bin corresponding to a time delta of 0 is set to 0.

    Args:
        spike_times_1: Array of spike times for a neural unit, in seconds. Need not be in chronological order.
        spike_times_2: Array of spike times for a second neural unit, in seconds. Passing the same array into both
            arguments yields the auto-correlogram for a single neuron.
        span_ms: Correlogram span in milliseconds. Default = 100. Minimum value of 20.
    Returns:
        A 2-tuple (C, n). C is a Numpy array containing the computed correlogram, while n is the # of spikes in the
            first neuron that were included in the analysis (spikes within 'span_ms' of the beginning or end of the
            recorded time frame must be excluded). The correlogram is essentially a histogram of how likely a spike
            occurs in the second neuron at some time before or after a spike occurred in the first neuron at t=0. The
            histogram will have 2S+1 1-ms bins between -S and S, inclusive, where S is the specified correlogram span.
            Each element contains the count of unit 2 spikes in the corresponding time bin. To get a relative measure of
            the likelihood of a spike occurring in each bin, divide each bin count by n.
    Raises:
        ValueError: If either array of spike times is empty, or contains a non-finite value.
    """
    if (len(spike_times_1) == 0) or (len(spike_times_2) == 0):
        raise ValueError("Empty spike times array")

    is_acg = np.array_equal(spike_times_1, spike_times_2)

    # convert spike times to integer milliseconds
    spikes_ms_1 = np.ceil(spike_times_1 * 1000.0).astype(np.int64)
    spikes_ms_2 = np.ceil(spike_times_2 * 1000.0).astype(np.int64)

    # creaate a binned spike train for the second unit, with each 1-ms bin containing the # of spikes in that bin
    train_len = int(np.ceil(max(np.nanmax(spikes_ms_1, axis=0), np.nanmax(spikes_ms_2, axis=0))))
    spike_train_2 = np.zeros(train_len+1, dtype=np.uint8)

    for i in range(len(spikes_ms_2)):
        spike_train_2[spikes_ms_2[i]] += 1

    # compute the correlogram
    span_ms = max(20, span_ms)
    counts = np.zeros(span_ms * 2 + 1)
    n = 0
    for i, t_ms in enumerate(spikes_ms_1):
        start = t_ms - span_ms
        stop = t_ms + span_ms + 1
        # we omit spikes in unit 1 that are too close to the beginning or end of the recorded timeframe, since we
        # could miss spikes in unit 2 that are within the correlogram span.
        if (start >= 0) and (stop < train_len):
            counts += spike_train_2[start:stop]
            n = n + 1

    # set t=0 bin to 0 for ACG
    if is_acg:
        counts[span_ms] = 0

    return counts, n


def gen_cross_correlogram_vs_firing_rate(
        spike_times_1: np.ndarray, spike_times_2: np.ndarray, span_ms: int = 100, num_fr_bins: int = 10,
        smooth: Optional[float] = 250e-3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a series of cross-correlograms of the spike trains for two neural units across the observed instantaneous
    firing rates. The result is a 2D matrix: the first dimension is binned firing rate, while the second is time
    relative to the spike occurrence time at t=0, in one-millisecond bins.
        **NOTE**: If the two spike trains are identical, the result is an auto-correlogram of the unit's spikes across
    the range of observed instantaneous firing rates. This can be thought of as a 3D ACG that shows firing regularity
    when the unit is firing at different rates. In the ACG case, by convention, all firing rate bins for a time delta
    of 0 are set to 0.

    :param spike_times_1: Array of spike times for a neural unit, in seconds. Must be in chronological order.
    :param spike_times_2: Array of spike times for a second neural unit, in seconds. Passing the same array into both
        arguments yields the auto-correlogram of the spike train vs firing rate for a single unit.
    :param span_ms: Correlogram span in milliseconds. Default = 100. Minimum value of 20.
    :param num_fr_bins: Number of bins, N, for the firing rate axis. Default = 10.
    :param smooth: Width of the boxcar filter used to smooth the calculated instantaneous firing rate, in seconds. If
        None, firing rate is not filtered. Default = 0.25s.
    :return: 2-tuple (A, B), where A is a 1D Numpy array of length N holding the bin centers for the firing rate axis,
        and B is a NxM matrix holding the result, where M = span_ms*2 + 1.
    :raises ValueError: If there are fewer than 2 spikes in either spike train, or if either spike train contains a
        non-finite (NaN, infinite) value.
    """
    if (len(spike_times_1) < 2) or (not np.all(np.isfinite(spike_times_1))) or \
            (len(spike_times_2) < 2) or (not np.all(np.isfinite(spike_times_2))):
        raise ValueError("Non-finite spike time, or not enough spike times")

    is_acg = np.array_equal(spike_times_1, spike_times_2)

    # convert spike times to integer milliseconds
    spikes_ms_1 = np.ceil(spike_times_1 * 1000.0).astype(np.int64)
    spikes_ms_2 = np.ceil(spike_times_2 * 1000.0).astype(np.int64)

    # creaate a binned spike train for the second unit, with each 1-ms bin containing the # of spikes in that bin
    train_len = int(np.ceil(max(np.nanmax(spikes_ms_1, axis=0), np.nanmax(spikes_ms_2, axis=0))))
    spike_train_2 = np.zeros(train_len+1, dtype=np.uint8)
    for i in range(len(spikes_ms_2)):
        spike_train_2[spikes_ms_2[i]] += 1

    # compute instantaneous firing rate for the second unit using the inverse ISI method.
    firing_rate = np.zeros(spike_train_2.shape)
    for i in range(len(spikes_ms_2)-1):
        t0_ms, t1_ms = spikes_ms_2[i], spikes_ms_2[i+1]
        if t0_ms == t1_ms:
            continue
        current_fr = 1000.0 / (t1_ms - t0_ms)
        firing_rate[int(t0_ms):int(t1_ms)] = current_fr
        if i == 0:
            firing_rate[0:int(t0_ms)] = current_fr
        elif i == len(spikes_ms_2) - 2:
            firing_rate[int(t1_ms):] = current_fr

    # smooth the instantaneous firing rate waveform if requested
    if isinstance(smooth, float):
        width = int(smooth * 1000.0)   # smoothing width in millisecs
        half_width = round(width / 2)
        n = len(firing_rate)
        firing_rate = np.convolve(np.ones(width) / width, firing_rate)[half_width:half_width+n]

    # get the IFR for the second unit at each spike occurrence time for the first unit, then determine the N bins
    # for the list of rates
    quantile_probabilities = [i/(num_fr_bins+1) for i in range(1, num_fr_bins+1)]
    rates = firing_rate[spikes_ms_1]
    firing_rate_axis = np.quantile(rates, quantile_probabilities)

    # prepare the correlogram vs firing rate matrix: for every spike occurrence time T in the first unit, get the firing
    # rate in the second unit at that time and locate the corresponding firing rate bin. Increment the spike counts for
    # that bin IAW the spikes found in the segment [T-span_ms:T+span_ms] in the second unit's binned spike train...
    counts = np.zeros(shape=(len(firing_rate_axis), 2*span_ms+1))
    num_per_bin = np.zeros(len(firing_rate_axis))
    for i, t_ms in enumerate(spikes_ms_1):
        start = int(t_ms) - span_ms
        stop = start + 2 * span_ms + 1
        if (start < 0) or stop >= train_len:  # skip spikes too close to the beginning or end of the spike train
            continue
        current_fr = firing_rate[t_ms]
        # omit any spike where the corresponding firing rate is outside the bounds of the firing rate axis
        if (current_fr > firing_rate_axis[-1] + (firing_rate_axis[-1] - firing_rate_axis[-2]) / 2.0) or \
                (current_fr < firing_rate_axis[0] - (firing_rate_axis[1] - firing_rate_axis[0]) / 2.0):
            continue
        bin_num = int(np.argmin(np.abs(firing_rate_axis - current_fr)))
        counts[bin_num, :] += spike_train_2[start:stop]
        num_per_bin[bin_num] = num_per_bin[bin_num] + 1

    # normalize by dividing by the number of samples contributing to each firing rate bin
    counts = counts / num_per_bin[:, np.newaxis]

    # for ACG case, set to 0 at spike occurrence time (t=0) in each firing rate bin
    if is_acg:
        counts[:, span_ms] = 0

    return firing_rate_axis, counts


def compute_principal_components(samples: np.ndarray, num_cmpts: int = 2) -> np.ndarray:
    """
    Given N samples of M variables, compute the first P principal components of the data set.

        Each column of the NxM matrix is interpreted as N samples of the "variable" represented by that column. The
    matrix is "standardized" by subtracting the sample mean from each value and dividing by the sample variance. The MxM
    covariance matrix is then computed, and then the eigenvalues and eigenvectors for that matrix. The eigenvectors
    represent the principal components for the sample set, and the corresponding eigenvalues indicate the amount of
    variance along each eigenvector. The method returns an M x P matrix, where the i-th column is the eigenvector with
    the i-th largest variance (eigenvalue).

    :param samples: An NxM matrix containing N samples of M variables; N,M >= 2.
    :param num_cmpts: The number of principal components to be computed. Maximum of 10.
    :return: An MXP matrix containing the first P principal components (in order of decreasing variance) of the
        M variables. In this form, multiplying the original NxM matrix by this matrix "reduces the dimensionality" of
        the data set from M to P.
    """
    if (len(samples.shape) != 2) or (samples.shape[0] < 2) or (samples.shape[1] < 2):
        raise Exception("Input matrix must be NxM with N >= 2 and M >= 2.")
    num_cmpts = max(1, min(num_cmpts, 10))
    num_cmpts = min(num_cmpts, min(samples.shape))
    std_samples = (samples - np.mean(samples, axis=0)) / np.std(samples, axis=0)
    covariance = np.cov(std_samples, rowvar=False)
    res = np.linalg.eig(covariance)
    desc_var_indices = np.flip(np.argsort(res.eigenvalues))
    return np.real(res.eigenvectors[:, desc_var_indices[0:num_cmpts]])
