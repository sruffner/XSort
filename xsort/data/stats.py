"""
stats.py: A collection of functions implementing various statistical calculations

    NOTE: Copied from sglportalapi package on 10/2/2023.
"""
from typing import Tuple
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
        ValueError: If either array of spike times is empty.
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


def compute_principal_components(samples: np.ndarray, num_cmpts: int = 2) -> np.ndarray:
    """
    Given N samples of M variables, compute the first P principal components of the data set.

        Each column of the NxM matrix is interpreted as N samples of the "variable" represented by that column. The
    matrix is "standardized" by subtracting the sample mean from each value and dividing by the sample variance. The MxM
    covariance matrix is then computed, and then the eigenvalues and eigenvectors for that matrix. The eigenvectors
    represent the principal components for the sample set, and the corresponding eigenvalues indicate the amount of
    variance along each eigenvector. The method returns an M x P matrix, where the i-th column is the eigenvector with
    the i-th largest variance (eigenvalue).

    :param samples: An NxM matrix containing N samples of M variables.
    :param num_cmpts: The number of principal components to be computed. Maximum of 10.
    :return: An MXP matrix containing the first P principal components (in order of decreasing variance) of the
        M variables. In this form, multiplying the original NxM matrix by this matrix "reduces the dimensionality" of
        the data set from M to P.
    """
    num_cmpts = max(1, min(num_cmpts, 10))
    num_cmpts = min(num_cmpts, min(samples.shape))
    std_samples = (samples - np.mean(samples, axis=0)) / np.std(samples, axis=0)
    covariance = np.cov(std_samples, rowvar=False)
    res = np.linalg.eig(covariance)
    desc_var_indices = np.flip(np.argsort(res.eigenvalues))
    return np.real(res.eigenvectors[:, desc_var_indices[0:num_cmpts]])   # TODO: OK to discard imag part??
