"""
Copyright (c) 2018 David Herzfeld

Written by David J. Herzfeld <herzfeldd@gmail.com>

This module contains functions written by DH to parse the contents of a Plexon/Omniplex PL2 data file. In the original
code, the load functions take the filename of the PL2 file to be read. Here, the functions have been adapted to take a
a Python file-like object so that they can be used with an already open file handle. All functions leave that file
handle open on return.

The information returned by load_file_information() is essentially a "table of contents" for the entire PL2 file, and
we can save time if we supply it to the various load functions rather than re-loading that information on each load()
call.

Unlike the original implementation, the load methods DO NOT accumulate channel data in the 'info' dictionary object
that is initially set up by load_file_information(), as that could consume significant memory resources. For example,
consider a large Omniplex file containing 200,000,000 samples recorded on each of 32 channels!

I have also made some cosmetic changes such as docstrings and some type annotations.

25jul2022 - Updated to reflect changes David H made to handle PL2 software version 1.19.2
03aug2022 - Further changes to fix _get_channel_offset()
"""

import struct
import os

import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, List, IO


# Define the constants for this module
# Constants for data blocks
PL2_DATA_BLOCK_ANALOG_CHANNEL = 0x42
PL2_DATA_BLOCK_EVENT_CHANNEL = 0x5A
PL2_DATA_BLOCK_SPIKE_CHANNEL = 0x31
PL2_DATA_BLOCK_START_STOP_CHANNEL = 0x59

# Constants for footers
PL2_FOOTER_ANALOG_CHANNEL = 0xDD
PL2_FOOTER_EVENT_CHANNEL = 0xDF
PL2_FOOTER_SPIKE_CHANNEL = 0xDE
PL2_FOOTER_START_STOP_CHANNEL = 0xE0

# Constants for headers
PL2_HEADER_ANALOG_CHANNEL = 0xD4
PL2_HEADER_SPIKE_CHANNEL = 0xD5
PL2_HEADER_EVENT_CHANNEL = 0xD6

# Data sources/subtypes
PL2_ANALOG_TYPE_WB = 0x03   # Wide band analog
PL2_ANALOG_TYPE_AI = 0x0C   # Prior to PL v1.19, this was the source for 'AI' channels. For v1.19+, it's "Cineplex Data"
PL2_ANALOG_TYPE_AI2 = 0x0D  # As of PL v1.19, this is the source for 'AI' channels.
PL2_ANALOG_TYPE_FP = 0x07
PL2_ANALOG_TYPE_SPKC = 0x04   # Narrow band analog
PL2_EVENT_TYPE_KBD = 0x08  # Keyboard events
PL2_EVENT_TYPE_SINGLE_BIT = 0x09  # TTL events
PL2_EVENT_TYPE_STROBED = 0x0A  # Source name = 'Other Events': 'Strobed', 'RSTART', and 'RSTOP'
PL2_EVENT_TYPE_CINEPLEX = 0x0C  # "Cineplex Data" event channels introduced in PL v1.19 (?)
PL2_SPIKE_TYPE_SPK = 0x06
PL2_SPIKE_TYPE_SPK_SPKC = 0x01


def load_file_information(fp: IO) -> Dict[str, Any]:
    """
    Loads metadata from the header, various channel subheaders, and the footer of a PL2 file. This information can be
    passed to other loading functions to locate the data for a particular channel.

    Args:
        fp: The PL2 file object. The file must be open and is NOT closed on return.

    Returns:
        Dictionary containing the metadata culled from the PL2 file.
    """
    data: Dict[str, Any] = dict()
    # read the header from the file (storing contents in the dictionary)
    _read_header(fp, data)

    # Create empty start/stop channel values
    data["start_stop_channels"] = {}
    data["start_stop_channels"]["block_offsets"] = []
    data["start_stop_channels"]["block_timestamps"] = []
    data["start_stop_channels"]["block_num_items"] = []
    data["start_stop_channels"]["num_events"] = 0

    fp.seek(0x480)
    data["spike_channels"] = []
    for i in range(0, data["total_number_of_spike_channels"]):
        data["spike_channels"].append(_read_spike_channel_header(fp))

    data["analog_channels"] = []
    for i in range(0, data["total_number_of_analog_channels"]):
        data["analog_channels"].append(_read_analog_channel_header(fp))

    data["event_channels"] = []
    for i in range(0, data["number_of_event_channels"]):
        data["event_channels"].append(_read_event_channel_header(fp))

    if data["internal_value_4"] != 0:
        _read_footer(fp, data)
    else:
        _reconstruct_footer(fp, data)

    return data


def load_analog_channel(fp: IO, channel: int, info: Dict[str, Any] = None,
                        scale: bool = False) -> np.ndarray:
    """
    Loads recorded data for a specified analog channel in a PL2 file.

    Args:
        fp: The PL2 file object. The file must be open and is NOT closed on return.
        channel: The analog channel index (zero-based).
        info: Dictionary containing "table of contents" information needed to locate channel data. If None,
            load_file_information() is called first to load the table of contents.
        scale: If True, returns the results as an array of single precision floating point numbers, appropriately scaled
            by the conversion factor specified in the file header and then converted to millivolts. Otherwise, the raw
            unscaled ADC data is returned. Defaults to False.

    Returns:
        A Numpy array of the analog channel data, optionally scaled to millivolts.

    Raises:
        A generic Exception if analog data not found for specified channel or file contains invalid data
    """
    if info is None:
        info = load_file_information(fp)

    # Ensure that the appropriate analog channel exists and there is data there
    if channel >= len(info["analog_channels"]) or channel < 0:
        raise Exception(f"Invalid analog channel index: {channel}")

    if ("block_offsets" not in info["analog_channels"][channel]) or \
            (len(info["analog_channels"][channel]["block_offsets"]) == 0):
        raise Exception(f"Missing block information for analog channel index: {channel}")

    # Attempt to load the results
    total_items = sum(info["analog_channels"][channel]["block_num_items"])
    results = np.zeros(total_items, dtype=np.int16)
    for i in range(0, len(info["analog_channels"][channel]["block_offsets"])):
        block_offset = info["analog_channels"][channel]["block_offsets"][i]
        fp.seek(block_offset)
        _read(fp, "<B")   # data_type not used
        _read(fp, "<B")   # data_subtype not used

        num_items = _read(fp, "<H")
        if num_items != info["analog_channels"][channel]["block_num_items"][i]:
            raise RuntimeError(f"Invalid number of items encountered for analog channel index {channel}.")
        _read(fp, "<H")  # Channel
        _read(fp, "<H")  # Unknown
        timestamp = _read(fp, "<Q")  # Timestamp
        if timestamp != info["analog_channels"][channel]["block_timestamps"][i]:
            raise Exception(f"Invalid timestamp encountered for analog channel index {channel}")

        # _read each of the items
        values = _read(fp, "<{:d}h".format(num_items))
        start = sum(info["analog_channels"][channel]["block_num_items"][0:i])
        stop = start + num_items
        results[start:stop] = values

    if scale:
        results = results.astype(np.float32)
        results *= info["analog_channels"][channel]["coeff_to_convert_to_units"] * 1000  # to mV
    return results


def load_analog_channel_block(fp: IO, channel: int, block: int, info: Dict[str, Any]) -> np.ndarray:
    """
    Loads one block of samples for a specified analog data channel in a PL2 file.

    Args:
        fp: The PL2 file object. The file must be open and is NOT closed on return.
        channel: The analog channel index (zero-based).
        block: The block index (zero-based).
        info: Dictionary containing "table of contents" information needed to locate channel data -- as retrieved by
            load_file_information().

    Returns:
        A Numpy array of the analog channel data samples for the block specified (raw ADC samples; int16).

    Raises:
        A generic Exception if analog data not found for specified channel or file contains invalid data
    """
    # Ensure that the appropriate analog channel exists and there is data there for the block index specified.
    if channel >= len(info["analog_channels"]) or channel < 0:
        raise Exception(f"Invalid analog channel index: {channel}")
    if ("block_offsets" not in info["analog_channels"][channel]) or \
            (len(info["analog_channels"][channel]["block_offsets"]) == 0):
        raise Exception(f"Missing block information for analog channel index: {channel}")
    if block >= len(info["analog_channels"][channel]["block_offsets"]) or block < 0:
        raise Exception(f"Invalid block index ({block}) for analog channel index {channel}")

    block_offset = info["analog_channels"][channel]["block_offsets"][block]
    fp.seek(block_offset)
    _read(fp, "<B")  # data_type not used
    _read(fp, "<B")  # data_subtype not used

    num_items = _read(fp, "<H")
    if num_items != info["analog_channels"][channel]["block_num_items"][block]:
        raise Exception(f"Invalid number of items encountered for analog channel index {channel}, block={block}.")
    _read(fp, "<H")  # Channel
    _read(fp, "<H")  # Unknown
    timestamp = _read(fp, "<Q")  # Timestamp
    if timestamp != info["analog_channels"][channel]["block_timestamps"][block]:
        raise Exception(f"Invalid timestamp encountered for analog channel index {channel}, block={block}")

    return np.array(_read(fp, "<{:d}h".format(num_items)), dtype=np.int16)


def load_event_channel(fp: IO, channel: int, info: Dict[str, Any] = None,
                       scale: bool = False) -> Optional[Dict[str, np.ndarray]]:
    """
    Load all of the events from a from a given event channel.

    Args:
        fp: The PL2 file object. The file must be open and is NOT closed on return.
        channel: The event channel index (zero-based).
        info: Dictionary containing "table of contents" information needed to locate channel data. If None,
            load_file_information() is called first to load the file's table of contents.
        scale: If True, event timestamps are converted to seconds since the start of the recording; otherwise, they
            remain as integer tick counts. Defaults to False.

    Returns:
        A dictionary with two keys. "timestamps" is a Numpy array holding the event timestamps (in seconds if scale is
            True), and "strobed" is a Numpy array holding the corresponding event values. Returns None if event
            channel data not found.
    """
    if info is None:
        info = load_file_information(fp)

    # Ensure that the appropriate analog channel exists and there is data there
    if channel >= len(info["event_channels"]) or channel < 0:
        raise RuntimeError(f"Invalid event channel index: {channel}")

    if ("block_offsets" not in info["event_channels"][channel]) or \
            (len(info["event_channels"][channel]["block_offsets"]) == 0):
        return None

    # Attempt to load the results
    total_items = sum(info["event_channels"][channel]["block_num_items"])
    results = dict()
    results["timestamps"] = np.zeros(total_items, dtype=np.uint64)
    results["strobed"] = np.zeros(total_items, dtype=np.uint16)
    for i in range(0, len(info["event_channels"][channel]["block_offsets"])):
        block_offset = info["event_channels"][channel]["block_offsets"][i]
        fp.seek(block_offset)
        _read(fp, "<B")   # data_type not used
        _read(fp, "<B")   # data_subtype not used

        _read(fp, "<H")   # ??
        _read(fp, "<H")   # Channel not used
        num_items = _read(fp, "<Q")
        if num_items != info["event_channels"][channel]["block_num_items"][i]:
            raise RuntimeError(f"Invalid number of items encountered for event channel index {channel}")
        _read(fp, "<H")

        # _read each of the items
        start = sum(info["event_channels"][channel]["block_num_items"][0:i])
        stop = start + num_items
        results["timestamps"][start:stop] = _read(fp, "<{:d}Q".format(num_items))
        results["strobed"][start:stop] = _read(fp, "<{:d}H".format(num_items))

    if scale:
        results["timestamps"] = results["timestamps"].astype(np.float32)
        results["timestamps"] /= info["timestamp_frequency"]
    return results


def load_spike_channel(fp: IO, channel: int, info: Dict[str, Any] = None, scale: bool = False,
                       spike_number: Optional[int] = None) -> Optional[Dict[str, np.ndarray]]:
    """
    Load the timestamps and waveform clips for spikes that were identified/sorted in real time on a given spike channel
    during the PL2 recording. NOTE: Since researchers do their own spike sorting offline, and those spike sorting
    results must be supplied when committing an experiment session to the Lisberger lab database, we don't plan to use
    this function.

    Args:
        fp: The PL2 file object. The file must be open and is NOT closed on return.
        channel: The spike channel index (zero-based).
        info: Dictionary containing "table of contents" information needed to locate channel data. If None,
            load_file_information() is called first to load the file's table of contents.
        scale: If True, timestamps and spike waveform clips are scaled and converted to seconds (since the start of
            recording) and millivolts, respectively. Otherwise, they are left in their raw digitized form.
        spike_number: A given spike channel may contain multiple identified spike units -- numbered 0..3. If not None,
            then this function only returns the data for the specified spike unit, rather than for all units identified
            on the channel specified.
    Returns:
        A dictionary with 3 keys. "timestamps" is a Numpy array holding the spike timestamps (in seconds if scale is
            True), "assignments" specifies the assigned spike number for each individual spike, and "spikes" (confusing
            name) are small clips of the voltage waveform around each spike timestamp. The waveform clips will be in
            millivolts if scale is True.
    """
    if info is None:
        info = load_file_information(fp)

    # Ensure that the appropriate analog channel exists and there is data there
    if channel >= len(info["spike_channels"]) or channel < 0:
        raise RuntimeError(f"Invalid spike channel index: {channel}")

    if ("block_offsets" not in info["spike_channels"][channel]) or \
            (len(info["spike_channels"][channel]["block_offsets"]) == 0):
        return None

    # Attempt to load the results
    # noinspection PyUnresolvedReferences
    total_items = np.sum(info["spike_channels"][channel]["block_num_items"])
    results = dict()
    results["num_points"] = info["spike_channels"][channel]["samples_per_spike"]
    results["timestamps"] = np.empty(total_items, dtype=np.uint64)
    results["spikes"] = np.empty((total_items, results["num_points"]), dtype=np.int16)
    results["assignments"] = np.empty(total_items, dtype=np.uint16)

    for i in range(0, len(info["spike_channels"][channel]["block_offsets"])):
        block_offset = info["spike_channels"][channel]["block_offsets"][i]
        fp.seek(block_offset)
        _read(fp, "<B")   # data_type not used
        _read(fp, "<B")   # data_subtype not used

        _read(fp, "<H")   # ??
        _read(fp, "<H")   # Channel
        num_sample_points = _read(fp, "<H")
        if num_sample_points != info["spike_channels"][channel]["samples_per_spike"]:
            raise RuntimeError(f"Invalid number of samples per spike encountered for spike channel index {channel}:"
                               f" expected {info['spike_channels'][channel]['block_num_items']} but "
                               f"got {num_sample_points}")

        num_items = _read(fp, "<Q")
        if num_items != info["spike_channels"][channel]["block_num_items"][i]:
            raise RuntimeError(f"Invalid number of items encountered for spike channel index {channel}: expected "
                               f"{info['spike_channels'][channel]['block_num_items']} but got {num_items}")

        # read each of the items
        start = sum(info["spike_channels"][channel]["block_num_items"][0:i])
        stop = start + num_items
        results["timestamps"][start:stop] = _read(fp, "<{:d}Q".format(num_items))
        results["assignments"][start:stop] = _read(fp, "<{:d}H".format(num_items))
        for j in range(start, stop):
            results["spikes"][j, :] = _read(fp, "<{:d}h".format(results["num_points"]))

    if spike_number is not None:
        select = np.array(results["assignments"]) == spike_number
        results["timestamps"] = results["timestamps"][select]
        results["spikes"] = results["spikes"][select, :]
        results["assignments"] = results["assignments"][select]
    if scale:
        results["timestamps"] = results["timestamps"].astype(np.float32)
        results["timestamps"] /= info["timestamp_frequency"]
        results["spikes"] = results["spikes"].astype(np.float32)
        results["spikes"] *= info["spike_channels"][channel]["coeff_to_convert_to_units"] * 1000  # to mV
    return results


def _read(fp: IO, data_types: str, force_list=False) -> Union[Any, Tuple[Any], List[Any]]:
    """
    Read a series of bytes starting at the current file location and unpack them using struct.unpack(). This function
    serves to avoid needing a byte array that is exactly the same size of the size of the data type

    Args:
        fp: The PL2 file object. The file pointer advances to the byte after the series of bytes unpacked.
        data_types: A string defining how to unpack the bytes starting at the current file location.
        force_list: If True, the unpacked data is always returned as a list. Otherwise, returns a single unpacked value
            or a tuple (if more than one unpacked value).

    Returns:
        The unpacked value or values
    """
    num_bytes = struct.calcsize(data_types)
    _read_bytes = fp.read(num_bytes)
    values = struct.unpack(data_types, _read_bytes)
    if len(values) == 1:
        if force_list:
            return [values[0]]
        return values[0]
    else:
        if force_list:
            return list(values)
        return values


def _read_header(fp: IO, data: Dict[str, Any]) -> None:
    """
    Read the contents of the PL2 file header.

    Args:
        fp: The PL2 file object. Upon return, the file pointer should be positioned on the byte after the header.
        data: A dictionary in which the header information is stored.
    """
    fp.seek(0, os.SEEK_END)
    data["file_length"] = fp.tell()
    fp.seek(0)

    data["version"] = {}
    data["version"]["major_version"] = _read(fp, "<B")
    data["version"]["minor_version"] = _read(fp, "<B")
    data["version"]["bug_version"] = _read(fp, "<B")

    fp.seek(0x20)
    data["internal_value_1"] = _read(fp, "<Q")  # End of header
    data["internal_value_2"] = _read(fp, "<Q")  # First data block
    data["internal_value_3"] = _read(fp, "<Q")  # End of data blocks
    data["internal_value_4"] = _read(fp, "<Q")  # This is the start of footer
    data["start_recording_time_ticks"] = _read(fp, "<Q")
    data["duration_of_recording_ticks"] = _read(fp, "<Q")

    fp.seek(0xE0)
    data["creator_comment"] = bytearray(_read(fp, "<256B")).decode('ascii').split('\0', 1)[0]
    data["creator_software_name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    data["creator_software_version"] = bytearray(_read(fp, "<16B")).decode('ascii').split('\0', 1)[0]
    data["creator_date_time"] = _read_date_time(fp)
    data["timestamp_frequency"] = _read(fp, "<d")
    data["duration_of_recording_sec"] = data["duration_of_recording_ticks"] / data["timestamp_frequency"]
    _read(fp, "<I")  # Off by 4 bytes
    data["total_number_of_spike_channels"] = _read(fp, "<I")
    data["number_of_recorded_spike_channels"] = _read(fp, "<I")
    data["total_number_of_analog_channels"] = _read(fp, "<I")
    data["number_of_recorded_analog_channels"] = _read(fp, "<I")
    data["number_of_event_channels"] = _read(fp, "<I")
    data["minimum_trodality"] = _read(fp, "<I")
    data["maximum_trodality"] = _read(fp, "<I")
    data["number_of_non_omniplex_sources"] = _read(fp, "<I")

    fp.seek(4, os.SEEK_CUR)
    data["reprocessor_comment"] = bytearray(_read(fp, "<256B")).decode('ascii').split('\0', 1)[0]
    data["reprocessor_software_name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    # data["reprocessor_date_time"] = _read_date_time(fp)


def _read_date_time(fp: IO) -> Dict[str, int]:
    """
    Read a date/time structure starting at the current location in the file.

    Args:
        fp: The PL2 file object. Upon return, current location is on the byte immediately after the date/time structure.

    Returns:
        A dictionary holding the date/time structure, with fields 'second', 'minute', 'hour', 'month_day', 'month',
        'year', 'week_day', 'year_day', 'is_daylight_savings', and 'millisecond'. All integer values.
    """
    data = dict()
    data["second"] = _read(fp, "<I")
    data["minute"] = _read(fp, "<I")
    data["hour"] = _read(fp, "<I")
    data["month_day"] = _read(fp, "<I")
    data["month"] = _read(fp, "<I")
    data["year"] = _read(fp, "<I")
    data["week_day"] = _read(fp, "<I")
    data["year_day"] = _read(fp, "<I")
    data["is_daylight_savings"] = _read(fp, "<I")
    data["millisecond"] = _read(fp, "<I")
    return data


def _read_spike_channel_header(fp: IO) -> Dict[str, Any]:
    """
    Read the header for a spike data channel in the file, starting at the current file location.

    Args:
        fp: The PL2 file object. Upon return, current location is immediately after the spike channel header.

    Returns:
        A dictionary holding the contents of the spike channel header.
    """
    data = {}
    data_type = _read(fp, "<B")
    _read(fp, "<3B")  # data_subtype - not used
    if data_type != PL2_HEADER_SPIKE_CHANNEL:  # D5 06 08 05
        raise RuntimeError(f"Invalid type in spike channel header. Got {data_type}, "
                           f"expected {PL2_HEADER_SPIKE_CHANNEL}")

    data["plex_channel"] = _read(fp, "<I")

    # Two more empty items
    _read(fp, "<I")
    _read(fp, "<I")

    data["name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    data["source"] = _read(fp, "<I")
    data["channel"] = _read(fp, "<I")
    data["enabled"] = _read(fp, "<I")
    data["recording_enabled"] = _read(fp, "<I")
    data["units"] = bytearray(_read(fp, "<16B")).decode('ascii').split('\0', 1)[0]
    data["samples_per_second"] = _read(fp, "<d")
    data["coeff_to_convert_to_units"] = _read(fp, "<d")
    data["samples_per_spike"] = _read(fp, "<I")
    data["threshold"] = _read(fp, "<i")
    data["pre_threshold_samples"] = _read(fp, "<I")
    data["sort_enabled"] = _read(fp, "<I")
    data["sort_method"] = _read(fp, "<I")
    data["number_of_units"] = _read(fp, "<I")
    data["sort_range_start"] = _read(fp, "<I")
    data["sort_range_end"] = _read(fp, "<I")
    data["unit_counts"] = list(_read(fp, "<256Q"))
    data["source_trodality"] = _read(fp, "<I")
    data["trode"] = _read(fp, "<I")
    data["channel_in_trode"] = _read(fp, "<I")
    data["number_of_channels_in_source"] = _read(fp, "<I")
    data["device_id"] = _read(fp, "<I")
    data["number_of_channels_in_device"] = _read(fp, "<I")
    data["source_name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    data["source_device_name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    data["probe_device_name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    data["probe_source_id"] = _read(fp, "<I")
    data["probe_device_channel"] = _read(fp, "<I")
    data["probe_device_channel"] = _read(fp, "<I")
    data["probe_device_id"] = _read(fp, "<I")
    data["input_voltage_minimum"] = _read(fp, "<d")
    data["input_voltage_maximum"] = _read(fp, "<d")
    data["total_gain"] = _read(fp, "<d")

    # Create empty vectors for our block offsets
    data["block_offsets"] = []
    data["block_num_items"] = []
    data["block_timestamps"] = []
    data["num_spikes"] = 0

    # Skip 128 bytes
    fp.seek(128, os.SEEK_CUR)
    return data


def _read_analog_channel_header(fp: IO):
    """
    Read the header for an analog data channel in the file, starting at the current file location.

    Args:
        fp: The PL2 file object. Upon return, current location is immediately after the analog channel header.

    Returns:
        A dictionary holding the contents of the analog channel header.
    """
    data = {}

    data_type = _read(fp, "<B")
    _read(fp, "<3B")  # data_subtype - not used
    if data_type != PL2_HEADER_ANALOG_CHANNEL:  # D4 03 F8 00 or D4 04 F8 00 01, D4 07 F8 00
        raise RuntimeError(f"Invalid type in analog channel header. Got {data_type}, "
                           f"expected {PL2_HEADER_ANALOG_CHANNEL}")
    data["plex_channel"] = _read(fp, "<I")

    # Two more empty items
    _read(fp, "<I")
    _read(fp, "<I")

    data["name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    data["source"] = _read(fp, "<I")
    data["channel"] = _read(fp, "<I")
    data["enabled"] = _read(fp, "<I")
    data["recording_enabled"] = _read(fp, "<I")
    data["units"] = bytearray(_read(fp, "<16B")).decode('ascii').split('\0', 1)[0]
    data["samples_per_second"] = _read(fp, "<d")
    data["coeff_to_convert_to_units"] = _read(fp, "<d")
    data["source_trodality"] = _read(fp, "<I")
    data["trode"] = _read(fp, "<I")
    data["channel_in_trode"] = _read(fp, "<I")
    data["number_of_channels_in_source"] = _read(fp, "<I")
    data["device_id"] = _read(fp, "<I")
    data["number_of_channels_in_device"] = _read(fp, "<I")
    data["source_name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    data["source_device_name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    data["probe_device_name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    data["probe_source_id"] = _read(fp, "<I")
    data["probe_source_channel"] = _read(fp, "<I")
    data["probe_device_id"] = _read(fp, "<I")
    data["probe_device_channel"] = _read(fp, "<I")
    data["input_voltage_minimum"] = _read(fp, "<d")
    data["input_voltage_maximum"] = _read(fp, "<d")
    data["total_gain"] = _read(fp, "<d")

    # Create empty vectors for our block offsets
    data["block_offsets"] = []
    data["block_num_items"] = []
    data["block_timestamps"] = []
    data["num_values"] = 0

    # Skip 128 bytes
    fp.seek(128, os.SEEK_CUR)
    return data


def _read_event_channel_header(fp: IO):
    """
    Read the header for an event channel in the file, starting at the current file location.

    Args:
        fp: The PL2 file object. Upon return, current location is immediately after the event channel header.

    Returns:
        A dictionary holding the contents of the event channel header.
    """
    data = {}

    data_type = _read(fp, "<B")
    _read(fp, "<3B")  # data_subtype - not used
    if data_type != PL2_HEADER_EVENT_CHANNEL:  # D6
        raise RuntimeError(f"Invalid type in analog channel header. Got {data_type}, "
                           f"expected {PL2_HEADER_EVENT_CHANNEL}")
    data["plex_channel"] = _read(fp, "<I")

    # Two more empty items
    _read(fp, "<I")
    _read(fp, "<I")

    data["name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    data["source"] = _read(fp, "<I")
    data["channel"] = _read(fp, "<I")
    data["enabled"] = _read(fp, "<I")
    data["recording_enabled"] = _read(fp, "<I")
    data["number_of_channels_in_source"] = _read(fp, "<I")
    data["number_of_channels_in_device"] = _read(fp, "<I")
    data["device_id"] = _read(fp, "<I")
    data["num_events"] = _read(fp, "<I")  # TODO - this is not right
    data["source_name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]
    data["source_device_name"] = bytearray(_read(fp, "<64B")).decode('ascii').split('\0', 1)[0]

    # Create empty vectors for our block offsets
    data["block_offsets"] = []
    data["block_num_items"] = []
    data["block_timestamps"] = []
    data["num_events"] = 0

    # Skip 128 bytes
    fp.seek(128, os.SEEK_CUR)
    return data


def _get_creator_software_version(data: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Get the Plexon software version that created the PL2 file. This method assumes the data file header has already
    been parsed and that the software version string 'N.M.R' -- where N, M, and R all are integer strings -- is
    available in the field 'creator_software_version'.

    Args:
        data: Dictionary holding PL2 file contents culled thus far.

    Returns:
        A 3-tuple holding the major, minor and revision number (N, M, R). If version string not found or cannot be
            parsed, returns (1, 18, 0).
    """
    try:
        parts = data['creator_software_version'].split('.')
        if len(parts) != 3:
            raise Exception(f"Bad version string: {data['creator_software_version']}")
        return int(parts[0]), int(parts[1]), int(parts[2])
    except Exception:
        return 1, 18, 0


def get_analog_channel_record_index(data: Dict[str, Any], is_wide_band: bool, channel_number: int) -> int:
    """
    Get the (zero-based) index of the channel record within the PL2 file contents for the specified wide-band ("WB")
    or narrow-band ("SPKC") analog channel.

    Args:
        data: Dictionary holding PL2 file contents.
        is_wide_band: True for wide-band analog channel, False for narrow-band.
        channel_number: The requested channel number (a positive integer between 1 and the number of supported wide-band
            or narrow-band channels).
    Returns:
        Index of relevant channel record within the analog channel list in data['analog_channels'], or -1 if not found.
    """
    try:
        idx = _get_channel_offset(data, PL2_ANALOG_TYPE_WB if is_wide_band else PL2_ANALOG_TYPE_SPKC, channel_number)
    except Exception:
        idx = -1
    return idx


def _get_channel_offset(data: Dict[str, Any], data_subtype: int, channel_number: int) -> int:
    """
    Return the offset to the channel header information record for the specified channel data source or channel number.

    There are four categories of channel data stored in the Plexon data file -- analog data, spike data, event data, and
    start/stop channel data. Within each category -- except the last one -- are one or more channel "sources". For
    example, there are 4 different sources or "subtypes" of analog data. And within each channel source are 1 or more
    individual channels. For example, there are 32 "single-bit" (source = 9) event channels numbered 1-32.

    As PL2 file contents are loaded into the `data` dictionary, the channel data is organized into 4 subdictionaries:
     - `data['analog_channels']` = List of all analog data channels.
     - `data['spike_channels']` = List of all spike data channels.
     - `data['event_channels']` = List of all digital event data channels.
     - `data['start_stop_channels']` = The start/stop data (NOT a list of individual channels).

    Given the data source/subtype and the channel number within that subtype, this method returns the zero-based index
    of the relevant channel record within the channel list for the appropriate channel category.

    NOTES:
     1. The notion of multiple channels per category does not apply to the start/stop data. For this category
        (source == 0), the method always returns 0.
     2. The method makes no assumption about how the channel records are stored within each of 3 channel lists.

    Args:
        data: Dictionary holding PL2 file contents culled thus far.
        data_subtype: The channel data source or subtype. Must be one of PL2_ANALOG_TYPE_WB, _AI, _AI2, _FP, _SPKC;
            PL2_EVENT_TYPE_SINGLE_BIT, _STROBED; PL2_SPIKE_TYPE_SPK, or PL2_SPIKE_TYPE_SPK_SPKC. Will be zero for
            start/stop data.
        channel_number: The channel index.
    Returns:
        The offset value
    Raises:
        RuntimeError: If channel data source is not recognized.
    """
    major, minor, rev = _get_creator_software_version(data)
    spike_sources = [PL2_SPIKE_TYPE_SPK, PL2_SPIKE_TYPE_SPK_SPKC]
    analog_sources = [PL2_ANALOG_TYPE_WB, PL2_ANALOG_TYPE_FP, PL2_ANALOG_TYPE_SPKC,
                      PL2_ANALOG_TYPE_AI if ((major < 1) or (major == 1 and minor < 19)) else PL2_ANALOG_TYPE_AI2]
    event_sources = [PL2_EVENT_TYPE_KBD, PL2_EVENT_TYPE_SINGLE_BIT, PL2_EVENT_TYPE_STROBED]
    if (major < 1) or (major == 1 and minor < 19):
        event_sources.append(PL2_EVENT_TYPE_CINEPLEX)

    channel_list: Optional[List[Dict]] = None
    if data_subtype == 0:
        return 0
    elif data_subtype in spike_sources:
        channel_list = data['spike_channels']
    elif data_subtype in analog_sources:
        channel_list = data['analog_channels']
    elif data_subtype in event_sources:
        channel_list = data['event_channels']

    if channel_list:
        for i in range(len(channel_list)):
            if (channel_list[i]['source'] == data_subtype) and (channel_list[i]['channel'] == channel_number):
                return i

    raise RuntimeError(f"Channel record not found in file header: source={data_subtype}, channel={channel_number}")


def _read_footer(fp: IO, data: Dict[str, Any]) -> None:
    """
    Parse the PL2 file's footer and add its contents to the data dictionary provided.

    Args:
        fp: The PL2 file object.
        data: The data/information dictionary culled from the file thus far. The footer contents are added to this. At a
            minimum, it must include the contents of the file header in order to locate the start of the footer.
    """
    fp.seek(data["internal_value_4"])  # Seek to start of footer

    while fp.tell() < data["file_length"]:
        data_type = _read(fp, "<B")
        data_subtype = _read(fp, "<B")

        # All items are stored as the following
        num_words = _read(fp, "<H")
        channel = _get_channel_offset(data, data_subtype, _read(fp, "<H"))
        _read(fp, "<H")  # Skipped

        # Determine how many items we have based on the number of words
        # Each element is stored as position ("<Q"), timestamp ("<Q"),
        # and number of elements ("<H")
        num_items = int(num_words * 2 / (8 + 8 + 2))
        num_values = _read(fp, "<Q")

        if data_type == PL2_FOOTER_SPIKE_CHANNEL:
            data["spike_channels"][channel]["num_spikes"] = num_values
            data["spike_channels"][channel]["block_offsets"] = _read(fp, "<{:d}Q".format(num_items), True)
            data["spike_channels"][channel]["block_timestamps"] = _read(fp, "<{:d}Q".format(num_items), True)
            data["spike_channels"][channel]["block_num_items"] = _read(fp, "<{:d}H".format(num_items), True)
        elif data_type == PL2_FOOTER_ANALOG_CHANNEL:
            data["analog_channels"][channel]["num_values"] = num_values
            data["analog_channels"][channel]["block_offsets"] = _read(fp, "<{:d}Q".format(num_items), True)
            data["analog_channels"][channel]["block_timestamps"] = _read(fp, "<{:d}Q".format(num_items), True)
            data["analog_channels"][channel]["block_num_items"] = _read(fp, "<{:d}H".format(num_items), True)
        elif data_type == PL2_FOOTER_EVENT_CHANNEL:
            data["event_channels"][channel]["num_events"] = num_values
            data["event_channels"][channel]["block_offsets"] = _read(fp, "<{:d}Q".format(num_items), True)
            data["event_channels"][channel]["block_timestamps"] = _read(fp, "<{:d}Q".format(num_items), True)
            data["event_channels"][channel]["block_num_items"] = _read(fp, "<{:d}H".format(num_items), True)
        elif data_type == PL2_FOOTER_START_STOP_CHANNEL:
            data["start_stop_channels"]["num_events"] = _read(fp, "<Q")
            data["start_stop_channels"]["block_offsets"] = _read(fp, "<{:d}Q".format(num_items), True)
            data["start_stop_channels"]["block_timestamps"] = _read(fp, "<{:d}Q".format(num_items), True)
            data["start_stop_channels"]["block_num_items"] = _read(fp, "<{:d}H".format(num_items), True)
        else:
            raise RuntimeError(f"Unknown data type in footer at {fp.tell()-2}: Got 0x{data_type:x}")
        # Skip to next 16 byte aligned value
        fp.seek(int((fp.tell() + 15) / 16) * 16)


def _reconstruct_footer(fp, data: Dict[str, Any]) -> None:
    """
    Given a PL2 file without a footer, attempt to reconstruct the footer by parsing individual data records.

    Args:
        fp: The PL2 file object. It must be open and is NOT closed upon return.
        data: The data/information dictionary culled from the file thus far. Fields of the reconstructed footer will be
            added to this dictionary.
    """
    # Seek to the first data block and begin _reading
    fp.seek(data["internal_value_2"])

    if data["internal_value_3"] == 0:
        data["internal_value_3"] = data["internal_value_4"]  # File is not complete

    while fp.tell() < data["internal_value_3"]:
        # _read type
        block_offset = fp.tell()
        data_type = _read(fp, "<B")
        data_subtype = _read(fp, "<B")

        if data_type == PL2_DATA_BLOCK_ANALOG_CHANNEL:
            num_items = _read(fp, "<H")
            channel = _get_channel_offset(data, data_subtype, _read(fp, "<H"))
            _read(fp, "<H")  # Unknown
            timestamp = _read(fp, "<Q")

            data["analog_channels"][channel]["num_values"] += num_items
            data["analog_channels"][channel]["block_offsets"].append(block_offset)
            data["analog_channels"][channel]["block_timestamps"].append(timestamp)
            data["analog_channels"][channel]["block_num_items"].append(num_items)

            # Skip over the values
            fp.seek(2 * num_items, os.SEEK_CUR)
        elif data_type == PL2_DATA_BLOCK_SPIKE_CHANNEL:
            _read(fp, "<H")
            channel = _get_channel_offset(data, data_subtype, _read(fp, "<H"))
            num_sample_points = _read(fp, "<H")
            num_items = _read(fp, "<Q")

            # _read each of the items (64 byte values)
            timestamp = _read(fp, "<Q")

            data["spike_channels"][channel]["num_spikes"] += num_items
            data["spike_channels"][channel]["block_offsets"].append(block_offset)
            data["spike_channels"][channel]["block_timestamps"].append(timestamp)
            data["spike_channels"][channel]["block_num_items"].append(num_items)

            # Skip to next instance
            # Spikes: Int16 * * num_items * num_samples pints
            # Assignments: Unit16 * num_items
            # Timestamps = Uint64 * num_items (but we read one already)
            fp.seek(2 * num_items + 2 * num_sample_points * num_items + (num_items - 1) * 8, os.SEEK_CUR)
        elif data_type == PL2_DATA_BLOCK_EVENT_CHANNEL:
            _read(fp, "<H")
            channel = _get_channel_offset(data, data_subtype, _read(fp, "<H"))
            num_items = _read(fp, "<Q")
            _read(fp, "<H")
            timestamp = _read(fp, "<Q")

            data["event_channels"][channel]["num_events"] += num_items
            data["event_channels"][channel]["block_offsets"].append(block_offset)
            data["event_channels"][channel]["block_timestamps"].append(timestamp)
            data["event_channels"][channel]["block_num_items"].append(num_items)

            # Skip to next item
            fp.seek((num_items - 1) * 8 + num_items * 2, os.SEEK_CUR)

        elif data_type == PL2_DATA_BLOCK_START_STOP_CHANNEL:
            # Start-stop Channel
            _read(fp, "<H")
            _read(fp, "<H") - 1  # channel - not used
            num_items = _read(fp, "<Q")
            timestamp = _read(fp, "<Q")

            data["start_stop_channels"]["num_events"] += num_items
            data["start_stop_channels"]["block_offsets"].append(block_offset)
            data["start_stop_channels"]["block_timestamps"].append(timestamp)
            data["start_stop_channels"]["block_num_items"].append(num_items)
            fp.seek((num_items - 1) * 8 + num_items * 2, os.SEEK_CUR)
        else:
            raise RuntimeError(f"Unknown data type at position {fp.tell()-2}. Got 0x{data_type:x}")
        # Align to next 16 byte boundary
        fp.seek(int((fp.tell() + 15) / 16) * 16)
