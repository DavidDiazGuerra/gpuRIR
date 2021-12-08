import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from math import ceil
from scipy.io import wavfile
import time

from gpuRIR.extensions.filters.filter import Filter
from gpuRIR.extensions.filters.characteristic_filter import CharacteristicFilter
from gpuRIR.extensions.filters.air_absorption_bandpass import AirAbsBandpass
from gpuRIR.extensions.filters.air_absorption_stft import AirAbsSTFT
from gpuRIR.extensions.filters.linear_filter import LinearFilter
import gpuRIR.extensions.filters.characteristic_models as cm
import gpuRIR.extensions.filters.air_absorption_calculation as aa
from gpuRIR.extensions.wall_absorption.materials import Materials as mat
import gpuRIR.extensions.wall_absorption.freq_dep_abs_coeff as fdac
from gpuRIR.extensions.create_spectrogram import create_spectrogram_from_file
import gpuRIR.extensions.room_parameters as rp
from gpuRIR.extensions.generate_RIR import generate_RIR
from gpuRIR.extensions.create_spectrogram import *
# TODO: optimize imports


"""
Generates an impulse response WAV file (IR) with optional filters.
Example usage: Convolving (reverberating) an audio signal in an impulse response loader plug-in like Space Designer in Logic Pro X.
TODO: Update doc
"""


def mono_adaptive_gain(source, bit_depth, ceiling):
    '''
    Increases amplitude (loudness) to defined ceiling.

    :param list source: Sound data to process.
    :param int bit_depth: Bit depth of source sound data.
    :param int ceiling: Maximum loudness (relative dB, e.g. -1dB) the sound data should be amplified to
    :return: Amplified source sound data.
    '''
    peak = np.max(source)
    negative_peak = np.abs(np.min(source))

    # Check if the negative or positive peak is of a higher magnitude
    if peak < negative_peak:
        peak = negative_peak

    max_gain = np.iinfo(bit_depth).max*10**(-ceiling/10)
    factor = max_gain/peak

    return source * factor


def stereo_adaptive_gain(source_l, source_r, bit_depth, ceiling):
    '''
    Increases amplitude (loudness) to defined ceiling.

    :param list source_l: Left channel sound data to process.
    :param list source_r: Right channel sound data to process.
    :param int bit_depth: Bit depth of source sound data.
    :param int ceiling: Maximum loudness (relative dB, e.g. -1dB) the sound data should be amplified to
    :return: Amplified source sound data.
    '''
    # Read out positive and negative peak values
    peak_l = np.max(source_l)
    peak_r = np.max(source_r)
    negative_peak_l = np.abs(np.min(source_l))
    negative_peak_r = np.abs(np.min(source_r))

    # Check if left or right negative or positive peak is of a higher magnitude
    if peak_l < negative_peak_l:
        peak_l = negative_peak_l
    if peak_r < negative_peak_r:
        peak_r = negative_peak_r
    if peak_l < peak_r:
        peak_l = peak_r

    # Calculate amplification factor based on bit depth and highest overall peak of both channels
    max_gain = np.iinfo(bit_depth).max * 10 ** (-ceiling / 10)
    factor = max_gain / peak_l

    return (source_l * factor, source_r * factor)


def generate_mono_IR(source, filters, bit_depth, fs, enable_adaptive_gain=True,  visualize=False, verbose=False):
    '''
    Generates an IR file out of given source sound data and an optional array of filters to be applied.

    :param list source: Sound data to be converted into an impulse response file.
    :param list filters: List of filters to be applied (in that order)

    TODO: Update doc
    '''
    # Prepare sound data arrays.
    source_signal = np.copy(source)
    filename_appendix = ""

    # Apply filters
    for i in range(len(filters)):
        start_time = time.time()
        source_signal = Filter(filters[i]).apply(source_signal)
        end_time = time.time()

        # Print processing time per filter (relevant for benchmarking)
        if verbose:
            print(f"{filters[i].NAME} time = {end_time-start_time} seconds")
        filename_appendix = f"{filename_appendix}_{filters[i].NAME}"

    # Stack array vertically
    impulseResponseArray = np.vstack(source_signal)

    # Increase Amplitude to usable levels
    if enable_adaptive_gain:
        impulseResponseArray = mono_adaptive_gain(
            impulseResponseArray, bit_depth, 3)

    # Create stereo file (dual mono)
    impulseResponseArray = np.concatenate(
        (impulseResponseArray, impulseResponseArray), axis=1)

    # Write impulse response file
    filename = f'IR_{filename_appendix}_{time.time()}.wav'
    wavfile.write(filename, fs, impulseResponseArray.astype(bit_depth))

    if visualize:
        # Create spectrogram
        create_spectrogram_from_file(filename, filename_appendix)

        # Visualize waveform of IR
        # plt.title(filename_appendix)
        plt.plot(impulseResponseArray)
        plt.show()


def generate_stereo_IR(source_r, source_l, filters_r, filters_l, bit_depth, fs, enable_adaptive_gain=True, verbose=False, visualize=False):
    '''
    Generates an IR file out of given source sound data and an optional array of filters to be applied.

    :param list source: Sound data to be converted into an impulse response file.
    :param list filters: List of filters to be applied (in that order)

    TODO: Update doc
    '''
    # Prepare stereo sound data arrays.
    source_signal_r = np.copy(source_r)
    source_signal_l = np.copy(source_l)
    filename_appendix = ""

    # Apply filters for both stereo channels
    for i in range(len(filters_r)):
        start_time = time.time()
        source_signal_r = Filter(filters_r[i]).apply(source_signal_r)
        end_time = time.time()
        # Print processing time per filter (relevant for benchmarking)
        if verbose:
            print(
                f"Right Channel {filters_r[i].NAME} time = {end_time-start_time} seconds")
        filename_appendix = f"{filename_appendix}_{filters_r[i].NAME}"

    for i in range(len(filters_l)):
        start_time = time.time()
        source_signal_l = Filter(filters_l[i]).apply(source_signal_l)
        end_time = time.time()
        # Print processing time per filter (relevant for benchmarking)
        if verbose:
            print(
                f"Left Channel {filters_l[i].NAME} time = {end_time-start_time} seconds")
        filename_appendix = f"{filename_appendix}_{filters_l[i].NAME}"

    # Stack array vertically
    ir_array_r = np.vstack(source_signal_r)
    ir_array_l = np.vstack(source_signal_l)

    # Increase Amplitude to usable levels
    if enable_adaptive_gain:
        ir_array_l, ir_array_r = stereo_adaptive_gain(
            ir_array_l, ir_array_r, bit_depth, 3)

    # Create stereo file (dual mono)
    ir_array = np.concatenate(
        (ir_array_l, ir_array_r), axis=1)

    # Write impulse response file
    filename = f'IR_{filename_appendix}_{time.time()}.wav'
    wavfile.write(filename, fs, ir_array.astype(bit_depth))

    if visualize:
        # Create spectrogram
        create_spectrogram_from_data(ir_array, fs, filename_appendix)

        # Visualize waveform of IR
        plt.plot(ir_array_l, label="Left channel")
        plt.title("Left channel")
        plt.show()
        plt.plot(ir_array_r, label="Right channel")
        plt.title("Right channel")
        plt.show()
