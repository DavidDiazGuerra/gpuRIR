"""
Generates an impulse response WAV file (IR) with optional filters.
Example usage: Convolving (reverberating) an audio signal in an impulse response loader plug-in like Space Designer in Logic Pro X.
"""
from filters.filter import Filter
import librosa
from filters.characteristic_filter import CharacteristicFilter
from filters.air_absorption_bandpass import AirAbsBandpass
from filters.air_absorption_stft import AirAbsSTFT
from filters.linear_filter import LinearFilter

import filters.characteristic_models as cm
import filters.air_absorption_calculation as aa

from wall_absorption.materials import Materials as mat
import wall_absorption.freq_dep_abs_coeff as fdac

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from math import ceil
from scipy.io import wavfile
import time
from create_spectrogram import create_spectrogram

import room_parameters as rp
from generate_RIR import generate_RIR


def automatic_gain_increase(source, bit_depth, ceiling):
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


def generate_IR(source, filters, bit_depth, fs, visualize=True):
    '''
    Generates an IR file out of given source sound data and an optional array of filters to be applied.

    :param list source: Sound data to be converted into an impulse response file.
    :param list filters: List of filters to be applied (in that order)
    '''
    # Prepare sound data arrays.
    source_signal = np.copy(source)
    filename_appendix = ""

    # Apply filters
    for i in range(len(filters)):
        start_time = time.time()
        source_signal = Filter(filters[i]).apply(source_signal)
        end_time = time.time()
        print(f"{filters[i].NAME} time = {end_time-start_time} seconds")
        filename_appendix = f"{filename_appendix}_{filters[i].NAME}"

    # Stack array vertically
    impulseResponseArray = np.vstack(source_signal)

    # Increase Amplitude to usable levels
    impulseResponseArray = automatic_gain_increase(
        impulseResponseArray, bit_depth, 3)

    # Create stereo file (dual mono)
    impulseResponseArray = np.concatenate(
        (impulseResponseArray, impulseResponseArray), axis=1)

    # Write impulse response file
    filename = f'IR_{filename_appendix}_{time.time()}.wav'
    wavfile.write(filename, fs, impulseResponseArray.astype(bit_depth))

    if visualize:
        # Create spectrogram
        create_spectrogram(filename, filename_appendix)

        # Visualize waveform of IR
        # plt.title(filename_appendix)
        plt.plot(impulseResponseArray)
        plt.show()


if __name__ == "__main__":
    # If True, apply frequency dependent wall absorption coefficients to simulate realistic wall/ceiling/floor materials
    # Needs more resources!
    freq_dep_abs_coeff = True

    # Wall, floor and ceiling materials the room is consisting of
    # Structure: Array of six materials (use 'mat.xxx') corresponding to:
    # Left wall | Right wall | Front wall | Back wall | Floor | Ceiling
    wall_materials = 6 * [mat.wood_16mm_on_40mm_slats]

    # Define room parameters
    params = rp.RoomParameters(
        room_sz = [5, 4, 3],  # Size of the room [m]
        pos_src = [[1, 1, 1.6]],  # Positions of the sources ([m]
        pos_rcv = [[4, 3, 1.6]],  # Positions of the receivers [m]
        orV_src = [0, -1, 0],  # Steering vector of source
        orV_rcv = [0, 1, 0],  # Steering vector of receiver
        spkr_pattern = "omni",  # Source polar pattern
        mic_pattern = "card",  # Receiver polar pattern
        T60 = 1.0,  # Time for the RIR to reach 60dB of attenuation [s]
        # Attenuation when start using the diffuse reverberation model [dB]
        att_diff = 15.0,
        att_max = 60.0,  # Attenuation at the end of the simulation [dB]
        fs = 44100,  # Sampling frequency [Hz]
        # Bit depth of WAV file. Either np.int8 for 8 bit, np.int16 for 16 bit or np.int32 for 32 bit
        bit_depth = np.int32,
        # Absorption coefficient of walls, ceiling and floor.
        wall_materials = wall_materials
    )

    if freq_dep_abs_coeff:
        receiver_channels = fdac.generate_RIR_freq_dep_walls(params)
    else:
        receiver_channels = generate_RIR(params)

    for i in range(len(params.pos_rcv)):
        # All listed filters wil be applied in that order.
        # Leave filters array empty if no filters should be applied.

        filters = [
            # Speaker simulation
            # LinearFilter(101, (0, 100, 150, 7000, 7001, params.fs/2), (0, 0, 1, 1, 0, 0), params.fs),

            # Air absorption simulation
            # AirAbsBandpass(),

            # Mic simulation
            # CharacteristicFilter(cm.sm57_freq_response, params.fs),
        ]

        generate_IR(receiver_channels[i], filters, params.bit_depth, params.fs)
