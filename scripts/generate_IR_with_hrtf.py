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


def generate_stereo_IR(source_r, source_l, filters, bit_depth, fs, visualize=True):
    '''
    Generates an IR file out of given source sound data and an optional array of filters to be applied.

    :param list source: Sound data to be converted into an impulse response file.
    :param list filters: List of filters to be applied (in that order)
    '''
    # Prepare stereo sound data arrays.
    source_signal_r = np.copy(source_r)
    source_signal_l = np.copy(source_l)
    filename_appendix = ""

    # Apply filters for both stereo channels
    for i in range(len(filters)):
        start_time = time.time()
        source_signal_r = Filter(filters[i]).apply(source_signal_r)
        source_signal_l = Filter(filters[i]).apply(source_signal_l)
        end_time = time.time()
        print(f"{filters[i].NAME} time = {end_time-start_time} seconds")
        filename_appendix = f"{filename_appendix}_{filters[i].NAME}"

    # Stack array vertically
    ir_array_r = np.vstack(source_signal_r)
    ir_array_l = np.vstack(source_signal_l)

    # Increase Amplitude to usable levels
    ir_array_r = automatic_gain_increase(ir_array_r, bit_depth, 3)
    ir_array_l = automatic_gain_increase(ir_array_l, bit_depth, 3)

    # Create stereo file (dual mono)
    ir_array = np.concatenate(
        (ir_array_l, ir_array_r), axis=1)

    # Write impulse response file
    filename = f'IR_hrtf_{filename_appendix}_{time.time()}.wav'
    wavfile.write(filename, fs, ir_array.astype(bit_depth))

    if visualize:
        # Create spectrogram
        create_spectrogram(filename, filename_appendix)

        # Visualize waveform of IR
        # plt.title(filename_appendix)
        plt.plot(ir_array)
        plt.show()


def rotate_xy_plane(vec, angle):
    x_rotation = np.array([
        [1,     0,              0],
        [0,     np.cos(angle),  -np.sin(angle)],
        [0,     np.sin(angle),  np.cos(angle)]
    ])

    y_rotation = np.array([
        [np.cos(angle),     0,  np.sin(angle)],
        [0,                 1,  0],
        [-np.sin(angle),    0,  np.cos(angle)]
    ])

    vec = vec @ x_rotation
    vec = vec @ y_rotation

    return vec


'''    
a=[1,1,1] # head posish
b=[0,1,0] # steering vector

r=rotate_xy_plane(b, np.pi/2)
l=-r
print(f"l=\n{l}")
print(f"r=\n{r}")
'''

if __name__ == "__main__":
    # If True, apply frequency dependent wall absorption coefficients to simulate realistic wall/ceiling/floor materials.
    # Caution: Needs more resources!
    freq_dep_abs_coeff = False

    # Wall, floor and ceiling materials the room is consisting of
    # Structure: Array of six materials (use 'mat.xxx') corresponding to:
    # Left wall | Right wall | Front wall | Back wall | Floor | Ceiling
    wall_materials = 4 * \
        [mat.wallpaper_on_lime_cement_plaster] + \
        [mat.parquet_glued] + [mat.concrete]

    # Parameters referring to head related transfer functions (HRTF).
    head_width = 0.1449  # [m]
    head_position = [3, 2, 1]
    head_direction = [0, 1, 0]

    ear_direction_r = rotate_xy_plane(head_direction, np.pi/2)
    ear_direction_l = -ear_direction_r

    ear_position_r = (head_position + ear_direction_r * (head_width / 2))
    ear_position_l = (head_position + ear_direction_l * (head_width / 2))

    print(f"ear_direction_r = \n{ear_direction_r}")
    print(f"ear_direction_l = \n{ear_direction_l}")

    print(f"ear_position_r = \n{ear_position_r}")
    print(f"ear_position_l = \n{ear_position_l}")

    # Common gpuRIR parameters
    room_sz = [5, 4, 3],  # Size of the room [m]
    pos_src = [[1, 1, 1.6]],  # Positions of the sources [m]
    orV_src = [0, -1, 0],  # Steering vector of source(s)
    spkr_pattern = "omni",  # Source polar pattern
    mic_pattern = "homni",  # Receiver polar pattern
    T60 = 1.0,  # Time for the RIR to reach 60dB of attenuation [s]
    # Attenuation when start using the diffuse reverberation model [dB]
    att_diff = 15.0,
    att_max = 60.0,  # Attenuation at the end of the simulation [dB]
    fs = 44100,  # Sampling frequency [Hz]
    # Bit depth of WAV file. Either np.int8 for 8 bit, np.int16 for 16 bit or np.int32 for 32 bit
    bit_depth = np.int32,
    # Absorption coefficient of walls, ceiling and floor.
    wall_materials = wall_materials

    # Define room parameters
    params_left = rp.RoomParameters(
        room_sz=room_sz, pos_src=pos_src, orV_src=orV_src, spkr_pattern=spkr_pattern,
        mic_pattern=mic_pattern, T60=T60, att_diff=att_diff, att_max=att_max, fs=fs,
        bit_depth=bit_depth, wall_materials=wall_materials,

        # Positions of the receivers [m]
        pos_rcv=[ear_position_l],  # Position of left ear
        orV_rcv=ear_direction_l,  # Steering vector of left ear
    )

    params_right = rp.RoomParameters(
        room_sz=room_sz, pos_src=pos_src, orV_src=orV_src, spkr_pattern=spkr_pattern,
        mic_pattern=mic_pattern, T60=T60, att_diff=att_diff, att_max=att_max, fs=fs,
        bit_depth=bit_depth, wall_materials=wall_materials,

        # Positions of the receivers [m]
        pos_rcv=[ear_position_r],  # Position of right ear
        orV_rcv=ear_direction_r,  # Steering vector of right ear
    )

    # Generate two room impulse responses (RIR) with given parameters for each ear
    if freq_dep_abs_coeff:
        receiver_channel_r = fdac.generate_RIR_freq_dep_walls(params_right)
        receiver_channel_l = fdac.generate_RIR_freq_dep_walls(params_left)

    else:
        receiver_channel_r = generate_RIR(params_right)
        receiver_channel_l = generate_RIR(params_left)

    # All listed filters wil be applied in that order.
    # Leave filters array empty if no filters should be applied.

    filters = [
        # Speaker simulation.
        # Comment either one out
        # CharacteristicFilter(cm.tiny_speaker)
        # LinearFilter(101, (0, 100, 150, 7000, 7001, params.fs/2), (0, 0, 1, 1, 0, 0), params.fs)

        # Air absorption simulation.
        # Comment either one out
        # AirAbsBandpass(),
        # AirAbsSTFT()

        # Mic simulation.
        # Comment either one out
        # CharacteristicFilter(cm.sm57_freq_response, params.fs)
        # LinearFilter(101, (0, 100, 150, 7000, 7001, params.fs/2), (0, 0, 1, 1, 0, 0), params.fs)
    ]

    generate_stereo_IR(receiver_channel_r, receiver_channel_l, filters, bit_depth, fs)


    
