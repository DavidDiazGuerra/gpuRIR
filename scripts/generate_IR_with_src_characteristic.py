""" 
Generates an impulse response WAV file (IR).
Example usage: Convolving (reverberating) an audio signal in an impulse response loader plug-in like Space Designer in Logic Pro X.
"""
from filters.filter import Filter
import librosa
#from filters.linear_filter import LinearFilter
from filters.characteristic_filter import CharacteristicFilter
import filters.air_absorption_calculation as aa
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from math import ceil
from scipy.io import wavfile
import time
import gpuRIR
from create_spectrogram import create_spectrogram
from generate_white_noise import generate_white_noise

'''
Speaker models
'''
# Tiny speaker, comparable to smartphone speaker
tiny_speaker = np.array([
    [200, -10],
    [300, -2],
    [1000, 0],
    [5000, 3],
    [7000, 5],
    [10000, -2],
    [12000, -5],
    [14000, -10]
])


gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(False)

room_sz = [8, 10, 3]  # Size of the room [m]
nb_src = 1  # Number of sources
pos_src = np.array([[4, 2, 1.7]])  # Positions of the sources ([m]
nb_rcv = 1  # Number of receivers
pos_rcv = np.array([[4, 8, 1.7]])	 # Position of the receivers [m]
# Vectors pointing in the same direction than the receivers
orV_rcv = np.matlib.repmat(np.array([0, 1, 0]), nb_rcv, 1)
mic_pattern = "card"  # Receiver polar pattern
abs_weights = [0.9]*5+[0.5]  # Absortion coefficient ratios of the walls
T60 = 1.0	 # Time for the RIR to reach 60dB of attenuation [s]
# Attenuation when start using the diffuse reverberation model [dB]
att_diff = 15.0
att_max = 60.0  # Attenuation at the end of the simulation [dB]
fs = 44100  # Sampling frequency [Hz]
# Bit depth of WAV file. Either np.int8 for 8 bit, np.int16 for 16 bit or np.int32 for 32 bit
bit_depth = np.int32

beta = gpuRIR.beta_SabineEstimation(
    room_sz, T60, abs_weights=abs_weights)  # Reflection coefficients
# Time to start the diffuse reverberation model [s]
Tdiff = gpuRIR.att2t_SabineEstimator(att_diff, T60)
# Time to stop the simulation [s]
Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)
# Number of image sources in each dimension
nb_img = gpuRIR.t2n(Tdiff, room_sz)
RIRs = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img,
                          Tmax, fs, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern)


# Generate source white noise
white_noise = generate_white_noise(Tdiff, Tmax, fs, 1, 1)

# Apply characteristic to source signal
speaker_filter = CharacteristicFilter(tiny_speaker, fs, plot=True)
filtered_white_noise = Filter(speaker_filter).apply(white_noise)

# Send filtered white noise through gpuRIR
filtered_signal=gpuRIR.simulateTrajectory(filtered_white_noise, RIRs, fs=fs)

'''
Increases amplitude (loudness) to defined ceiling.
'''


def automatic_gain_increase(data, bit_depth, ceiling):
    peak = np.max(data)
    negative_peak = np.abs(np.min(data))

    # Check if the negative or positive peak is of a higher magnitude
    if peak < negative_peak:
        peak = negative_peak

    max_gain = np.iinfo(bit_depth).max*10**(-ceiling/10)
    factor = max_gain/peak

    return data*factor


def create_soundfile(data, fs, filename):
    wavfile.write(filename, fs, data)


def generate_IR(source, filter=None, apply_filter=False):
    # Prepare sound data arrays.
    source_signal = np.copy(source)

    # Apply filter
    if apply_filter:
        start_time = time.time()
        filtered_signal = Filter(filter).apply(source_signal)
        end_time = time.time()
        print(f"{filter.NAME} time = {end_time-start_time} seconds")
    else:
        filtered_signal=source_signal

    # Stack array vertically
    impulseResponseArray = np.vstack(filtered_signal)

    # Increase Amplitude to usable levels
    impulseResponseArray = automatic_gain_increase(impulseResponseArray, bit_depth, 3)

    # Create stereo file (dual mono)
    impulseResponseArray = np.concatenate((impulseResponseArray, impulseResponseArray), axis=1)

    # Write impulse response file
    filename = f'impulse_response_rcv_{filter.NAME}_char_{i}_{time.time()}.wav'
    create_soundfile(impulseResponseArray.astype(bit_depth), fs, filename)

    # Create spectrogram
    create_spectrogram(filename, filter.NAME)

    # Visualize waveform of IR
    plt.title(filter.NAME)
    plt.plot(impulseResponseArray)
    plt.show()

    return impulseResponseArray


for i in range(0, len(pos_rcv)):
    characteristic_filter = CharacteristicFilter(tiny_speaker, fs, plot=True)
    rcv_filter = generate_IR(filtered_signal, characteristic_filter, apply_filter=False)
