""" 
Generates an impulse response WAV file (IR).
Example usage: Convolving (reverberating) an audio signal in an impulse response loader plug-in like Space Designer in Logic Pro X.
"""
from filter import Filter

import librosa

from linear_filter import LinearFilter
import air_absorption_calculation as aa
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from math import ceil
from scipy.io import wavfile
import time
import gpuRIR
from create_spectrogram import create_spectrogram

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

receiver_channels = RIRs[0]  # Extract receiver channels (mono) from RIRs.

sm57_freqs = np.array(
    [50, 500, 900, 1000, 1100]
)
sm57_relative_response = np.array(
    [0.1, 0, -10, -20, -40]
)

plt.plot(sm57_freqs, sm57_relative_response)

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


def generate_IR(source, filter):
    # Prepare sound data arrays.
    source_signal = np.copy(source)

    # Apply filter
    start_time = time.time()
    filtered_signal = Filter(filter).apply(source_signal)
    end_time = time.time()
    print(f"{filter.NAME} time = {end_time-start_time} seconds")

    # Stack array vertically
    impulseResponseArray = np.vstack(filtered_signal)

    # Increase Amplitude to usable levels
    impulseResponseArray = automatic_gain_increase(
        impulseResponseArray, bit_depth, 3)

    # Create stereo file (dual mono)
    impulseResponseArray = np.concatenate(
        (impulseResponseArray, impulseResponseArray), axis=1)

    # Write impulse response file
    filename = f'impulse_response_rcv_atten_{filter.NAME}_{i}_{time.time()}.wav'
    create_soundfile(impulseResponseArray.astype(bit_depth), fs, filename)

    # Create spectrogram
    create_spectrogram(filename, filter.NAME)

    # Visualize waveform of IR
    plt.title(filter.NAME)
    plt.plot(impulseResponseArray)
    plt.show()

    return impulseResponseArray


for i in range(0, len(pos_rcv)):
    linear_filter = LinearFilter(
        sm57_freqs, sm57_relative_response, receiver_channels[i])
    rcv_filter = generate_IR(receiver_channels[i], linear_filter)

t = np.arange(int(ceil(Tmax * fs))) / fs
