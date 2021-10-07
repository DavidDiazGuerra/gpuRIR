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

'''
Microphone models
'''

# Shure SM57 dynamic microphone. Standard mic for US presidential speeches
sm57_freq_response = np.array([
    # Frequency in Hz | Relative response in dB
    [50, -10],
    [100, -4],
    [200, 0],
    [400, -1],
    [700, -0.1],
    [1000, 0],
    [1500, 0.05],
    [2000, 0.1],
    [3000, 2],
    [4000, 3],
    [5000, 5],
    [6000, 6.5],
    [7000, 3],
    [8000, 2.5],
    [9000, 4],
    [10000, 3.5],
    [12000, 1],
    [14000, 2],
    [15000, -5]
])

# iPhone X.
iphone_x_freq_response = np.array([
    [30, -12],
    [40, -3],
    [50, 0],
    [70, 1],
    [100, 2],
    [150, 3],
    [170, 2],
    [200, 1.5],
    [250, 1.5],
    [300, 3],
    [400, 2.5],
    [500, 2],
    [600, 2],
    [1000, 2.5],
    [1500, 3],
    [2000, 3],
    [2500, 4],
    [3000, 4],
    [4000, 3],
    [5000, 3.5],
    [6000, 3.8],
    [7000, 4],
    [8000, 4.5],
    [9000, 4.75],
    [10000, 5],
    [12500, 1],
    [15000, -5],
    [17000, -7],
    [20000, -8]
])


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
    filename = f'impulse_response_rcv_{filter.NAME}_char_{i}_{time.time()}.wav'
    create_soundfile(impulseResponseArray.astype(bit_depth), fs, filename)

    # Create spectrogram
    create_spectrogram(filename, filter.NAME)

    return impulseResponseArray


for i in range(0, len(pos_rcv)):
       
    # Apply air abs
    characteristic_filter = CharacteristicFilter(iphone_x_freq_response, fs, plot=True)
    rcv_filter = generate_IR(receiver_channels[i], characteristic_filter)

    # Apply source char..

