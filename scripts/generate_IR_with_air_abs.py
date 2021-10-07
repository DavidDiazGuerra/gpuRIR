""" 
Generates an impulse response WAV file (IR).
Example usage: Convolving (reverberating) an audio signal in an impulse response loader plug-in like Space Designer in Logic Pro X.
"""
from filters.filter import Filter
from filters.air_absorption_bandpass import Bandpass
from filters.air_absorption_stft import STFT

from create_spectrogram import create_spectrogram

import librosa
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from math import ceil
from scipy.io import wavfile
import time
import gpuRIR

gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(False)

room_sz = [8,10,3]  # Size of the room [m]
nb_src = 1  # Number of sources
pos_src = np.array([[4,2,1.7]]) # Positions of the sources ([m]
nb_rcv = 1 # Number of receivers
pos_rcv = np.array([[4,8,1.7]])	 # Position of the receivers [m]
orV_rcv = np.matlib.repmat(np.array([0,1,0]), nb_rcv, 1) # Vectors pointing in the same direction than the receivers
mic_pattern = "card" # Receiver polar pattern
abs_weights = [0.9]*5+[0.5] # Absortion coefficient ratios of the walls
T60 = 1.0	 # Time for the RIR to reach 60dB of attenuation [s]
att_diff = 15.0	# Attenuation when start using the diffuse reverberation model [dB]
att_max = 60.0 # Attenuation at the end of the simulation [dB]
fs=44100 # Sampling frequency [Hz]
bit_depth=np.int32 # Bit depth of WAV file. Either np.int8 for 8 bit, np.int16 for 16 bit or np.int32 for 32 bit

beta = gpuRIR.beta_SabineEstimation(room_sz, T60, abs_weights=abs_weights) # Reflection coefficients
Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, T60) # Time to start the diffuse reverberation model [s]
Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)	 # Time to stop the simulation [s]
nb_img = gpuRIR.t2n( Tdiff, room_sz )	# Number of image sources in each dimension
RIRs = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern)

receiver_channels = RIRs[0] # Extract receiver channels (mono) from RIRs.

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
    source_signal=np.copy(source)

    # Apply filter
    start_time=time.time()
    filtered_signal = Filter(filter).apply(source_signal)
    end_time=time.time()
    print(f"{filter.NAME} time = {end_time-start_time} seconds")

    # Stack array vertically
    impulseResponseArray = np.vstack(filtered_signal)

    # Increase Amplitude to usable levels 
    impulseResponseArray=automatic_gain_increase(impulseResponseArray, bit_depth, 3)

    # Create stereo file (dual mono)
    impulseResponseArray = np.concatenate((impulseResponseArray, impulseResponseArray), axis=1)

    # Write impulse response file
    filename=f'impulse_response_rcv_atten_{filter.NAME}_{i}_{time.time()}.wav'
    create_soundfile(impulseResponseArray.astype(bit_depth), fs, filename)
    
    # Create spectrogram
    create_spectrogram(filename, filter.NAME)

    return impulseResponseArray


for i in range(0, len(pos_rcv)):
    bandpass_data=generate_IR(receiver_channels[i], Bandpass())
    stft_data=generate_IR(receiver_channels[i], STFT())
