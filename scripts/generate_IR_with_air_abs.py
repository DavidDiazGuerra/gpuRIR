""" 
Generates an impulse response WAV file (IR).
Example usage: Convolving (reverberating) an audio signal in an impulse response loader plug-in like Space Designer in Logic Pro X.
"""
from filter import Filter
import librosa
from air_absorption_bandpass import Bandpass
from air_absorption_stft import STFT

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
Parameters relating to air absorption
enable_air_absorption=True # Determines if air absorption is applied.
divisions=1 # How many partitions the frequency spectrum gets divided into. Roughly correlates to quality / accuracy.
min_frequency=20.0 # [Hz] Lower frequency boundary.
max_frequency=20000.0 # [Hz] Upper frequency boundary.
'''

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

    # Visualize waveform of IR
    plt.title(filter.NAME)
    plt.plot(impulseResponseArray)
    plt.show()

    return impulseResponseArray


for i in range(0, len(pos_rcv)):
    bandpass_data=generate_IR(receiver_channels[i], Bandpass())
    stft_data=generate_IR(receiver_channels[i], STFT())

    # Calculate and visualize difference of two waveforms
    '''
    difference=bandpass_data-stft_data
        
    print("difference:")
    print(difference)
    plt.plot(difference)
    plt.show()

    difference_filename=f'difference_{time.time()}.wav'
    create_soundfile(difference, fs, difference_filename)
    create_spectrogram(difference_filename)
    '''


t = np.arange(int(ceil(Tmax * fs))) / fs
