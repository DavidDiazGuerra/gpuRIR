#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Generates an impulse response WAV file (IR).
Example usage: Convolving (reverberating) an audio signal in an impulse response loader plug-in like Space Designer in Logic Pro X.
"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from math import ceil
from scipy.io import wavfile
import time

import gpuRIR
gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(False)

room_sz = [6,6,5]  # Size of the room [m]
nb_src = 1  # Number of sources
pos_src = np.array([[1,1,2.5]]) # Positions of the sources ([m]
nb_rcv = 3 # Number of receivers
pos_rcv = np.array([[5.9,5.9,4.9],[3,3,2.5],[1.5,1.5,0.5]])	 # Position of the receivers [m]
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
RIRs = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern, air_abs=0.0001)

receiver_channels = RIRs[0] # Extract receiver channels (mono) from RIRs.

for i in range(0, len(pos_rcv)):
    # Stack array vertically
    impulseResponseArray = np.vstack(receiver_channels[i])

    # Increase Amplitude to usable levels
    impulseResponseArray = impulseResponseArray * np.iinfo(bit_depth).max

    # Create stereo file (dual mono)
    impulseResponseArray = np.concatenate((impulseResponseArray, impulseResponseArray), axis=1)

    #impulseResponseArray=impulseResponseArray[1]
    print(impulseResponseArray)

    # Write impulse response file
    wavfile.write(f'impulse_response_rcv_{i}_{time.time()}.wav', fs, impulseResponseArray.astype(bit_depth))

    # Visualize waveform of IR
    plt.plot(impulseResponseArray)

t = np.arange(int(ceil(Tmax * fs))) / fs
plt.show()
