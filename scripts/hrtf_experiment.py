#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Example for simulating the recording of a moving source with a microphone array.
You need to have a 'source_signal.wav' audio file to use it as source signal and it will generate
the file 'filtered_signal.wav' with the stereo recording simulation.
"""
import numpy as np
import matplotlib.pyplot as plt
import generate_IR_with_hrtf as hrtf
from scipy.io import wavfile

import gpuRIR
gpuRIR.activateMixedPrecision(False)

fs, source_signal = wavfile.read('scripts/example.wav')

# gpuRIR parameters
room_sz = [3,4,2.5]  # Size of the room [m]
traj_pts = 4  # Number of trajectory points
pos_traj = np.tile(np.array([0.0,3.0,1.0]), (traj_pts,1))
pos_traj[:,0] = np.linspace(0.1, 2.9, traj_pts) # Positions of the trajectory points [m]
mic_pattern = "card" # Receiver polar pattern
T60 = 0.6 # Time for the RIR to reach 60dB of attenuation [s]
att_diff = 15.0	# Attenuation when start using the diffuse reverberation model [dB]
att_max = 60.0 # Attenuation at the end of the simulation [dB]

# HRTF parameters
head_width = 0.1449  # [m]
head_position = [2, 2, 1.6]
head_direction = [0, 0, 0]

ear_direction_r = np.round(hrtf.rotate_z_plane(head_direction, np.pi/2))
ear_direction_l = -ear_direction_r

ear_position_r = (head_position + ear_direction_r * (head_width / 2))
ear_position_l = (head_position + ear_direction_l * (head_width / 2))




"""

nb_img = gpuRIR.t2n( Tdiff, room_sz )	# Number of image sources in each dimension

filtered_signal = gpuRIR.simulateTrajectory(source_signal, RIRs)
wavfile.write('filtered_signal.wav', fs, filtered_signal)
plt.plot(filtered_signal)
plt.show()
"""