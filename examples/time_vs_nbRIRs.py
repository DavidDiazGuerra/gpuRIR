#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Measure the runtime of the library against the number of RIRs.
It may need several minutes.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

import gpuRIR
gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(False)

nb_src_vec = np.concatenate([2**np.arange(12), [4095]]) # Number of RIRs to measure
nb_test_per_point = 10 # Number of simulations per T60 to average the runtime

nb_rcv = 1 # Number of receivers
room_sz = np.array([3,4,2.5]) # Room size [m]
T60 = 0.7 # Reverberation time [s]
att_diff = 13.0 # Attenuation when start using the diffuse reverberation model [dB]
att_max = 50.0 # Attenuation at the end of the simulation [dB]
fs = 16000 # Sampling frequency [Hz]

pos_rcv = np.random.rand(nb_rcv, 3) * room_sz

time_max = 100 # Stop the measurements after find an average time greter than this time [s]
times = np.zeros((len(nb_src_vec),1))
for i in range(len(nb_src_vec)):
	nb_src = nb_src_vec[i]
	pos_src = np.random.rand(nb_src, 3) * room_sz
	start_time = time.time()
	
	for j in range(nb_test_per_point): 
		beta = gpuRIR.beta_SabineEstimation(room_sz, T60)
		Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, T60)
		Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)
		nb_img = gpuRIR.t2n( Tdiff, room_sz )
		RIRs = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff)
		
	times[i] = (time.time() - start_time) / nb_test_per_point
	
	if times[i] > time_max:
		break

print(times.transpose())

plt.loglog(nb_src_vec, times)
plt.ylabel("time [s]")
plt.xlabel("number of RIRs")
plt.show()
