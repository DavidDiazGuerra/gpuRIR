#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 12:09:15 2018

@author: daviddiaz
"""

import numpy as np
import matplotlib.pyplot as plt
import  time

import gpuRIR

nb_src_vec = np.concatenate([2**np.arange(12), [4095]]) #2**np.arange(8) #
nb_test_per_point = 10

nb_rcv = 1
room_sz = np.array([3,4,2.5])
T60 = 0.7
att_diff = 13
att_max = 50
fs = 16000

pos_rcv = np.random.rand(nb_rcv, 3) * room_sz
abs_weights = np.ones((6,1))

time_max = 100
times = np.zeros((len(nb_src_vec),1))
for i in range(len(nb_src_vec)):
	nb_src = nb_src_vec[i]
	pos_src = np.random.rand(nb_src, 3) * room_sz
	start_time = time.time()
	
	for j in range(nb_test_per_point): 
		beta = gpuRIR.beta_SabineEstimation(abs_weights, room_sz, T60)
		Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, T60)
		Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)
		nb_img = gpuRIR.t2n( Tdiff, room_sz )
		RIRs = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tdiff, Tmax, fs)
		
	times[i] = (time.time() - start_time) / nb_test_per_point
	
	if times[i] > time_max:
		break

print(times.transpose())

plt.loglog(nb_src_vec, times)
plt.ylabel("time [s]")
plt.xlabel("number of RIRs")