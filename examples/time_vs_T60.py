#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 11:27:28 2018

@author: daviddiaz
"""

import numpy as np
import matplotlib.pyplot as plt
import  time

import gpuRIR

T60_vec = np.arange(0.1, 0.8, 0.2) #np.arange(0.1, 2.2, 0.2)
nb_test_per_point = 1

nb_src = 32
nb_rcv = 4
room_sz = np.array([3,4,2.5])
att_diff = 50
att_max = 50
fs = 16000

pos_src = np.random.rand(nb_src, 3) * room_sz
pos_rcv = np.random.rand(nb_rcv, 3) * room_sz
abs_weights = np.ones((6,1))

time_max = 100
times = np.zeros((len(T60_vec),1))
for i in range(len(T60_vec)):
	T60 = T60_vec[i]
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

plt.semilogy(T60_vec, times)
plt.ylabel("time [s]")
plt.xlabel("T60 [s]")