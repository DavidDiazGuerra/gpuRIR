#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 12:25:42 2018

@author: daviddiaz
"""

import numpy as np
import matplotlib.pyplot as plt
from math import ceil

import gpuRIR

room_sz = [3,3,2.5]
nb_src = 2
pos_src = np.array([[1,2.9,0.5],[1,2,0.5]], dtype=np.float32)
nb_rcv = 3
pos_rcv = np.array([[0.5,1,0.5],[1,1,0.5],[1.5,1,0.5]], dtype=np.float32)
abs_weights = [0.9]*5+[0.5]
T60 = 1
att_diff = 15
att_max = 60
fs=16000.0

beta = gpuRIR.beta_SabineEstimation(abs_weights, room_sz, T60)
Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, T60)
Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)
nb_img = gpuRIR.t2n( Tdiff, room_sz )
RIRs = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tdiff, Tmax, fs)

t = np.arange(int(ceil(Tmax * fs))) / fs
plt.plot(t, RIRs.reshape(nb_src*nb_rcv, -1).transpose())