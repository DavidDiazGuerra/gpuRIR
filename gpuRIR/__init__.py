
import numpy as np
from scipy.optimize import minimize
from gpuRIR_bind import simulateRIR_bind

__all__ = ["beta_SabineEstimation", "att2t_SabineEstimator", "t2n", "simulateRIR"]

def beta_SabineEstimation(abs_weights, room_sz, T60):	
				
		def t60error(x, T60, room_sz, abs_weights):
			abs_weights /= np.array(abs_weights).max()
			alpha = x * abs_weights
			Sa = (alpha[0]+alpha[1]) * room_sz[1]*room_sz[2] + \
				(alpha[2]+alpha[3]) * room_sz[0]*room_sz[2] + \
				(alpha[4]+alpha[5]) * room_sz[0]*room_sz[1]
			V = np.prod(room_sz)
			if Sa == 0: return T60 - 0 # Anechoic chamber 
			return abs(T60 - 0.161 * V / Sa) # Sabine's formula
		
		result = minimize(t60error, 0.5, args=(T60, room_sz, abs_weights), bounds=[[0, 1]])		
		return np.sqrt(1 - result.x * abs_weights)
	
def att2t_SabineEstimator(att_dB, T60):
	return att_dB/60.0 * T60

def t2n(T, rooms_sz, c=343.0):
	nb_img = 2 * T / (np.array(rooms_sz) / c)
	return [ int(n) for n in np.ceil(nb_img) ]

def simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tdiff, Tmax, fs, c=343.0):
	assert np.sum(pos_src >= room_sz) == 0, "The sources must be inside the room"
	assert np.sum(pos_rcv >= room_sz) == 0, "The receivers must be inside the room"
	assert Tdiff <= Tmax, "Tmax must be equal or greater than Tdff"
	
	pos_src = pos_src.astype('float32', order='C', copy=False)
	pos_rcv = pos_rcv.astype('float32', order='C', copy=False)
	
	return simulateRIR_bind(room_sz, beta, pos_src, pos_rcv, nb_img, Tdiff, Tmax, fs, c)