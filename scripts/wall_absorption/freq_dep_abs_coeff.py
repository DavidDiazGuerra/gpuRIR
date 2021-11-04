import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from math import ceil
from scipy.interpolate import interp1d
import time
import gpuRIR
from .materials import Materials as mat

from scipy.signal import butter, lfilter, filtfilt
import multiprocessing
from multiprocessing import Pool

from generate_RIR import generate_RIR


'''
Interpolates frequency response array with cubic spline method.
'''
def interpolate_pair(abs_coeff, plot):
    # y: absorption coefficient
    # x: frequency [Hz]
    y = abs_coeff[:, 1]
    x = abs_coeff[:, 0]
    f = interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))
    x_interpolated = np.arange(1, 20000)
    y_interpolated = f(x_interpolated)
    if plot:
        plt.plot(x, y, 'o')
        plt.plot(x_interpolated, y_interpolated, "-")
    return f

'''
Shows plot of the room frequency response.
'''
def show_plot():
    plt.xlim(right=20000)
    plt.legend()
    plt.show()


'''
Returns a butterworth bandpass filter.
'''
def create_bandpass_filter(lowcut, highcut, fs, order=9):
    nyq = 0.5 * fs
    low = (lowcut / nyq)
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


'''
Applies a butterworth bandpass filter.
'''
def apply_bandpass_filter(data, lowcut, highcut, fs, order=10):
    b, a = create_bandpass_filter(
        lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)     # Single Filter
    return y


'''
abs_weights [6]:    Absortion coefficient ratios of the walls
freq_low:           Where bandpass starts to pass
freq_high:          Where bandpass starts to cut off
'''
def generate_RIR_freq_dep_walls(params, divisions=10, order=3, plot=False):
    min_frequency = 20
    max_frequency = 20000
    band = max_frequency / divisions

    # Structure: min / mean / max
    bands = np.zeros((divisions, 3))

    # Loop through each band
    for band_num in range(1, divisions + 1):
        # Upper ceiling of each band
        band_max = (band * band_num)

        # Lower ceiling of each band and handling of edge case
        if band_num == 1:
            band_min = min_frequency
        else:
            band_min = (band * (band_num - 1))

        # Calculating mean frequency of band which determines the attenuation.
        band_mean = (band_max + band_min) / 2

        # Fill up array
        bands[band_num - 1] = [band_min, band_mean, band_max]

    # We create 6 interpolating functions for each material:
    wall_mat_interp = [interpolate_pair(mat, plot) for mat in params.wall_materials]
    if plot:
        show_plot()

    receiver_channels = np.zeros((len(params.pos_rcv), 1))

    for i in range(len(bands)):
        band = bands[i]
        abs_coeffs = np.zeros(len(wall_mat_interp))
        for i in range(len(wall_mat_interp)):
            abs_coeffs[i] = wall_mat_interp[i](band[1])
        # Generate RIR
        params.beta = 6 * [1.] - abs_coeffs
        RIR = generate_RIR(params)

        for rcv in range(len(params.pos_rcv)):
            # Bandpass RIR
            bandpassed = apply_bandpass_filter(RIR[rcv], band[0], band[2], params.fs, order)
            print(bandpassed)
            receiver_channels.resize(len(params.pos_rcv), len(bandpassed))
            receiver_channels[rcv] += bandpassed

    # Sum up all bandpassed RIR's per receiver
    return receiver_channels
