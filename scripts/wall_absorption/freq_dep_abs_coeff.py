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
params:             gpuRIR parameters
band_width:         Initial width of frequency band. Lower means higher quality but less performant. (recommended: 10)
factor:             Multiplication factor of frequency band. Lower means higher quality but less performant. (recommended: 1.1)
order:              Butterworth filter order.
LR:                 Uses Linkwitz-Riley filtering. LR filter order is a double of order parameter (e.g. order = 2 -> LR order = 4)
plot:               Plots the interpolated material frequency response curve.
verbose:            Prints current band parameters.
'''


def generate_RIR_freq_dep_walls(params, band_width=100, factor=1.5, order=2, LR=True, plot=False, verbose=True):
    assert(factor > 1), "Factor must be greater than 1!"

    min_frequency = 20
    max_frequency = 20000

    # Structure: min / mean / max
    bands = []

    # Find out minimum and maximum frequency value of all materials
    material_frequencies = np.vstack(params.wall_materials)[:, 0]
    min_mat_frequency = np.min(material_frequencies)
    max_mat_frequency = np.max(material_frequencies)

    # Outside of the material bounds, absorption values are constant. (Not wasting bands)
    bands.append(
        [min_frequency, (min_frequency + min_mat_frequency) / 2, min_mat_frequency])
    if verbose:
        print(
            f"Min:{min_frequency}\tMean:{(min_frequency + min_mat_frequency) / 2}\tMax:{min_mat_frequency}")

    reached_max = False

    current_min = min_mat_frequency
    current_max = current_min + band_width
    current_mean = (current_min + current_max) / 2

    while not reached_max:
        if current_max > max_mat_frequency:
            reached_max = True

        bands.append([current_min, current_mean, current_max])

        if verbose:
            print(
                f"Min:{current_min}\tMean:{current_mean}\tMax:{current_max}\tBand width:{band_width}")
        band_width *= factor

        current_min = current_max
        current_max = current_min + band_width
        current_mean = (current_min + current_max) / 2

    bands.append(
        [current_min, (current_min + max_frequency) / 2, max_frequency])
    if verbose:
        print(
            f"Min:{current_min}\tMean:{(current_min + max_frequency) / 2}\tMax:{max_frequency}")

    # We create 6 interpolating functions for each material:
    wall_mat_interp = [interpolate_pair(mat, plot)
                       for mat in params.wall_materials]
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
            bandpassed = apply_bandpass_filter(
                RIR[rcv], band[0], band[2], params.fs, order)
            if LR:
                bandpassed = apply_bandpass_filter(
                    bandpassed, band[0], band[2], params.fs, order)

            receiver_channels.resize(len(params.pos_rcv), len(bandpassed))
            receiver_channels[rcv] += bandpassed

    # Sum up all bandpassed RIR's per receiver
    return receiver_channels
