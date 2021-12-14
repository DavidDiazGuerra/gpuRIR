import gpuRIR
from .materials import Materials as mat
import gpuRIR.extensions.generate_RIR as gRIR

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from math import ceil
from scipy.interpolate import interp1d
import time
import multiprocessing
from multiprocessing import Pool
from gpuRIR.extensions.filters.butterworth import Butterworth


def interpolate_pair(abs_coeff, visualize):
    """Interpolates frequency response array. Returns a function.
    
    Parameters
    ----------
    abs_coeff : 2D ndarray
        Per-frequency-range absorption coefficient of a virtual wall material, as (pre-) defined in extensions/wall_absorption/materials.py.
    visualize : bool
        Plots the interpolated frequency responses on a 2D graph.

    Returns
    -------
    functions
        Interpolation function of a single virtual room material as (pre-) defined in extensions/wall_absorption/materials.py.
    """
    # y: absorption coefficient
    # x: frequency [Hz]
    y = abs_coeff[:, 1]
    x = abs_coeff[:, 0]
    f = interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))
    x_interpolated = np.arange(1, 20000)
    y_interpolated = f(x_interpolated)
    if visualize:
        plt.title(
            'Frequency Dependent Absorption Coefficients: Interpolated Frequency Responses')
        plt.plot(x, y, 'o')
        plt.plot(x_interpolated, y_interpolated, "-")
    return f


def generate_RIR_freq_dep_walls(params, band_width=100, factor=1.5, order=4, LR=False, visualize=False, verbose=False):
    """ Generates a custom room impulse response (RIR) with virtual room materials applied using a bandpassing method.

    Parameters:
    -----------
    params : RoomParameters
        gpuRIR parameters
    band_width : int, optional
        Initial width of frequency band. Lower means higher quality but less performant. (recommended: 10)
    factor : float, optional
        Multiplication factor of frequency band. Lower means higher quality but less performant. (recommended: 1.1)
    order : int, optional
        Butterworth filter order.
    LR : bool, optional
        Enables Linkwitz-Riley filtering. LR filter order will get converted automatically.
    plot : bool, optional
        Plots the interpolated material frequency response curve.
    verbose : bool, optional
        Prints current band parameters.

    Returns
    -------
    2D ndarray
        Processed Room impulse response array.
    """

    assert(band_width > 1), "Band width must be greater than 1!"
    assert(factor > 1), "Factor must be greater than 1!"
    assert(order >= 4), "Order must be greater than 4!"

    if LR:
        order = order / 2

    min_frequency = 1
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
    wall_mat_interp = [interpolate_pair(mat, visualize)
                       for mat in params.wall_materials]
    if visualize:
        plt.xlim(right=20000)
        plt.show()

    receiver_channels = np.zeros((len(params.pos_rcv), 1))

    if visualize:
        plt.xscale('log')
        plt.title(
            'Frequency Dependent Absorption Coefficients: Butterworth filter frequency response')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        plt.ylim(bottom=-40)
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(100, color='green')

    for i in range(len(bands)):
        band = bands[i]
        if verbose:
            print(f"Band {i}: {band[0]}")
        abs_coeffs = np.zeros(len(wall_mat_interp))
        for j in range(len(wall_mat_interp)):
            abs_coeffs[j] = wall_mat_interp[j](band[1])
        # Generate RIR
        params.beta = 6 * [1.] - abs_coeffs
        RIR = gRIR.generate_RIR(params)

        # Apply band/lowpassing and re-compiling sound data
        for rcv in range(len(params.pos_rcv)):
            # Lowpass lowest frequency band
            if i == 0:
                processed = Butterworth.apply_pass_filter(
                    RIR[rcv], band[2], params.fs, 'lowpass', order, visualize
                )
                if verbose:
                    print(f"Lowpass frequency: {band[2]}")

            # Highpass highest frequency band
            elif i == (len(bands) - 1):
                processed = Butterworth.apply_pass_filter(
                    RIR[rcv], band[0], params.fs, 'highpass', order, visualize
                )
                if verbose:
                    print(f"Highpass frequency: {band[0]}")

            # All other bands are bandpassed
            else:
                processed = Butterworth.apply_bandpass_filter(
                    RIR[rcv], band[0], band[2], params.fs, order, LR, visualize)

            # Re-compiling sound data
            receiver_channels.resize(len(params.pos_rcv), len(processed))
            receiver_channels[rcv] += processed

    if visualize:
        plt.show()

    # Sum up all bandpassed RIR's per receiver
    return receiver_channels
