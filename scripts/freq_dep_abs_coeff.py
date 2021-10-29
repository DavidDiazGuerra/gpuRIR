from filters.filter import Filter
import librosa

import filters.air_absorption_calculation as aa
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from math import ceil
from scipy.io import wavfile
from scipy.interpolate import interp1d
import time
import gpuRIR
from create_spectrogram import create_spectrogram
import filters.characteristic_models as cm
import filters.materials as mat

from scipy.signal import butter, lfilter, filtfilt
import multiprocessing
from multiprocessing import Pool

'''
abs_weights [6]: Absortion coefficient ratios of the walls
'''
def generate_RIR(abs_weights):
    '''
    Generates RIRs from the gpuRIR library.

    :return: Receiver channels (mono)
    '''
    gpuRIR.activateMixedPrecision(False)
    gpuRIR.activateLUT(False)


    room_sz = [5, 4, 3]  # Size of the room [m]
    nb_src = 1  # Number of sources
    pos_src = np.array([[1, 1, 1.6]])  # Positions of the sources ([m]
    nb_rcv = 1  # Number of receivers
    pos_rcv = np.array([[4, 3, 1.6]])	 # Position of the receivers [m]
    # Vectors pointing in the same direction than the receivers
    orV_src = np.matlib.repmat(np.array([0, -1, 0]), nb_src, 1)
    orV_rcv = np.matlib.repmat(np.array([0, 1, 0]), nb_rcv, 1)
    spkr_pattern = "card"  # Source polar pattern
    mic_pattern = "card"  # Receiver polar pattern
    T60 = 1.0	 # Time for the RIR to reach 60dB of attenuation [s]
    # Attenuation when start using the diffuse reverberation model [dB]
    att_diff = 15.0
    att_max = 60.0  # Attenuation at the end of the simulation [dB]
    fs = 44100  # Sampling frequency [Hz]
    # Bit depth of WAV file. Either np.int8 for 8 bit, np.int16 for 16 bit or np.int32 for 32 bit
    bit_depth = np.int32
    beta = gpuRIR.beta_SabineEstimation(
        room_sz, T60, abs_weights=abs_weights)  # Reflection coefficients
    # Time to start the diffuse reverberation model [s]
    Tdiff = gpuRIR.att2t_SabineEstimator(att_diff, T60)
    # Time to stop the simulation [s]
    Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)
    # Number of image sources in each dimension
    nb_img = gpuRIR.t2n(Tdiff, room_sz)
    RIRs = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img,
                              Tmax, fs, Tdiff=Tdiff, orV_src=orV_src, orV_rcv=orV_rcv, spkr_pattern=spkr_pattern, mic_pattern=mic_pattern)

    # return receiver channels (mono), number of receivers, sampling frequency and bit depth from RIRs.
    return RIRs[0], pos_rcv, fs, bit_depth


'''
Interpolates frequency response array with cubic spline method.
'''
def interpolate_pair(abs_coeff, plot):
    # y: absorption coefficient
    # x: frequency [Hz]
    y = abs_coeff[:, 1]
    x = abs_coeff[:, 0]
    f = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
    x_interpolated = np.arange(1, 20000)
    y_interpolated = f(x_interpolated)
    if plot: 
        plt.plot(x, y, 'o')
        plt.plot(x_interpolated, y_interpolated, "-")    
    return f

def show_plot():
    plt.xlim(right=20000)
    plt.legend()
    plt.show()


'''
Returns a butterworth bandpass filter.
'''
def create_bandpass_filter(lowcut, highcut, fs, order=10):
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

def automatic_gain_increase(source, bit_depth, ceiling):
    '''
    Increases amplitude (loudness) to defined ceiling.

    :param list source: Sound data to process.
    :param int bit_depth: Bit depth of source sound data.
    :param int ceiling: Maximum loudness (relative dB, e.g. -1dB) the sound data should be amplified to
    :return: Amplified source sound data.
    '''
    peak = np.max(source)
    negative_peak = np.abs(np.min(source))

    # Check if the negative or positive peak is of a higher magnitude
    if peak < negative_peak:
        peak = negative_peak

    max_gain = np.iinfo(bit_depth).max*10**(-ceiling/10)
    factor = max_gain/peak

    return source*factor

'''
abs_weights [6]:    Absortion coefficient ratios of the walls
freq_low:           Where bandpass starts to pass
freq_high:          Where bandpass starts to cut off
'''
def generate_RIR_bandpassed():
    BUTTER_ORDER= 3
    min_frequency = 20
    max_frequency = 20000
    PLOT_ABS_COEFF_CURVES = False
    # number of bands
    divisions = 10

    band = max_frequency / divisions

    # Structure: min / mean / max
    bands = np.zeros((divisions, 3))

    # Loop through each band
    for band_num in range (1, divisions + 1):
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

    # Here we select the 6 materials we want the room to consist of:
    wall_materials =  2 * [mat.water, mat.double_window, mat.fiber_plaster]
    # We create 6 interpolating functions for each material:
    wall_mat_interp = [interpolate_pair(mat, PLOT_ABS_COEFF_CURVES) for mat in wall_materials]
    if PLOT_ABS_COEFF_CURVES: show_plot()


    RIRs = []
    bit_depth = 0
    fs = 0

    for band in bands:
        abs_coeffs=np.zeros(len(wall_mat_interp))
        for i in range(len(wall_mat_interp)):
            abs_coeffs[i]=wall_mat_interp[i](band[1])
        # Generate RIR
        RIR, _, fs, bit_depth = generate_RIR(abs_coeffs)
        # Bandpass RIR
        bandpassed = apply_bandpass_filter(RIR[0], band[0], band[2], fs, BUTTER_ORDER)
        RIRs.append(bandpassed)

    # Sum up all bandpassed RIR's
    final_RIR = np.add(0, np.array(RIRs).sum(axis=0))

    # Write file
    # Stack array vertically
    impulseResponseArray = np.vstack(final_RIR)

    # Increase Amplitude to usable levels
    impulseResponseArray = automatic_gain_increase(
        impulseResponseArray, bit_depth, 3)

    # Create stereo file (dual mono)
    impulseResponseArray = np.concatenate(
        (impulseResponseArray, impulseResponseArray), axis=1)

    filename = f'IR_abs_coeff_{time.time()}.wav'
    wavfile.write(filename, fs, impulseResponseArray.astype(bit_depth))

    create_spectrogram(filename, "Jerkop")
    

generate_RIR_bandpassed()

