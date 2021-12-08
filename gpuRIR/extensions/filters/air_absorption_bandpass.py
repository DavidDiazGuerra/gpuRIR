import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt, bilinear, sosfreqz, sosfilt
import multiprocessing
from multiprocessing import Pool

from gpuRIR.extensions.filters.filter import FilterStrategy
from gpuRIR.extensions.filters.air_absorption_calculation import air_absorption
from gpuRIR.extensions.filters.butterworth import Butterworth


class AirAbsBandpass(FilterStrategy):
    def __init__(self, max_frequency = 22050, min_frequency = 1, divisions = 50, fs = 44100, order = 4, LR = False, verbose = False, visualize = False):
        assert(order >= 4), "Order must be greater than 4!"

        self.min_frequency = min_frequency
        self.divisions = divisions
        self.fs = fs
        self.max_frequency = fs / 2
        self.order = order
        self.LR = LR
        if LR:
            order = order / 2
        self.NAME = "bandpass_air_abs"
        self.visualize = visualize
        self.verbose = verbose

    '''
    Calculates how much distance the sound has travelled. [m]
    '''
    @staticmethod
    def distance_travelled(sample_number, sampling_frequency, c):
        seconds_passed = sample_number * (sampling_frequency ** (-1))
        return (seconds_passed * c)  # [m]

    

    def apply_single_band(self, IR, band_num, frequency_range):
        band = frequency_range / self.divisions

        # Upper ceiling of each band
        band_max = (band * band_num)

        # Lower ceiling of each band and handling of edge case
        if band_num == 1:
            band_min = self.min_frequency
        else:
            band_min = (band * (band_num - 1))

        # Calculating mean frequency of band which determines the attenuation.
        band_mean = (band_max + band_min) / 2

        if self.verbose: print(f"Band {band_num}:\tMin:{band_min}\tMean:{band_mean}\tMax:{band_max}\tBand width:{band_max - band_mean}")

        # Prepare + apply bandpass filter
        if band_num == 1:
            filtered_signal = Butterworth.apply_pass_filter(
                IR, band_max, self.fs, 'lowpass', self.order*2, self.visualize
            )
        elif band_num == self.divisions:
            filtered_signal = Butterworth.apply_pass_filter(
                IR, band_min, self.fs, 'highpass', self.order*2, self.visualize
            )
        else:
            filtered_signal = Butterworth.apply_bandpass_filter(
                IR, band_min, band_max, self.fs, self.order, self.LR, self.visualize)

        # Calculate air absorption coefficients
        alpha, _, c, _ = air_absorption(band_mean)

        # Apply attenuation
        for k in range(0, len(filtered_signal)):
            distance = self.distance_travelled(k, self.fs, c)
            attenuation = distance*alpha  # [dB]
            filtered_signal[k] *= 10**(-attenuation / 10)

        return filtered_signal

    

    def air_absorption_bandpass(self, IR):
        '''
        Creates a multi processing pool and calls methods to apply bandpass based air absorption.

        :param IR Room impulse response array.
        :return Processed IR array.
        '''
        if self.visualize:
            plt.xscale('log')
            plt.title('Air Absorption: Butterworth filter frequency response')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Amplitude [dB]')
            plt.ylim(bottom=-40)
            plt.margins(0, 0.1)
            plt.grid(which='both', axis='both')
            plt.axvline(100, color='green')

        pool = multiprocessing.Pool()

        frequency_range = self.max_frequency - self.min_frequency

        if not self.visualize:
            if self.verbose: print("Processed bands are out of order due to multi-processing.")
            # Divide frequency range into defined frequency bands
            filtered_signals = pool.starmap(self.apply_single_band,
                                            ((IR, j, frequency_range)
                                             for j in range(1, self.divisions + 1))
                                            )
        else:
            filtered_signals = []
            for j in range(1, self.divisions + 1):
                filtered_signals.append(
                    self.apply_single_band(IR, j, frequency_range))

        if self.visualize:
            plt.show()

        arr = np.array(filtered_signals)
        return np.add(0, arr.sum(axis=0))

    def apply(self, IR):
        return self.air_absorption_bandpass(IR)
