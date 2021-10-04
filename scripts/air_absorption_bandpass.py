from filter import FilterStrategy
import numpy as np
from scipy.signal import butter, lfilter, filtfilt
import air_absorption_calculation as aa
import multiprocessing
from multiprocessing import Pool


class Bandpass(FilterStrategy):
    def __init__(self, max_frequency=20000, min_frequency=1, divisions=50, fs=44100):
        self.max_frequency = max_frequency
        self.min_frequency = min_frequency
        self.divisions = divisions
        self.fs = fs
        self.NAME = "Bandpass"

    '''
    Calculates how much distance the sound has travelled. [m]
    '''
    @staticmethod
    def distance_travelled(sample_number, sampling_frequency, c):
        seconds_passed = sample_number * (sampling_frequency ** (-1))
        return (seconds_passed * c)  # [m]

    '''
    Returns a butterworth bandpass filter.
    '''
    @staticmethod
    def create_bandpass_filter(lowcut, highcut, fs, order=10):
        nyq = 0.5 * fs
        low = (lowcut / nyq)
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandpass')
        return b, a

    '''
    Applies a butterworth bandpass filter.
    '''
    @staticmethod
    def apply_bandpass_filter(data, lowcut, highcut, fs, order=10):
        b, a = Bandpass.create_bandpass_filter(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)     # Single Filter
        #y = filtfilt(b, a, data)   # Forward and backward filtering
        return y

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

        # Prepare + apply bandpass filter
        filtered_signal = self.apply_bandpass_filter(
            IR, band_min, band_max, self.fs, 3)

        # Calculate air absorption coefficients
        alpha, _, c, _ = aa.air_absorption(band_mean)

        # Apply attenuation
        for k in range(0, len(filtered_signal)):
            distance = self.distance_travelled(k, self.fs, c)
            attenuation = distance*alpha  # [dB]
            filtered_signal[k] *= 10**(-attenuation / 10)

        return filtered_signal

    '''
    Creates a multi processing pool and calls methods to apply bandpass based air absorption.
    '''
    def air_absorption_bandpass(self, IR):
        pool = multiprocessing.Pool()

        frequency_range = self.max_frequency - self.min_frequency

        # Divide frequency range into defined frequency bands
        filtered_signals = pool.starmap(self.apply_single_band,
                                        ((IR, j, frequency_range)
                                         for j in range(1, self.divisions + 1))
                                        )

        arr = np.array(filtered_signals)
        return np.add(0, arr.sum(axis=0))

    def apply(self, IR):
        return self.air_absorption_bandpass(IR)
