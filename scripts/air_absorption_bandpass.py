from filter import FilterStrategy
import numpy as np
from scipy.signal import butter, lfilter
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

    def distance_travelled(self, sample_number, sampling_frequency, c):
        seconds_passed = sample_number*(sampling_frequency**(-1))
        return (seconds_passed*c)  # [m]

    '''
    Returns a butterworth bandpass filter.
    '''

    def create_bandpass_filter(self, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = (lowcut / nyq)
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandpass')
        return b, a

    '''
    Applies a butterworth bandpass filter.
    '''

    def apply_bandpass_filter(self, data, lowcut, highcut, fs, order=3):
        b, a = self.create_bandpass_filter(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def apply_single_band(self, IR, band_num, frequency_range):
        # Upper ceiling of each band
        band_max = ((frequency_range / self.divisions) * band_num)

        # Lower ceiling of each band and handling of edge case
        if band_num == 1:
            band_min = self.min_frequency
        else:
            band_min = ((frequency_range / self.divisions) * (band_num - 1))

        # Calculating mean frequency of band which determines the attenuation.
        band_mean = (band_max+band_min)/2

        # Prepare + apply bandpass filter
        filtered_signal = self.apply_bandpass_filter(
            IR, band_min, band_max, self.fs, 3)

        # Calculate air absorption coefficients
        alpha, alpha_iso, c, c_iso = aa.air_absorption(band_mean)

        # Apply attenuation
        for k in range(0, len(filtered_signal)):
            distance = self.distance_travelled(k, self.fs, c)
            attenuation = distance*alpha  # [dB]
            filtered_signal[k] *= 10**(-attenuation / 10)

        return filtered_signal

    # max_frequency, min_frequency, divisions, fs

    def air_absorption_bandpass(self, IR):
        pool = multiprocessing.Pool()

        frequency_range = self.max_frequency - self.min_frequency
        combined_signals = np.zeros(len(IR))

        # Divide frequency range into defined frequency bands
        filtered_signals = pool.starmap(self.apply_single_band,
                                        ((IR, j, frequency_range)
                                         for j in range(1, self.divisions + 1))
                                        )

        arr = np.array(filtered_signals)
        return np.add(0, arr.sum(axis=0))

    def apply(self, IR):
        return self.air_absorption_bandpass(IR)
