from filter import FilterStrategy
import numpy as np
from scipy.signal import butter, lfilter
import air_absorption_calculation as aa


class Bandpass(FilterStrategy):

    def __init__(self, max_frequency=20000, min_frequency=1, divisions=1, fs=44100):
        self.max_frequency = max_frequency
        self.min_frequency = min_frequency
        self.divisions = divisions
        self.fs = fs

    '''
    Calculates how much distance the sound has travelled. [m]
    '''

    def distance_travelled(self, sample_number, sampling_frequency, c):
        seconds_passed = sample_number*(sampling_frequency**(-1))
        return (seconds_passed*c)  # [m]

    '''
    Returns a butterworth bandpass filter.
    '''

    def create_bandpass_filter(self, lowcut, highcut, fs, order=3):
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

    # max_frequency, min_frequency, divisions, fs
    def air_absorption_bandpass(self, IR):
        frequency_range = self.max_frequency - self.min_frequency

        combined_signals = np.zeros(len(IR))

        # Divide frequency range into defined frequency bands
        for j in range(1, self.divisions + 1):
            # Upper ceiling of each band
            band_max = ((frequency_range / self.divisions) * j)

            # Lower ceiling of each band and handling of edge case
            if j == 1:
                band_min = self.min_frequency
            else:
                band_min = ((frequency_range / self.divisions) * (j - 1))

            # Calculating mean frequency of band which determines the attenuation.
            band_mean = (band_max+band_min)/2
            print(
                f"Band {j} frequencies: min: {band_min} max: {band_max} mean:{band_mean}")

            # Prepare + apply bandpass filter
            filtered_signal = self.apply_bandpass_filter(
                IR, band_min, band_max, self.fs, 3)

            # Apply attenuation
            for k in range(0, len(filtered_signal)):
                alpha, alpha_iso, c, c_iso = aa.air_absorption(band_mean)
                distance = self.distance_travelled(k, self.fs, c)
                attenuation = distance*alpha  # [dB]

                filtered_signal[k] *= 10**(-attenuation / 10)

            # Summing the different bands together
            for k in range(0, len(combined_signals)):
                combined_signals[k] += filtered_signal[k]
        
        return combined_signals

    def apply(self, IR):
        return self.air_absorption_bandpass(IR)
