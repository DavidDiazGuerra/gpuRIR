from filters.filter import FilterStrategy

import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

class LinearFilter(FilterStrategy):
    
    def __init__(self, freq_response, fs, plot):
        self.freq_response = freq_response
        self.fs = fs
        self.plot=plot
        self.NAME = "Linear"

    @staticmethod
    def rel_db_response_to_ratio(gain_dB):
        return 10**(gain_dB/10)

    '''
    Applies a linear filter.
    '''
    def apply_linear_filter(self, data):
        relative_response = self.freq_response[:, 1]
        bands = self.freq_response[:, 0]  # band frequencies in Hz

        for i in range(len(relative_response)):
            relative_response[i] = self.rel_db_response_to_ratio(relative_response[i])

        b_siumulated_mic = scipy.signal.firls(51, bands, relative_response, fs=self.fs)  # design filter

        if self.plot:
            # Plot frequency resonse of filter
            w, response = scipy.signal.freqz(b_siumulated_mic)
            freq = w/np.pi*self.fs/2
            _, ax1 = plt.subplots()
            ax1.set_title('Simulated Mic Freq Response')
            ax1.plot(freq, 20 * np.log10(abs(response)), 'b')
            ax1.set_ylabel('Amplitude [dB]', color='b')
            ax1.set_xlabel('Frequency [Hz]')
            plt.show()

        # Apply filter to audio signal
        return scipy.signal.lfilter(b_siumulated_mic, 1, data)

    def apply(self, IR):
        return self.apply_linear_filter(IR)
