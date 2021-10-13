from filters.filter import FilterStrategy

import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

class LinearFilter(FilterStrategy):
    
    def __init__(self, numtabs, bands, desired, fs, plot=False):
        self.numtabs=numtabs
        self.bands = bands
        self.desired = desired
        self.fs = fs
        self.plot=plot
        self.NAME = "linear"

    '''
    Applies a linear filter.
    '''
    def apply_linear_filter(self, data):
        b_siumulated_mic = scipy.signal.firls(self.numtabs, self.bands, self.desired, fs=self.fs)  # design filter

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
