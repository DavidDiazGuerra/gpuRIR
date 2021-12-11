import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

from gpuRIR.extensions.filters.filter import FilterStrategy

class LinearFilter(FilterStrategy):
    
    def __init__(self, numtaps, bands, desired, fs, visualize=False):
        ''' Instantiates linear filtering.
        For parameters reference see scipy.signal.firls.
        '''
        self.numtaps=numtaps
        self.bands = bands
        self.desired = desired
        self.fs = fs
        self.visualize=visualize
        self.NAME = "linear"

    
    def apply_linear_filter(self, data):
        ''' Applies a linear filter on given sound data.

        Parameters
	    ----------
        IR : 2D ndarray
            Room impulse response array.

        Returns
	    -------
        2D ndarray
            Processed Room impulse response array.
        '''
        b_siumulated_mic = scipy.signal.firls(self.numtaps, self.bands, self.desired, fs=self.fs)  # design filter

        if self.visualize:
            # Plot frequency resonse of filter
            w, response = scipy.signal.freqz(b_siumulated_mic)
            freq = w/np.pi*self.fs/2
            _, ax1 = plt.subplots()
            ax1.set_title('Linear Filter: Simulated Frequency Response')
            ax1.plot(freq, 20 * np.log10(abs(response)), 'b')
            ax1.set_ylabel('Amplitude [dB]', color='b')
            ax1.set_xlabel('Frequency [Hz]')
            plt.show()

        # Apply filter to audio signal
        return scipy.signal.lfilter(b_siumulated_mic, 1, data)

    def apply(self, IR):
        ''' Calls method to apply linear filtering on the source data.

        Parameters
	    ----------
        IR : 2D ndarray
            Room impulse response array.

        Returns
	    -------
        2D ndarray
            Processed Room impulse response array.

        '''
        return self.apply_linear_filter(IR)
