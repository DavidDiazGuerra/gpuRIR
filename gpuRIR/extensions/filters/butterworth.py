from scipy.signal import butter, sosfreqz, sosfilt
import matplotlib.pyplot as plt
import numpy as np


class Butterworth():
    ''' Encapsulates Butterworth/Linkwitz Riley functionality with visualization.
    '''

    @staticmethod
    def create_bandpass_filter(lowcut, highcut, fs, order, visualize=False):
        ''' Returns a butterworth bandpass filter.

        Parameters
	    ----------
        lowcut : int
            Lower bound of bandpass. [Hz]
        highcut : int
            Upper bound of bandpass. [Hz]
        fs : int
            Sample rate. [Hz]
        order : int
            Order of Butterworth filter.
        visualize : bool, optional
            Plots band divisions in a graph.

        Returns
	    -------
        ndarray
            Second order sections of IIR filter.
        '''
        sos = butter(order, [lowcut, highcut],
                     btype='bandpass', fs=fs, output='sos')
        if visualize:
            w, h = sosfreqz(sos, fs=fs)
            plt.plot(w, 20*np.log10(np.abs(h)+1e-7))
        return sos

    @staticmethod
    def apply_bandpass_filter(data, lowcut, highcut, fs, order, LR=False, visualize=False):
        ''' Applies a butterworth bandpass filter.

        Parameters
	    ----------
        lowcut : int
            Lower bound of bandpass. [Hz]
        highcut : int
            Upper bound of bandpass. [Hz]
        fs : int
            Sample rate. [Hz]
        order : int
            Order of Butterworth filter.
        LR : bool 
            Enables Linkwitz-Riley filter.
        visualize : bool, optional
            Plots band divisions in a graph.
        
        Returns
	    -------
        ndarray
            y Filtered sound data.
        '''
        sos = Butterworth.create_bandpass_filter(
            lowcut, highcut, fs, order, visualize)
        y = sosfilt(sos, data)
        if LR:
            # Filter once again for Linkwitz-Riley filtering
            y = sosfilt(sos, data)
        return y

    @staticmethod
    def create_pass_filter(cut, fs, pass_type, order, visualize=False):
        ''' Returns a butterworth filter.

        Parameters
	    ----------
        cut : int
            Cut frequency [Hz]
        fs : int
            Sample rate. [Hz]
        pass_type : str
            Type of butterworth filter (e.g. 'lowpass' or 'highpass').
        order : int
            Order of Butterworth filter.
        visualize : bool, optional
            Plots band divisions in a graph.

        Returns
	    -------
        ndarray
            Second order section of butterworth filter.
        '''
        sos = butter(order, cut, btype=pass_type, fs=fs, output='sos')

        if visualize:
            w, h = sosfreqz(sos, fs=fs)
            plt.plot(w, 20*np.log10(np.abs(h)+1e-7))
        return sos

    @staticmethod
    def apply_pass_filter(data, cut, fs, pass_type, order, visualize=False):
        ''' Applies a butterworth filter.

        Parameters
	    ----------
        cut : int
            Cut frequency [Hz]
        fs : int
            Sample rate. [Hz]
        pass_type : str
            Type of butterworth filter (e.g. 'lowpass' or 'highpass').
        order : int
            Order of Butterworth filter.
        visualize : bool, optional
            Plots band divisions in a graph.

        Returns
	    -------
        ndarray
            Second order section of butterworth filter.
        '''
        sos = Butterworth.create_pass_filter(
            cut, fs, pass_type, order=order, visualize=visualize)
        sos = sosfilt(sos, data)     # Single Filter
        return sos
