from scipy.signal import butter, sosfreqz, sosfilt
import matplotlib.pyplot as plt
import numpy as np

class Butterworth():
    
    @staticmethod
    def create_bandpass_filter(lowcut, highcut, fs, order, visualize=False):
        '''
        Returns a butterworth bandpass filter.
        :param lowcut Lower bound of bandpass
        :param highcut Upper bound of bandpass.
        :fs Sample rate
        :order Order of Butterworth filter.
        :returns sos.
        '''
        sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
        if visualize:
            w, h = sosfreqz(sos, fs=fs)
            plt.plot(w, 20*np.log10(np.abs(h)+1e-7))
        return sos

    
    @staticmethod
    def apply_bandpass_filter(data, lowcut, highcut, fs, order, LR = False, visualize = False):
        '''
        Applies a butterworth bandpass filter.
        '''
        sos = Butterworth.create_bandpass_filter(lowcut, highcut, fs, order, visualize)
        y = sosfilt(sos, data)
        if LR: y = sosfilt(sos, data) # Filter once again for Linkwitz-Riley filtering
        return y

    
    @staticmethod
    def create_pass_filter(cut, fs, pass_type, order, visualize=False):
        '''
        Returns a butterworth lowpass filter.
        '''
        sos = butter(order, cut, btype=pass_type, fs=fs, output='sos')

        if visualize:
            w, h = sosfreqz(sos, fs=fs)
            plt.plot(w, 20*np.log10(np.abs(h)+1e-7))
        return sos


    @staticmethod
    def apply_pass_filter(data, cut, fs, pass_type, order, visualize=False):
        '''
        Applies a butterworth lowpass filter.
        '''
        sos = Butterworth.create_pass_filter(
            cut, fs, pass_type, order=order, visualize=visualize)
        sos = sosfilt(sos, data)     # Single Filter
        return sos