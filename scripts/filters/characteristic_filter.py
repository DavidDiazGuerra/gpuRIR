from filters.filter import FilterStrategy

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import istft, stft

'''
Applies deviations of a frequency response (e.g. a microphone frequency response)
to an audio signal via short time fourier transformation.
'''
class CharacteristicFilter(FilterStrategy):

    def __init__(self, freq_response, T=25, hr=50, ps=1, fs=44100, nFFT=256, noverlap=int(256*0.75), window='hanning', plot=False):
        self.T = T
        self.hr = hr
        self.ps = ps
        self.fs = fs
        self.nFFT = nFFT
        self.noverlap = noverlap
        self.window = window
        self.freq_response = freq_response
        self.plot = plot
        self.NAME = "STFT"

    '''
    Interpolates frequency response array with cubic spline method.
    '''
    @staticmethod
    def interpolate_frequency_response(freq_response, plot):
        # y: relative response [dB]
        # x: frequency [Hz]
        y = freq_response[:, 1]
        x = freq_response[:, 0]
        f = CubicSpline(x, y)
        x_interpolated = np.arange(1, 20000)
        y_interpolated = f(x_interpolated)
        if plot:
            plt.plot(x, y, 'o', x_interpolated, y_interpolated, "-")
            plt.show
        return f

    '''
    Applies the characteristic on the source data.
    '''
    def apply_characteristic(self, RIR):
        characteristic=self.interpolate_frequency_response(self.freq_response, self.plot)

        # Calculate STFT of RIR (like spectrogram)
        f, t, RIR_TF = stft(
            RIR, 
            self.fs, 
            self.window, 
            nperseg=self.nFFT, 
            noverlap=self.noverlap,
            nfft=self.nFFT, 
            boundary='even', 
            padded=True, 
            return_onesided=False
        )

        alphas = np.zeros((len(f), len(t)))

        # Create array for relative gains based on interpolated frequency response
        gains=characteristic(f)

        # Apply gain deviations based on gains array.
        for i in range(len(t)):
            alphas[:, i] = gains

        # Set lower bound for attenuation
        alphas[alphas < -100] = -100


        # Get calculated coeffs and apply them to the STFT of the RIR
        RIR_TF_processed = RIR_TF*np.power(10, (alphas/20))

        # Transform processed STFT of RIR back to time domain
        _, RIR_processed = istft(
            RIR_TF_processed, 
            self.fs, 
            self.window, 
            nperseg=self.nFFT, 
            noverlap=self.noverlap, 
            nfft=self.nFFT
        )
        
        RIR_processed = RIR_processed[0:len(RIR)]
        return RIR_processed

    def apply(self, IR):
        return self.apply_characteristic(IR)
