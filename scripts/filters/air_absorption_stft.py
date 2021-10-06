from scipy.signal import istft, stft
from filters.filter import FilterStrategy
import filters.air_absorption_calculation as aa

import numpy as np


class STFT(FilterStrategy):
    def __init__(self, T=25, hr=50, ps=1, fs=44100, nFFT=256, noverlap=int(256*0.75), window='hanning'):
        self.T = T
        self.hr = hr
        self.ps = ps
        self.fs = fs
        self.nFFT = nFFT
        self.noverlap = noverlap
        self.window = window
        self.NAME = "STFT"

    def STFT_air_absorption(self, RIR):

        # Calculate STFT of RIR (like spectrogram)
        f, t, RIR_TF = stft(RIR, self.fs, self.window, nperseg=self.nFFT, noverlap=self.noverlap,
                            nfft=self.nFFT, boundary='even', padded=True, return_onesided=False)

        # Get air absorption coeffs for given configuration, coeffs are in dB/meter
        alphas_t0, _, c, _ = aa.air_absorption(f, self.T, self.hr, self.ps)

        # Get air absorption coeffs over distance/time for each band
        alphas = np.zeros((len(f), len(t)))
        for ii in range(len(t)):
            alphas[:, ii] = -alphas_t0*t[ii]*c

        # Set lower bound for attenuation
        alphas[alphas < -100] = -100

        # Get linear absorption coeffs and apply them to the STFT of the RIR
        RIR_TF_processed = RIR_TF*np.power(10, (alphas/20))

        # Transform processed STFT of RIR back to time domain
        _, RIR_processed = istft(RIR_TF_processed, self.fs, self.window,
                                 nperseg=self.nFFT, noverlap=self.noverlap, nfft=self.nFFT)
        RIR_processed = RIR_processed[0:len(RIR)]
        return RIR_processed

    def apply(self, IR):
        return self.STFT_air_absorption(IR)
