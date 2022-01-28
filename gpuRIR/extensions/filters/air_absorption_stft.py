from gpuRIR.extensions.filters.filter import FilterStrategy
import gpuRIR.extensions.filters.air_absorption_calculation as aa

import numpy as np
from scipy.signal import istft, stft


class AirAbsSTFT(FilterStrategy):
    '''
    Applies air absorption attenuation using short-time fourier transformation (STFT). 
    Splits the frequency bands into defined amount of divisions, applies air absorption and combines the bands back into one RIR.

    Reference:
    Hamilton Brian. Adding air attenuation to simulated room impulse responses: A modal approach, 2021. URL: https://arxiv.org/pdf/2107.11871.pdf
    '''

    def __init__(self, T=25, hr=50, ps=1, fs=44100, nFFT=256, noverlap=int(256*0.75), window='hanning'):
        ''' Instatiates STFT-driven air absorption.
        For further reference about parameters see scipy.signal.stft
        '''
        self.T = T
        self.hr = hr
        self.ps = ps
        self.fs = fs
        self.nFFT = nFFT
        self.noverlap = noverlap
        self.window = window
        self.NAME = "STFT_air_abs"

    def STFT_air_absorption(self, RIR):
        ''' Applies air absorption to RIR using STFT.

        Parameters
	    ----------
        RIR : 2D ndarray
            Room impulse response array.
        
        Returns
	    -------
        2D ndarray
            Processed Room impulse response array.
        '''
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
        '''
        Calls method to apply air absorption to RIR using STFT.
        
        Parameters
	    ----------
        IR : 2D ndarray
            Room impulse response array.

        Returns
	    -------
        2D ndarray
            Processed Room impulse response array.
        '''
        return self.STFT_air_absorption(IR)
