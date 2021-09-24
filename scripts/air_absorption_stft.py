import numpy as np
from scripts.filter import FilterStrategy
from scipy.signal import istft, stft
import air_absorption_calculation as aa


class STFT(FilterStrategy): # T=25, hr=50, ps=1, fs=44100, nFFT=256, noverlap=int(256*0.75), window='hanning'
    def STFT_air_absorption(self, RIR, params):

      # Calculate STFT of RIR (like spectrogram)
        f, t, RIR_TF = stft(RIR, params.fs, params.window, nperseg=params.nFFT, noverlap=params.noverlap,
                            nfft=params.nFFT, boundary='even', padded=True, return_onesided=False)

      # Get air absorption coeffs for given configuration, coeffs are in dB/meter
        alphas_t0, alpha_iso, c, c_iso = aa.air_absorption(
            f, params.T, params.hr, params.ps)

      # Get air absorption coeffs over distance/time for each band
        alphas = np.zeros((len(f), len(t)))
        for ii in range(len(t)):
            alphas[:, ii] = -alphas_t0*t[ii]*c

      # Set lower bound for attenuation
        alphas[alphas < -100] = -100

      # Get linear absorption coeffs and apply them to the STFT of the RIR
        RIR_TF_processed = RIR_TF*np.power(10, (alphas/20))

      # Transform processed STFT of RIR back to time domain
        _, RIR_processed = istft(RIR_TF_processed, params.fs, params.window,
                                 nperseg=params.nFFT, noverlap=params.noverlap, nfft=params.nFFT)
        RIR_processed = RIR_processed[0:len(RIR)]
        return RIR_processed

    def apply(self, IR, params):
        return self.STFT_air_absorption(IR, params)
