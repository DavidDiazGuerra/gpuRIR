import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt

class HRTF_RIR:
    ''' Head related transfer function room impulse response (HRFT RIR) interface.
    
    Reference: Algazi V.R., Duda R.O., Thompson D.M. and Avendano C. The CIPIC
    HRTF database. In Proceedings of the 2001 IEEE Workshop on the Applications of
    Signal Processing to Audio and Acoustics (Cat. No.01TH8575) (2001), pp. 99–102.
    '''
    def __init__(self):
        ''' Instantates interface, creates pre-defined azimuth array.
        '''
        self.azimuths = np.float32([
            -80, -65, -55, -45, -40,
            -35, -30, -25, -20, -15,
            -10, -5, 0, 5, 10,
            15, 20, 25, 30, 35,
            40, 45, 55, 65, 80
        ])

        self.elev_step = (360/64)
        self.elevations = np.arange(-45, 230.625 + self.elev_step, self.elev_step, dtype=np.float32)
        self.hrir = loadmat('gpuRIR/extensions/hrtf/cipic_hrir/hrir_final.mat')

    def azimuth_to_idx(self, azimuth):
        ''' Translates azimuth to idx in order to access CIPIC database.

        Parameters
        ----------
        azimuth : float
            Azimuth value to translate into idx.

        Returns
        -------
        float
            idx
        '''
        return int(np.argmin(np.abs(self.azimuths - azimuth)))

    def elevation_to_idx(self, elevation):
        ''' Translates elevation to idx in order to access CIPIC database.

        Parameters
        ----------
        azimuth : float
            Elevation value to translate into idx.

        Returns
        -------
        float
            idx
        '''
        return int(np.argmin(np.abs(self.elevations - elevation)))


    def get_hrtf_rir(self, elevation, azimuth, channel, visualize=False):
        """ Retrieves Head related transfer function room impulse response (HRFT RIR)
        
        Parameters
        ----------
        elevation : float
            Elevation value between source and receiver.
        azimuth : float
            Azimuth value between source and receiver.
        channel : str
            Left or right channel, represented as 'l' or 'r'.
        visualize : bool
            Visualizes source and receiver in a 3D space.

        Returns
        -------
        2D ndarray
            HRTF RIR
        """
        hrir_channel = self.hrir['hrir_'+channel][:, self.elevation_to_idx(elevation), :]

        if visualize:
            x = hrir_channel[self.azimuth_to_idx(azimuth)]
            plt.title("HRTF Frequency Response")
            plt.rcParams.update({'font.size': 18})
            plt.plot(x)
            plt.xlabel("Sample [n]")
            plt.ylabel("Amplitude")
            plt.show()
        return hrir_channel[self.azimuth_to_idx(azimuth)]