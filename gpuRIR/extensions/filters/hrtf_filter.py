import numpy as np

from gpuRIR.extensions.filters.filter import FilterStrategy
from gpuRIR.extensions.hrtf.hrtf_rir import HRTF_RIR


class HRTF_Filter(FilterStrategy):
    ''' Head related transfer function to simulate psychoacoustic effects of a human head and upper body.
    
    Reference: Algazi V.R., Duda R.O., Thompson D.M. and Avendano C. The CIPIC
    HRTF database. In Proceedings of the 2001 IEEE Workshop on the Applications of
    Signal Processing to Audio and Acoustics (Cat. No.01TH8575) (2001), pp. 99–102.
    '''
    # 90 degree angle in radiants
    ANGLE_90 = np.pi/2

    # 180 degree angle in radiants
    ANGLE_180 = np.pi

    def __init__(self, channel, params, verbose=False):
        ''' Initialized HRTF filter.

        Parameters
        ----------
        channel : str
            Determines if left or right channel is being processed. Either 'l' or 'r'.
        params : RoomParameters
            Abstracted gpuRIR parameter class object.
        verbose : bool, optional
            Terminal output for debugging or further information
        '''
        self.channel = channel
        self.NAME = "HRTF"
        self.params = params
        self.hrtf_rir = HRTF_RIR()
        self.verbose = verbose

    @staticmethod
    def find_angle(u, v):
        ''' Find angle between two vectors on a 2D plane.

        Parameters
        ----------
        u : ndarray
            Vector with two elements
        v : ndarray
            Vector with two elements

        Returns
        -------
        float
            Scalar angle in radiants.
        
        '''
        norm_product = (np.linalg.norm(u) * np.linalg.norm(v))

        if norm_product != 0:
            return np.arccos((u @ v) / norm_product)

        return 0

    # Find elevation between head and source

    @staticmethod
    def calculate_elevation(pos_src, pos_rcv, head_direction):
        ''' Calculates elevation between head position / direction and signal source position.

        Parameters
        ----------
        pos_src : 3D ndarray
            Position of signal source.
        pos_rcv : 3D ndarray
            Position of signal receiver (center of head).
        head_direction : 3D ndarray
            Direction in which the head is pointing towards.

        Returns
        -------
        float
            Elevation angle between head position / direction and signal source.
        '''
        # Height of source
        opposite = np.abs(pos_src[2] - pos_rcv[2])

        # Length of floor distance between head and source
        adjacent = np.linalg.norm(
            np.array([pos_src[0], pos_src[1]]) - np.array([pos_rcv[0], pos_rcv[1]]))

        # Find elevation between head and source positions
        if adjacent != 0:
            el_rcv_src = np.arctan(opposite / adjacent)
        else:
            el_rcv_src = np.arctan(np.inf)

        # Edge case if source is below head
        if pos_rcv[2] > pos_src[2]:
            el_rcv_src = -el_rcv_src

        # Height of receiver
        opposite = np.abs(head_direction[2])

        # Length of floor distance between head and head direction vector
        adjacent = np.linalg.norm(np.array([head_direction[0], head_direction[1]]))

        # Calculate elevation between head and head direction
        el_rcv_dir = np.arctan(opposite / adjacent)

        # Edge case if source is below head
        if pos_rcv[2] > pos_src[2]:
            elevation_angle = el_rcv_src + el_rcv_dir
        else:
            elevation_angle = el_rcv_src - el_rcv_dir

        # Edge case if source is behind head
        angle, _, _ = HRTF_Filter.vector_between_points(
            pos_src, pos_rcv, head_direction)
        if angle > HRTF_Filter.ANGLE_90:
            # Source is behind head
            elevation_angle = HRTF_Filter.ANGLE_180 - elevation_angle

        # Subtract elevation between head and source and between head and head direction
        return elevation_angle

    @staticmethod
    def vector_between_points(pos_src, pos_rcv, head_direction):
        ''' Calculates a vector between two points in a 2D plane.

        Parameters
        ----------
        pos_src : 3D ndarray
            Position of signal source.
        pos_rcv : 3D ndarray
            Position of signal receiver (center of head).
        head_direction : 3D ndarray
            Direction in which the head is pointing towards.

        Returns
        -------
        float
            Scalar angle in radiants.
        2D ndarray
            2D Vector between head and signal source.
        2D ndarray
            2D vector of head direction.

        '''
        # 3D vector from head position (origin) to source
        head_to_src = pos_src - pos_rcv

        # Extract 2D array from 3D
        head_to_src = np.array([head_to_src[0], head_to_src[1]])
        # Extract 2D array from 3D
        headdir_xy = [head_direction[0], head_direction[1]]

        # Return angle using trigonometry
        return HRTF_Filter.find_angle(headdir_xy, head_to_src), head_to_src, headdir_xy

    @staticmethod
    def calculate_azimuth(pos_src, pos_rcv, head_direction):
        ''' Calculates azimuth between head position / direction and signal source position.

        Parameters
        ----------
        pos_src : 3D ndarray
            Position of signal source.
        pos_rcv : 3D ndarray
            Position of signal receiver (center of head).
        head_direction : 3D ndarray
            Direction in which the head is pointing towards.

        Returns
        -------
        float
            Azimuth angle between head position / direction and signal source.
        '''
        # Find angle using trigonometry
        angle, head_to_src, headdir_xy = HRTF_Filter.vector_between_points(
            pos_src, pos_rcv, head_direction)

        # Check if azimuth goes above 90°
        if angle > HRTF_Filter.ANGLE_90:
            angle = np.pi - angle

        # Check left/right. If positive direction is left, if negative direction is right.
        side = np.sign(np.linalg.det([headdir_xy, head_to_src]))

        return angle * (-side)

    def hrtf_convolve(self, IR):
        '''
        Convolves an impulse response (IR) array with a HRTF room impulse response (RIR) retrieved from the CIPIC database.

        Parameters
        ----------
        IR : 2D ndarray
            Room impulse response array.

        Returns
	    -------
        2D ndarray
            Processed Room impulse response array.
        '''
        elevation = self.calculate_elevation(
            self.params.pos_src[0], self.params.head_position, self.params.head_direction)

        if self.verbose:
            print(f"Elevation = {elevation * (180 / np.pi)}")

        azimuth = self.calculate_azimuth(
            self.params.pos_src[0], self.params.head_position, self.params.head_direction)

        if self.verbose:
            print(f"Azimuth = {azimuth * (180 / np.pi)}")

        hrir_channel = self.hrtf_rir.get_hrtf_rir(
            elevation, azimuth, self.channel)

        return np.convolve(IR, hrir_channel, mode='same')

    def apply(self, IR):
        ''' Calls method to apply HRTF filtering on the source data.

        Parameters
	    ----------
        IR : 2D ndarray
            Room impulse response array.

        Returns
	    -------
        2D ndarray
            Processed Room impulse response array.

        '''
        return self.hrtf_convolve(IR)
