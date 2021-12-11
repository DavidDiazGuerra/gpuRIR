import numpy as np
import numpy.matlib as ml
import gpuRIR


class RoomParameters:
    ''' Room parameter class for gpuRIR.
    '''
    def __init__(
        self,
        room_sz,
        pos_src,
        pos_rcv,
        orV_src,
        orV_rcv,
        spkr_pattern,
        mic_pattern,
        T60,
        att_diff,
        att_max,
        fs,
        bit_depth,
        abs_weights = 5 * [0.9] + [0.5],
        wall_materials = None,
        head_position = None, # Only for HRTF
        head_direction = None, # Only for HRTF
        beta = None):
        ''' Instantiates a consolidated room parameter object for gpuRIR.

        Parameters
        ----------
        room_sz : 3D ndarray
            Size of room [m]
        pos_src : 3D ndarray
            Position of signal source [m]
        pos_rcv : 3D ndarray
            Position of singal receiver [m]
        orV_src : ndarray with 2 dimensions and 3 columns or None, optional
            Orientation of the sources as vectors pointing in the same direction.
            None (default) is only valid for omnidirectional patterns.
        orV_rcv : ndarray with 2 dimensions and 3 columns or None, optional
            Orientation of the receivers as vectors pointing in the same direction.
            None (default) is only valid for omnidirectional patterns.
        spkr_pattern : {"omni", "homni", "card", "hypcard", "subcard", "bidir"}, optional
            Polar pattern of the sources (the same for all of them).
                "omni" : Omnidireccional (default).
                "homni": Half omnidirectional, 1 in front of the microphone, 0 backwards.
                "card": Cardioid.
                "hypcard": Hypercardioid.
                "subcard": Subcardioid.
                "bidir": Bidirectional, a.k.a. figure 8.
        mic_pattern : {"omni", "homni", "card", "hypcard", "subcard", "bidir"}, optional
            Polar pattern of the receivers (the same for all of them).
                "omni" : Omnidireccional (default).
                "homni": Half omnidirectional, 1 in front of the microphone, 0 backwards.
                "card": Cardioid.
                "hypcard": Hypercardioid.
                "subcard": Subcardioid.
                "bidir": Bidirectional, a.k.a. figure 8.
	    T60 : float
		    Reverberation time of the room (seconds to reach 60dB attenuation).
        att_diff : 
            TODO
        att_max : 
            TODO
        fs : float
            RIRs sampling frequency (in Hertz).
        bit_depth : int
            Bit depth of source sound data.
        abs_weights : array_like with 6 elements, optional
            List of six float elements, determining absorption coefficient (0..1)
        wall_materials : array_like with 6 elements, optional
            List of six Materials objects, representing virtual room materials.
        head_position : 3D ndarray, optional
            Only for HRTF. Position the head is located at.
        head_direction : 3D ndarray, optional
            Only for HRTF. Steering vector of the head, determining the direction it is pointing towards.
        beta : array_like with 6 elements, optional
            Reflection coefficients of the walls as $[beta_{x0}, beta_{x1}, beta_{y0}, beta_{y1}, beta_{z0}, beta_{z1}]$,
            where $beta_{x0}$ and $beta_{x1}$ are the reflection coefficents of the walls orthogonal to the x axis at
            x=0 and x=room_sz[0], respectively.
        '''

        self.room_sz = room_sz
        self.pos_src = np.array(pos_src)
        self.pos_rcv = np.array(pos_rcv)
        self.orV_src = ml.repmat(np.array(orV_src), len(pos_src), 1)
        self.orV_rcv = ml.repmat(np.array(orV_rcv), len(pos_rcv), 1)
        self.spkr_pattern = spkr_pattern
        self.mic_pattern = mic_pattern
        self.T60 = T60
        self.fs = fs
        self.bit_depth = bit_depth
        self.abs_weights = abs_weights
        self.wall_materials = wall_materials


        if head_position is None:
            self.head_position = pos_rcv
        else:  # Only for HRTF
            self.head_position = head_position

        if head_direction is None:
            self.head_direction = orV_rcv
        else:  # Only for HRTF
            self.head_direction = head_direction

        if beta is None:
            # Switch between self-determined wall coefficients used for frequency dependent wall absorption coefficients
            self.beta = gpuRIR.beta_SabineEstimation(room_sz, T60, abs_weights=abs_weights)  # Reflection coefficients
        else:
            self.beta = beta

        # Time to start the diffuse reverberation model [s]
        self.Tdiff = gpuRIR.att2t_SabineEstimator(att_diff, T60)

        # Time to stop the simulation [s]
        self.Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)

        # Number of image sources in each dimension
        self.nb_img = gpuRIR.t2n(self.Tdiff, room_sz)
