import numpy as np
import gpuRIR


class RoomParameters:

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
        head_direction = None): # Only for HRTF

        self.room_sz = room_sz
        self.pos_src = np.array(pos_src)
        self.pos_rcv = np.array(pos_rcv)
        self.orV_src = np.matlib.repmat(np.array(orV_src), len(pos_src), 1)
        self.orV_rcv = np.matlib.repmat(np.array(orV_rcv), len(pos_rcv), 1)
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


        # Switch between self-determined wall coefficients used for frequency dependent wall absorption coefficients
        self.beta = gpuRIR.beta_SabineEstimation(room_sz, T60, abs_weights=abs_weights)  # Reflection coefficients

        # Time to start the diffuse reverberation model [s]
        self.Tdiff = gpuRIR.att2t_SabineEstimator(att_diff, T60)

        # Time to stop the simulation [s]
        self.Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)

        # Number of image sources in each dimension
        self.nb_img = gpuRIR.t2n(self.Tdiff, room_sz)
