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
        wall_coeffs,
        beta_direct=False):

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
        self.wall_coeffs = wall_coeffs

        # Switch between self-determined wall coefficients used for frequency dependent wall absorption coefficients
        if not beta_direct:
            self.beta = gpuRIR.beta_SabineEstimation(
                room_sz, T60, abs_weights=wall_coeffs)  # Reflection coefficients
        #else:
            #self.beta = 6*[1.] - wall_coeffs

        # Time to start the diffuse reverberation model [s]
        self.Tdiff = gpuRIR.att2t_SabineEstimator(att_diff, T60)

        # Time to stop the simulation [s]
        self.Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)

        # Number of image sources in each dimension
        self.nb_img = gpuRIR.t2n(self.Tdiff, room_sz)
