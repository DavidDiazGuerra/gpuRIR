import gpuRIR

def generate_RIR(param):
    '''
    Generates RIRs from the gpuRIR library.

    :return: Receiver channels (mono)
    '''
    gpuRIR.activateMixedPrecision(False)
    gpuRIR.activateLUT(False)

    RIRs = gpuRIR.simulateRIR(param.room_sz, param.beta, param.pos_src, param.pos_rcv, param.nb_img,
                              param.Tmax, param.fs, Tdiff=param.Tdiff,
                              orV_src=param.orV_src, orV_rcv=param.orV_rcv,
                              spkr_pattern=param.spkr_pattern, mic_pattern=param.mic_pattern)

    # return receiver channels (mono), number of receivers, sampling frequency and bit depth from RIRs.
    return RIRs[0]