''' Room Impulse Responses (RIRs) simulations with GPU acceleration.
'''
from __future__ import division

import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve

from gpuRIR_bind import gpuRIR_bind

__all__ = ["polar_patterns", "beta_SabineEstimation", "att2t_SabineEstimator", "t2n", "simulateRIR", "simulateTrajectory", "activate_mixed_precision", "activate_lut"]

polar_patterns =	{
  "omni": 0,
  "homni": 1,
  "card": 2,
  "hypcard": 3,
  "subcard": 4,
  "bidir": 5
}

def beta_SabineEstimation(room_sz, T60, abs_weights=[1.0]*6):	
    '''  Estimation of the reflection coefficients needed to have the desired reverberation time.
    
    Parameters
    ----------
    room_sz : 3 elements list or numpy array
        Size of the room (in meters).
    T60 : float
        Reverberation time of the room (seconds to reach 60dB attenuation).
    abs_weights : array_like with 6 elements, optional
        Absoprtion coefficient ratios of the walls (the default is [1.0]*6).
    
    Returns
    -------
    ndarray with 6 elements
        Reflection coefficients of the walls as $[beta_{x0}, beta_{x1}, beta_{y0}, beta_{y1}, beta_{z0}, beta_{z1}]$,
        where $beta_{x0}$ is the coeffcient of the wall parallel to the x axis closest
        to the origin of coordinates system and $beta_{x1}$ the farthest.

    '''

    def t60error(x, T60, room_sz, abs_weights):
        alpha = x * abs_weights
        Sa = (alpha[0]+alpha[1]) * room_sz[1]*room_sz[2] + \
            (alpha[2]+alpha[3]) * room_sz[0]*room_sz[2] + \
            (alpha[4]+alpha[5]) * room_sz[0]*room_sz[1]
        V = np.prod(room_sz)
        if Sa == 0: return T60 - 0 # Anechoic chamber
        return abs(T60 - 0.161 * V / Sa) # Sabine's formula

    abs_weights /= np.array(abs_weights).max()
    result = minimize(t60error, 0.5, args=(T60, room_sz, abs_weights), bounds=[[0, 1]])
    return np.sqrt(1 - result.x * abs_weights).astype(np.float32)


def t60_SabineEstimation(room_sz, beta):
    '''  Estimation of the T60 of a room given its size and its reflection coefficients using the Sabine equation.
    
    Parameters
    ----------
    room_sz : 3 elements list or numpy array
        Size of the room (in meters).
    beta : ndarray with 6 elements
        Reflection coefficients of the walls as $[beta_{x0}, beta_{x1}, beta_{y0}, beta_{y1}, beta_{z0}, beta_{z1}]$,
        where $beta_{x0}$ is the coeffcient of the wall parallel to the x axis closest
        to the origin of coordinates system and $beta_{x1}$ the farthest.
    
    Returns
    -------
    float
        Estimated reverberation time of the room (seconds to reach 60dB attenuation).

    '''

    alpha = 1 - beta**2
    Sa = (alpha[0]+alpha[1]) * room_sz[1]*room_sz[2] + \
        (alpha[2]+alpha[3]) * room_sz[0]*room_sz[2] + \
        (alpha[4]+alpha[5]) * room_sz[0]*room_sz[1]
    V = np.prod(room_sz)
    return 0 if Sa == 0 else 0.161 * V / Sa


def att2t_SabineEstimator(att_dB, T60):
    ''' Estimation of the time for the RIR to reach a certain attenuation using the Sabine model.

    Parameters
    ----------
    att_dB : float
        Desired attenuation (in dB).
    T60 : float
        Reverberation time of the room (seconds to reach 60dB attenuation).

    Returns
    -------
    float
        Time (in seconds) to reach the desired attenuation.

    '''
    return att_dB/60.0 * T60

def t2n(T, rooms_sz, c=343.0):
    ''' Estimation of the number of images needed for a correct RIR simulation.

    Parameters
    ----------
    T : float
        RIRs length (in seconds).
    room_sz : 3 elements list or numpy array
        Size of the room (in meters).
    c : float, optional
        Speed of sound (the default is 343.0).

    Returns
    -------
    3 elements list of integers
        The number of images sources to compute in each dimension.

    '''
    nb_img = 2 * T / (np.array(rooms_sz) / c)
    return [ int(n) for n in np.ceil(nb_img) ]

def simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs, Tdiff=None, spkr_pattern="omni", mic_pattern="omni", orV_src=None, orV_rcv=None, c=343.0):
    ''' Room Impulse Responses (RIRs) simulation using the Image Source Method (ISM).

    Parameters
    ----------
    room_sz : array_like with 3 elements
        Size of the room (in meters).
    beta : array_like with 6 elements
        Reflection coefficients of the walls as $[beta_{x0}, beta_{x1}, beta_{y0}, beta_{y1}, beta_{z0}, beta_{z1}]$,
        where $beta_{x0}$ and $beta_{x1}$ are the reflection coefficents of the walls orthogonal to the x axis at
        x=0 and x=room_sz[0], respectively.
    pos_src, pos_rcv : ndarray with 2 dimensions and 3 columns
        Position of the sources and the receivers (in meters).
    nb_img : array_like with 3 integer elements
        Number of images to simulate in each dimension.
    Tmax : float
        RIRs length (in seconds).
    fs : float
        RIRs sampling frequency (in Hertz).
    Tdiff : float, optional
        Time (in seconds) when the ISM is replaced by a diffuse reverberation model.
        Default is Tmax (full ISM simulation).
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
    orV_src : ndarray with 2 dimensions and 3 columns or None, optional
        Orientation of the sources as vectors pointing in the same direction.
        None (default) is only valid for omnidirectional patterns.
    orV_rcv : ndarray with 2 dimensions and 3 columns or None, optional
        Orientation of the receivers as vectors pointing in the same direction.
        None (default) is only valid for omnidirectional patterns.
    c : float, optional
        Speed of sound [m/s] (the default is 343.0).

    Returns
    -------
    3D ndarray
        The first axis is the source, the second the receiver and the third the time.

    Warnings
    --------
    Asking for too much and too long RIRs (specially for full ISM simulations) may exceed
    the GPU memory and crash the kernel.

    '''

    assert not ((pos_src >= room_sz).any() or (pos_src <= 0).any()), "The sources must be inside the room"
    assert not ((pos_rcv >= room_sz).any() or (pos_rcv <= 0).any()), "The receivers must be inside the room"
    assert Tdiff is None or Tdiff <= Tmax, "Tmax must be equal or greater than Tdiff"
    assert mic_pattern in polar_patterns, "mic_pattern must be omni, homni, card, hypcard, subcard or bidir"
    assert mic_pattern == "omni" or orV_rcv is not None, "the mics are not omni but their orientation is undefined"
    assert spkr_pattern in spkr_pattern, "spkr_pattern must be omni, homni, card, hypcard, subcard or bidir"
    assert spkr_pattern == "omni" or orV_src is not None, "the sources are not omni but their orientation is undefined"
    assert min(nb_img) > 0, "The number of images must be greater than 0 in every dimension"


    pos_src = pos_src.astype('float32', order='C', copy=False)
    pos_rcv = pos_rcv.astype('float32', order='C', copy=False)

    if Tdiff is None: Tdiff = Tmax
    if mic_pattern is None: mic_pattern = "omni"
    if spkr_pattern is None: spkr_pattern = "omni"
    if orV_rcv is None: orV_rcv = np.zeros_like(pos_rcv)
    else: orV_rcv = orV_rcv.astype('float32', order='C', copy=False)
    if orV_src is None: orV_src = np.zeros_like(pos_src)
    else: orV_src = orV_src.astype('float32', order='C', copy=False)

    amp, tau = gpuRIR_bind_simulator.compute_echogram_bind(room_sz, beta,
                                                           pos_src, pos_rcv, orV_src, orV_rcv,
                                                           polar_patterns[spkr_pattern], polar_patterns[mic_pattern],
                                                           nb_img, Tdiff, Tmax, fs, c)
    dp_amp = amp[..., nb_img[0]//2, nb_img[1]//2, nb_img[2]//2]
    dp_tau = tau[..., nb_img[0]//2, nb_img[1]//2, nb_img[2]//2]
    assert (dp_amp == amp.max((2,3,4))).all()  # TODO: Remove after checking that this never happens
    assert (dp_tau == tau.min((2,3,4))).all()  # TODO: Remove after checking that this never happens
    amp = amp.reshape(amp.shape[0], amp.shape[1], -1)
    tau = tau.reshape(tau.shape[0], tau.shape[1], -1)

    T60 = t60_SabineEstimation(room_sz, beta)
    rir = gpuRIR_bind_simulator.render_echogram_bind(amp, tau, dp_tau, Tdiff, Tmax, T60, fs)
    return rir

def simulateTrajectory(source_signal, RIRs, timestamps=None, fs=None):
    ''' Filter an audio signal by the RIRs of a motion trajectory recorded with a microphone array.

    Parameters
    ----------
    source_signal : array_like
        Signal of the moving source.
    RIRs : 3D ndarray
        Room Impulse Responses generated with simulateRIR.
    timestamps : array_like, optional
        Timestamp of each RIR [s]. By default, the RIRs are equispaced through the trajectory.
    fs : float, optional
        Sampling frequency (in Hertz). It is only needed for custom timestamps.

    Returns
    -------
    2D ndarray
        Matrix with the signals captured by each microphone in each column.

    '''
    nSamples = len(source_signal)
    nPts = RIRs.shape[0]
    nRcv = RIRs.shape[1]
    lenRIR = RIRs.shape[2]

    assert timestamps is None or fs is not None, "fs must be indicated for custom timestamps"
    assert timestamps is None or timestamps[0] == 0, "The first timestamp must be 0"
    if timestamps is None:
        fs = nSamples / nPts
        timestamps = np.arange(nPts)

    w_ini = np.append((timestamps*fs).astype(int), nSamples)
    w_len = np.diff(w_ini)
    segments = np.zeros((nPts, w_len.max()))
    for n in range(nPts):
        segments[n,0:w_len[n]] = source_signal[w_ini[n]:w_ini[n+1]]
    segments = segments.astype('float32', order='C', copy=False)
    convolution = gpuRIR_bind_simulator.gpu_conv(segments, RIRs)

    filtered_signal = np.zeros((nSamples+lenRIR-1, nRcv))
    for m in range(nRcv):
        for n in range(nPts):
            filtered_signal[w_ini[n] : w_ini[n+1]+lenRIR-1, m] += convolution[n, m, 0:w_len[n]+lenRIR-1]

    return filtered_signal

def activateMixedPrecision(activate=True):
    ''' Activate the mixed precision mode, only for Pascal GPU architecture or superior.

    Parameters
    ----------
    activate : bool, optional
        True for activate and Flase for deactivate. True by default.

    '''
    gpuRIR_bind_simulator.activate_mixed_precision_bind(activate)

def activateLUT(activate=True):
    ''' Activate the lookup table for the sinc computations.

    Parameters
    ----------
    activate : bool, optional
        True for activate and Flase for deactivate. True by default.

    '''
    gpuRIR_bind_simulator.activate_lut_bind(activate)

# Create the simulator object when the module is loaded
gpuRIR_bind_simulator = gpuRIR_bind()
