
# gpuRIR

**gpuRIR** is a free and open-source Python library for Room Impulse Response (RIR) simulation using the Image Source Method (ISM) with GPU acceleration. It can compute the RIRs between several source and receivers positions in parallel using CUDA GPUs. It is approximately 100 times faster than CPU implementations [[1]](#references).

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [License](#license)
- [Documentation](#documentation)
  * [`simulateRIR`](#simulaterir)
  * [`simulateTrajectory`](#simulatetrajectory)
  * [`beta_SabineEstimation`](#beta_sabineestimation)
  * [`att2t_SabineEstimator`](#att2t_sabineestimator)
  * [`t2n`](#t2n)
- [References](#references)


## Prerequisites

* **OS**: It has been tested on GNU/Linux systems (Ubuntu and centOS) and Windows 10. Please, let me know if you successfully [install it on Mac OSX systems](https://github.com/DavidDiazGuerra/gpuRIR/issues/3).

* **Compilers**: To install the package you will need the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (it has been tested with the release 8.0 and 10.0 but it should work fine with any version that includes cuRAND) and a C++11 compiler, such as [GCC](https://gcc.gnu.org/) or [MSVC++](https://visualstudio.microsoft.com).

* **CMake**: Finally, you will need, at least, the version 3.12 of [CMake](https://cmake.org/).  You can easily get it by `pip install cmake`.

* **Python**: It has been tested in Python 3, but should work fine with Python 2.


## Installation
You can use `pip` to install **gpuRIR** from our repository through `pip install  https://github.com/DavidDiazGuerra/gpuRIR/zipball/master`. You can also clone or download our repository and run `python setup.py install`.


## License

The library is subject to AGPL-3.0 license and comes with no warranty. If you find it useful for your research work, please, acknowledge it to [[1]](#references).


## Documentation

### `simulateRIR`

Room Impulse Responses (RIRs) simulation using the Image Source Method (ISM). For further details see [[1]](#references).

#### Parameters

* **room_sz** : *array_like with 3 elements.*
        Size of the room (in meters).
* **beta** : *array_like with 6 elements.*
        Reflection coefficients of the walls as $[beta_{x0}, beta_{x1}, beta_{y0}, beta_{y1}, beta_{z0}, beta_{z1}]$, where $beta_{x0}$ is the coeffcient of the wall parallel to the x axis closest to the origin of coordinates system and $beta_{x1}$ the farthest.
* **pos_src**, **pos_rcv** : *ndarray with 2 dimensions and 3 columns.*
        Position of the sources and the receivers (in meters).
* **nb_img** : *array_like with 3 integer elements*
        Number of images to simulate in each dimension.
* **Tmax** : *float*
        RIRs length (in seconds).
* **fs** : *float*
        RIRs sampling frequency (in Hertz).
* **Tdiff** : *float, optional*
        Time (in seconds) when the ISM is replaced by a diffuse reverberation model. Default is Tmax (full ISM simulation).
* **mic_pattern** : *{"omni", "homni", "card", "hypcard", "subcard", "bidir"}, optional.*
        Polar pattern of the receivers (the same for all of them).
	* *"omni"* : Omnidireccional (default).
	* *"homni"*: Half omnidireccional, 1 in front of the microphone, 0 backwards.
	* *"card"*: Cardioid.
	* *"hypcard"*: Hypercardioid.
	* *"subcard"*: Subcardioid.
	* *"bidir"*: Bidirectional, a.k.a. figure 8.
* **orV_rcv** : *ndarray with 2 dimensions and 3 columns or None, optional.*
        Orientation of the receivers as vectors pointing in the same direction. None (default) is only valid for omnidireccional patterns.
* **c** : *float, optional.*
        Speed of sound (in m/s). The default is 343.0.

#### Returns

*3D ndarray*
        The first axis is the source, the second the receiver and the third the time.

#### Warnings

Asking for too much and too long RIRs (specially for full ISM simulations) may exceed the GPU memory and crash the kernel.

### `simulateTrajectory`

Filter an audio signal by the RIRs of a motion trajectory recorded with a microphone array.

#### Parameters

* **source_signal** : *array_like.*
	Signal of the moving source.
* **RIRs** : *3D ndarray*
	Room Impulse Responses generated with simulateRIR.
* **timestamps** : *array_like, optional*
	Timestamp of each RIR [s]. By default, the RIRs are equispaced through the trajectory.
* **fs** : *float, optional*
	Sampling frequency (in Hertz). It is only needed for custom timestamps.

#### Returns

*2D ndarray*
	Matrix with the signals captured by each microphone in each column.

### `beta_SabineEstimation`

Estimation of the reflection coefficients needed to have the desired reverberation time.

#### Parameters

* **room_sz** : *3 elements list or numpy array.*
        Size of the room (in meters). 
* **T60** : *float.*
        Reverberation time of the room (seconds to reach 60dB attenuation).        
* **abs_weights** : *array_like with 6 elements, optional.*
        Absorption coefficient ratios of the walls (the default is [1.0]*6). 

#### Returns

*ndarray with 6 elements.*
        Reflection coefficients of the walls as $[beta_{x0}, beta_{x1}, beta_{y0}, beta_{y1}, beta_{z0}, beta_{z1}]$, where $beta_{x0}$ is the coeffcient of the wall parallel to the x axis closest to the origin of coordinates system and $beta_{x1}$ the farthest.

### `att2t_SabineEstimator`

Estimation of the time for the RIR to reach a certain attenuation using the Sabine model.

#### Parameters

* **att_dB** : *float.*
        Desired attenuation (in dB).
* **T60** : *float.*
        Reverberation time of the room (seconds to reach 60dB attenuation).

#### Returns

*float.*
        Time (in seconds) to reach the desired attenuation.

### `t2n`

Estimation of the number of images needed for a correct RIR simulation.

#### Parameters

* **T** : *float.*
        RIRs length (in seconds).
* **room_sz** : *3 elements list or numpy array.*
        Size of the room (in meters).
* **c** : *float, optional.*
        Speed of sound (the default is 343.0).

#### Returns

*3 elements list of integers.*
        The number of images sources to compute in each dimension.


## References

[1] D. Diaz-Guerra, A. Miguel, J.R. Beltran, "gpuRIR: A python library for Room Impulse Response simulation with GPU acceleration," [[arXiv preprint](https://arxiv.org/abs/1810.11359)].



