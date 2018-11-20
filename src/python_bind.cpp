
#include <vector>
#include <assert.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "gpuRIR_cuda.h"

namespace py = pybind11;

py::array simulateRIR_bind(std::vector<scalar_t> room_sz, // Size of the room [m]
						   std::vector<scalar_t> beta, // Reflection coefficients
						   py::array_t<scalar_t, py::array::c_style> pos_src, // positions of the sources [m]
						   py::array_t<scalar_t, py::array::c_style> pos_rcv, // positions of the receivers [m]
						   py::array_t<scalar_t, py::array::c_style> orV_rcv, // orientation of the receivers
						   micPattern mic_pattern, // Polar pattern of the receivers (see gpuRIR_cuda.h)
						   std::vector<int> nb_img, // Number of sources in each dimension
						   scalar_t Tdiff, // Time when the ISM is replaced by a diffusse reverberation model [s]
						   scalar_t Tmax, // RIRs length [s]
						   scalar_t Fs, // Sampling frequency [Hz]
						   scalar_t c=343.0 // Speed of sound [m/s]
						   ) 
{
	py::buffer_info info_pos_src = pos_src.request();
	py::buffer_info info_pos_rcv = pos_rcv.request();
	py::buffer_info info_orV_rcv = orV_rcv.request();
	
	// Check input sizes
	assert(room_sz.size() == 3);
	assert(beta.size() == 6);
	assert(nb_img.size() == 3);
	assert(pos_src.ndim == 2);
	assert(pos_rcv.ndim == 2);
	assert(orV_rcv.ndim == 2);
	assert(info_pos_src.shape[1] == 3);
	assert(info_pos_rcv.shape[1] == 3);
	assert(info_orV_rcv.shape[1] == 3);
	assert(info_pos_rcv.shape[0] == info_orV_rcv.shape[0]);
	int M_src = info_pos_src.shape[0];
	int M_rcv = info_pos_rcv.shape[0];
	
	scalar_t* rir = cuda_simulateRIR(&room_sz[0], &beta[0], 
									 (scalar_t*) info_pos_src.ptr, M_src, 
									 (scalar_t*) info_pos_rcv.ptr, (scalar_t*) info_orV_rcv.ptr, mic_pattern, M_rcv, 
									 &nb_img[0], Tdiff, Tmax, Fs, c);
	
	int nSamples = ceil(Tmax*Fs);
	std::vector<int> shape = {M_src, M_rcv, nSamples};
	std::vector<size_t> strides = {M_rcv*nSamples*sizeof(scalar_t), nSamples*sizeof(scalar_t), sizeof(scalar_t)};
	return py::array(py::buffer_info(rir, sizeof(scalar_t), py::format_descriptor<scalar_t>::format(), 3, shape, strides)); 

}

PYBIND11_MODULE(gpuRIR_bind,m)
{
  m.doc() = "Room Impulse Response (RIR) simulation through Image Source Method (ISM) with GPU acceleration.";

  m.def("simulateRIR_bind", &simulateRIR_bind, "RIR simulation", py::arg("room_sz"), py::arg("beta"), py::arg("pos_src"), 
		py::arg("pos_rcv"), py::arg("orV_rcv"), py::arg("mic_pattern"), py::arg("nb_img"), py::arg("Tdiff"), py::arg("Tmax"), 
		py::arg("Fs"), py::arg("c")=343.0f );
}
