
#include <vector>
#include <assert.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "gpuRIR_cuda.h"

namespace py = pybind11;

class gpuRIR_bind {
    public:
        gpuRIR_bind(bool mPrecision=false, bool lut=true) : mixed_precision(mPrecision), lookup_table(lut), gpuRIR_cuda_simulator(mPrecision, lut) {};

        py::array compute_echogram_bind(std::vector<float>, std::vector<float>, py::array_t<float, py::array::c_style>, py::array_t<float, py::array::c_style>, py::array_t<float, py::array::c_style>, py::array_t<float, py::array::c_style>, polarPattern, polarPattern, std::vector<int> ,float, float, float, float);
        py::array render_echogram_bind(py::array_t<float, py::array::c_style>, py::array_t<float, py::array::c_style>, py::array_t<float, py::array::c_style>, float, float, float, float);
        py::array gpu_conv(py::array_t<float, py::array::c_style>, py::array_t<float, py::array::c_style>);
        bool activate_mixed_precision_bind(bool);
        bool activate_lut_bind(bool);

        bool mixed_precision;
        bool lookup_table;

    private:
        gpuRIR_cuda gpuRIR_cuda_simulator;
};

py::array gpuRIR_bind::compute_echogram_bind(std::vector<float> room_sz, // Size of the room [m]
                                             std::vector<float> beta, // Reflection coefficients
                                             py::array_t<float, py::array::c_style> pos_src, // positions of the sources [m]
                                             py::array_t<float, py::array::c_style> pos_rcv, // positions of the receivers [m]
                                             py::array_t<float, py::array::c_style> orV_src, // orientation of the sources
                                             py::array_t<float, py::array::c_style> orV_rcv, // orientation of the receivers
                                             polarPattern spkr_pattern, // Polar pattern of the sources (see gpuRIR_cuda.h)
                                             polarPattern mic_pattern, // Polar pattern of the receivers (see gpuRIR_cuda.h)
                                             std::vector<int> nb_img, // Number of sources in each dimension
                                             float Tdiff, // Time when the ISM is replaced by a diffusse reverberation model [s]
                                             float Tmax, // RIRs length [s]
                                             float Fs, // Sampling frequency [Hz]
                                             float c=343.0 // Speed of sound [m/s]
                                            )
{
    py::buffer_info info_pos_src = pos_src.request();
    py::buffer_info info_pos_rcv = pos_rcv.request();
    py::buffer_info info_orV_src = orV_src.request();
    py::buffer_info info_orV_rcv = orV_rcv.request();

    // Check input sizes
    assert(room_sz.size() == 3);
    assert(beta.size() == 6);
    assert(nb_img.size() == 3);
    assert(pos_src.ndim == 2);
    assert(pos_rcv.ndim == 2);
    assert(orV_src.ndim == 2);
    assert(orV_rcv.ndim == 2);
    assert(info_pos_src.shape[1] == 3);
    assert(info_pos_rcv.shape[1] == 3);
    assert(info_orV_src.shape[1] == 3);
    assert(info_orV_rcv.shape[1] == 3);
    assert(info_pos_src.shape[0] == info_orV_src.shape[0]);
    assert(info_pos_rcv.shape[0] == info_orV_rcv.shape[0]);
    int M_src = info_pos_src.shape[0];
    int M_rcv = info_pos_rcv.shape[0];

    std::array<float*, 2> echogram = gpuRIR_cuda_simulator.cuda_compute_echogram(&room_sz[0], &beta[0], c,
                                                                                  (float*) info_pos_src.ptr,
                                                                                  (float*) info_orV_src.ptr,
                                                                                  M_src,
                                                                                  spkr_pattern,
                                                                                  (float*) info_pos_rcv.ptr,
                                                                                  (float*) info_orV_rcv.ptr,
                                                                                  M_rcv,
                                                                                  mic_pattern,
                                                                                  &nb_img[0], Fs);
    float* amp = echogram[0];
    float* tau = echogram[1];

    py::capsule free_when_done_amp(amp, [](void *f) {
        float *foo = reinterpret_cast<float *>(f);
        delete[] foo;
    });

    py::capsule free_when_done_tau(tau, [](void *f) {
        float *foo = reinterpret_cast<float *>(f);
        delete[] foo;
    });

    std::vector<int> shape = {M_src, M_rcv, nb_img[0], nb_img[1], nb_img[2]};
    std::vector<size_t> strides = {M_rcv * nb_img[0] * nb_img[1] * nb_img[2] * sizeof(float),
                                   nb_img[0] * nb_img[1] * nb_img[2] * sizeof(float),
                                   nb_img[1] * nb_img[2] * sizeof(float),
                                   nb_img[2] * sizeof(float),
                                   sizeof(float)};
    py::array_t<float> py_amp(shape, strides, amp, free_when_done_amp);
    py::array_t<float> py_tau(shape, strides, tau, free_when_done_tau);
    return py::make_tuple(py_amp, py_tau);

//    int nSamples = ceil(Tmax*Fs);
//    nSamples += nSamples%2; // nSamples must be even
//    std::vector<int> shape = {M_src, M_rcv, nSamples};
//    std::vector<size_t> strides = {M_rcv*nSamples*sizeof(float), nSamples*sizeof(float), sizeof(float)};
//    return py::array_t<float>(shape, strides, rir, free_when_done);

}

py::array gpuRIR_bind::render_echogram_bind(py::array_t<float, py::array::c_style> amp, // Amplitude of every echo
                                            py::array_t<float, py::array::c_style> tau, // Delay of every echo [samples]
                                            py::array_t<float, py::array::c_style> dp_tau, // Delay of the direct path [samples]
                                            float Tdiff, // Time when the ISM is replaced by a diffuse reverberation model [s]
                                            float Tmax, // RIRs length [s]
                                            float T60, // Reverberation time [s, only used for the diffuse reverberation model]
                                            float Fs // Sampling frequency [Hz]
                                            )
{
    py::buffer_info info_amp = amp.request();
    py::buffer_info info_tau = tau.request();
    py::buffer_info info_dp_tau = dp_tau.request();

    // Check input sizes
    assert(amp.ndim == 3);
    assert(tau.ndim == 3);
    assert(info_tau.shape[0] == info_amp.shape[0]);
    assert(info_tau.shape[1] == info_amp.shape[1]);
    assert(info_tau.shape[2] == info_amp.shape[2]);
    assert(info_dp_tau.shape[0] == info_amp.shape[0]);
    assert(info_dp_tau.shape[1] == info_amp.shape[1]);
    int M_src = info_amp.shape[0];
    int M_rcv = info_amp.shape[1];
    int N = info_amp.shape[2];

    float* rir = gpuRIR_cuda_simulator.cuda_render_echogram((float*) info_amp.ptr,
                                                            (float*) info_tau.ptr,
                                                            (float*) info_dp_tau.ptr,
                                                            M_src, M_rcv, N,
                                                            Tdiff, Tmax, T60, Fs);

    py::capsule free_when_done(rir, [](void *f) {
        float *foo = reinterpret_cast<float *>(f);
        delete[] foo;
    });

    int nSamples = ceil(Tmax*Fs);
    nSamples += nSamples%2; // nSamples must be even
    std::vector<int> shape = {M_src, M_rcv, nSamples};
    std::vector<size_t> strides = {M_rcv*nSamples*sizeof(float), nSamples*sizeof(float), sizeof(float)};
    return py::array_t<float>(shape, strides, rir, free_when_done);
}




py::array gpuRIR_bind::gpu_conv(py::array_t<float, py::array::c_style> source_segments, // Source signal segment for each trajectory point
                                py::array_t<float, py::array::c_style> RIR // 3D array with the RIR from each point of the trajectory to each receiver
                               )
{
    py::buffer_info info_source_segments = source_segments.request();
    py::buffer_info info_RIR = RIR.request();

    // Check input sizes
    assert(source_segments.ndim == 2);
    assert(RIR.ndim == 3);
    assert(info_source_segments.shape[0] == info_RIR.shape[0]);
    int M_src = info_source_segments.shape[0];
    int segment_len = info_source_segments.shape[1];
    int M_rcv = info_RIR.shape[1];
    int RIR_len = info_RIR.shape[2];

    float* convolution = gpuRIR_cuda_simulator.cuda_convolutions((float*)info_source_segments.ptr, M_src, segment_len,
                                                                    (float*)info_RIR.ptr, M_rcv, RIR_len);

    py::capsule free_when_done(convolution, [](void *f) {
        float *foo = reinterpret_cast<float *>(f);
        delete[] foo;
    });

    int nSamples = segment_len+RIR_len-1;
    std::vector<int> shape = {M_src, M_rcv, nSamples};
    std::vector<size_t> strides = {M_rcv*nSamples*sizeof(float), nSamples*sizeof(float), sizeof(float)};
    return py::array_t<float>(shape, strides, convolution, free_when_done);

}

bool gpuRIR_bind::activate_mixed_precision_bind(bool activate) {
    return gpuRIR_cuda_simulator.activate_mixed_precision(activate);
}

bool gpuRIR_bind::activate_lut_bind(bool activate) {
    return gpuRIR_cuda_simulator.activate_lut(activate);
}


PYBIND11_MODULE(gpuRIR_bind,m)
{
  m.doc() = "Room Impulse Response (RIR) simulation through Image Source Method (ISM) with GPU acceleration.";
  
  py::class_<gpuRIR_bind>(m, "gpuRIR_bind")
        .def(py::init<bool &>(), py::arg("mixed_precision")=false)
        .def("compute_echogram_bind", &gpuRIR_bind::compute_echogram_bind, "Echogram simulation", py::arg("room_sz"), py::arg("beta"), py::arg("pos_src"),
             py::arg("pos_rcv"), py::arg("orV_src"), py::arg("orV_rcv"), py::arg("spkr_pattern"), py::arg("mic_pattern"), py::arg("nb_img"), py::arg("Tdiff"), py::arg("Tmax"),
             py::arg("Fs"), py::arg("c")=343.0f )
        .def("render_echogram_bind", &gpuRIR_bind::render_echogram_bind, "Echogram rendering", py::arg("amp"), py::arg("tau"), py::arg("dp_tau"),
             py::arg("Tdiff"), py::arg("Tmax"), py::arg("T60"), py::arg("Fs") )
        .def("gpu_conv", &gpuRIR_bind::gpu_conv, "Batched convolution using FFTs in GPU", py::arg("source_segments"), py::arg("RIR"))
        .def("activate_mixed_precision_bind", &gpuRIR_bind::activate_mixed_precision_bind, "Activate the mixed precision mode, only for Pascal GPU architecture or superior",
             py::arg("activate"))
        .def("activate_lut_bind", &gpuRIR_bind::activate_lut_bind, "Activate the lookup table", py::arg("activate"));
}
