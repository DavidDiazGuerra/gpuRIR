
typedef float scalar_t;

scalar_t* cuda_simulateRIR(scalar_t room_sz[3], scalar_t beta[6], scalar_t* h_pos_src, int M_src, scalar_t* h_pos_rcv, int M_rcv, int nb_img[3], scalar_t Tdiff, scalar_t Tmax, scalar_t Fs=16000.0, scalar_t c=343.0);
