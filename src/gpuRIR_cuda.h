

typedef float scalar_t;

// Accepted polar patterns for the receivers:
typedef int micPattern;
#define DIR_OMNI 0
#define DIR_HOMNI 1
#define DIR_CARD 2
#define DIR_HYPCARD 3
#define DIR_SUBCARD 4
#define DIR_BIDIR 5

#define PI 3.141592654f

scalar_t* cuda_simulateRIR(scalar_t room_sz[3], scalar_t beta[6], scalar_t* h_pos_src, int M_src, scalar_t* h_pos_rcv, scalar_t* h_orV_rcv, micPattern mic_pattern, int M_rcv, int nb_img[3], scalar_t Tdiff, scalar_t Tmax, scalar_t Fs, scalar_t c);
