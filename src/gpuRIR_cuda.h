
typedef float scalar_t;
//typedef float2 Complex;

// Accepted polar patterns for the receivers:
typedef int micPattern;
#define DIR_OMNI 0
#define DIR_HOMNI 1
#define DIR_CARD 2
#define DIR_HYPCARD 3
#define DIR_SUBCARD 4
#define DIR_BIDIR 5

#define PI 3.141592654f

scalar_t* cuda_simulateRIR(scalar_t[3], scalar_t[6], scalar_t*, int, scalar_t*, scalar_t*, micPattern, int, int[3], scalar_t, scalar_t, scalar_t, scalar_t);
scalar_t* cuda_convolutions(scalar_t*, int, int,scalar_t*, int, int);