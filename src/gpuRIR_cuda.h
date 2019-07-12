

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

struct cuRandGeneratorWrapper_t;

class gpuRIR_cuda {
	
	public:
		
		static scalar_t* cuda_simulateRIR(scalar_t[3], scalar_t[6], scalar_t*, int, scalar_t*, scalar_t*, micPattern, int, int[3], scalar_t, scalar_t, scalar_t, scalar_t);
		static scalar_t* cuda_convolutions(scalar_t*, int, int,scalar_t*, int, int);
		static void cuda_warmup();
		
	private:
	
		// Image Source Method
		static const int nThreadsISM_x = 4;
		static const int nThreadsISM_y = 4;
		static const int nThreadsISM_z = 4;

		// Time vector generation
		static const int nThreadsTime = 128;

		// RIR computation
		static const int initialReductionMin = 512;
		static const int nThreadsGen_t = 32;
		static const int nThreadsGen_m = 4;
		static const int nThreadsGen_n = 1; // Don't change it
		static const int nThreadsRed = 128;

		// Power envelope prediction
		static const int nThreadsEnvPred_x = 4;
		static const int nThreadsEnvPred_y = 4;
		static const int nThreadsEnvPred_z = 1; // Don't change it

		// Generate diffuse reverberation
		static const int nThreadsDiff_t = 16;
		static const int nThreadsDiff_src = 4;
		static const int nThreadsDiff_rcv = 2;

		// RIR filtering onvolution
		static const int nThreadsConv_x = 256;
		static const int nThreadsConv_y = 1;
		static const int nThreadsConv_z = 1;
		
		// cuRAND generator
		static cuRandGeneratorWrapper_t cuRandGenWrap; // I'm not able to compile if I include the cuda headers here... so I have to hide the cuRAND generator
		
		// Auxiliar host functions
		static scalar_t* cuda_rirGenerator(scalar_t*, scalar_t*, scalar_t*, int, int, int, scalar_t);
		static int PadData(scalar_t*, scalar_t**, int, scalar_t*, scalar_t**, int, int, int);
};