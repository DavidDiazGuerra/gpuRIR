

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
		gpuRIR_cuda(bool, bool);

		scalar_t* cuda_simulateRIR(scalar_t[3], scalar_t[6], scalar_t*, int, scalar_t*, scalar_t*, micPattern, int, int[3], scalar_t, scalar_t, scalar_t, scalar_t);
		scalar_t* cuda_convolutions(scalar_t*, int, int,scalar_t*, int, int);
		bool activate_mixed_precision(bool);
		bool activate_lut(bool);
		
	private:
		// cuRAND generator
		static cuRandGeneratorWrapper_t cuRandGenWrap; // I'm not able to compile if I include the cuda headers here... so I have to hide the cuRAND generator
		
		// Mixed precision flag
		bool mixed_precision;
		
		// Lookup table flag
		bool lookup_table;

		// Auxiliar host functions
		void cuda_rirGenerator(scalar_t*, scalar_t*, scalar_t*, int, int, int, scalar_t);
		int PadData(scalar_t*, scalar_t**, int, scalar_t*, scalar_t**, int, int, int);
};