
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

		float* cuda_simulateRIR(float[3], float[6], float*, int, float*, float*, micPattern, int, int[3], float, float, float, float);
		float* cuda_convolutions(float*, int, int, float*, int, int);
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
		void cuda_rirGenerator(float*, float*, float*, int, int, int, float);
		int PadData(float*, float**, int, float*, float**, int, int, int);
};