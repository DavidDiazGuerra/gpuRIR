
#include <iostream>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuda_fp16.h>

#include <vector>
#include "gpuRIR_cuda.h"

#if CUDART_VERSION < 9000
#define __h2div h2div
#endif

// Image Source Method
static const int nThreadsISM_x = 4;
static const int nThreadsISM_y = 4;
static const int nThreadsISM_z = 4;

// RIR computation
static const int initialReductionMin = 512;
static const int lut_oversamp = 16; 
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

#if __CUDA_ARCH__ >= 530
#define h2zeros __float2half2_rn(0.0)
#define h2ones __float2half2_rn(1.0)
#define h2pi __float2half2_rn(PI)
#endif

// To hide the cuRAND generator in the header and don't need to include the cuda headers there
struct cuRandGeneratorWrapper_t
{
   curandGenerator_t gen;
};
cuRandGeneratorWrapper_t gpuRIR_cuda::cuRandGenWrap;

// CUDA architecture in format xy0
int cuda_arch;


/***************************/
/* Auxiliar host functions */
/***************************/

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(curandStatus_t code, const char *file, int line, bool abort=true) {
   if (code != CURAND_STATUS_SUCCESS) 
   {
      fprintf(stderr,"cuRAND: %d %s %d\n", code, file, line);
      if (abort) exit(code);
   }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cufftResult_t code, const char *file, int line, bool abort=true) {
   if (code != CUFFT_SUCCESS) 
   {
      fprintf(stderr,"cuFFT error: %d %s %d\n", code, file, line);
      if (abort) exit(code);
   }
}

inline unsigned int pow2roundup (unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}

/*****************************/
/* Auxiliar device functions */
/*****************************/

__device__ __forceinline__ float hanning_window(float t, float Tw) {
	return 0.5f * (1.0f + __cosf(2.0f*PI*t/Tw));
}

__device__ __forceinline__ float sinc(float x) {
	return (x==0)? 1 : sinf(x)/x;
}

__device__ __forceinline__ float image_sample(float amp, float tau, float t, float Tw) {
	float t_tau = t - tau;
	return (abs(t_tau)<Tw/2)? hanning_window(t_tau, Tw) * amp * sinc( (t_tau) * PI ) : 0.0f;
}

__device__ __forceinline__ float SabineT60( float room_sz_x, float room_sz_y, float room_sz_z,
							float beta_x1, float beta_x2, float beta_y1, float beta_y2, float beta_z1, float beta_z2 ) {
	float Sa = ((1.0f-beta_x1*beta_x1) + (1.0f-beta_x2*beta_x2)) * room_sz_y * room_sz_z +
				  ((1.0f-beta_y1*beta_y1) + (1.0f-beta_y2*beta_y2)) * room_sz_x * room_sz_z +
				  ((1.0f-beta_z1*beta_z1) + (1.0f-beta_z2*beta_z2)) * room_sz_x * room_sz_y;
	float V = room_sz_x * room_sz_y * room_sz_z;
	return 0.161f * V / Sa;
}

__device__ __forceinline__ float directivity(float doaVec[3], float orVec[3], polarPattern pattern) {
	if (pattern == DIR_OMNI) return 1.0f;
	
	float cosTheta = doaVec[0]*orVec[0] + doaVec[1]*orVec[1] + doaVec[2]*orVec[2];
	cosTheta /= sqrtf(doaVec[0]*doaVec[0] + doaVec[1]*doaVec[1] + doaVec[2]*doaVec[2]);
	cosTheta /= sqrtf(orVec[0]*orVec[0] + orVec[1]*orVec[1] + orVec[2]*orVec[2]);
	
	switch(pattern) {
		case DIR_HOMNI:		return (cosTheta>0.0f)? 1.0f : 0.0f;
		case DIR_CARD: 		return 0.5f  +  0.5f*cosTheta;
		case DIR_HYPCARD:	return 0.25f + 0.75f*cosTheta;
		case DIR_SUBCARD: 	return 0.75f + 0.25f*cosTheta;
		case DIR_BIDIR: 	return cosTheta;
		default: printf("Invalid polar pattern.\n"); return 0.0f;
	}
}

// cufftComplex scale
__device__ __forceinline__ cufftComplex ComplexScale(cufftComplex a, float s) {
    cufftComplex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// cufftComplex multiplication
__device__ __forceinline__ cufftComplex ComplexMul(cufftComplex a, cufftComplex b) {
    cufftComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

/*********************************************/
/* Mixed precision auxiliar device functions */
/*********************************************/

#if __CUDA_ARCH__ >= 530

__device__ __forceinline__ half2 h2abs(half2 x) {
	uint32_t i = *reinterpret_cast<uint32_t*>(&x) & 0x7FFF7FFF;
	return *reinterpret_cast<half2*>( &i );
}

__device__ __forceinline__ half2 my_h2sinpi(half2 x) {
	// Argument reduction to [-0.5, 0.5]
	half2 i = h2rint(x);
	half2 r = __hsub2(x, i);
	
	// sin(pi*x) polinomial approximation for x in [-0.5,0.5]
	half2 r2 = __hmul2(r, r);
	half2 s = __float2half2_rn(+2.31786431325108f);
	s = __hfma2(r2, s, __float2half2_rn(-5.14167814230801f));
	s = __hfma2(r2, s, __float2half2_rn(+3.14087446786993f));
	s = __hmul2(s, r);
	
	half2 i_2 = __hmul2(i, __float2half2_rn(0.5f));
	half2 sgn = __hfma2(__float2half2_rn(-2.0f), 
						__hne2(__hsub2(i_2, h2rint(i_2)), h2zeros), 
						h2ones); // 1 if i is even, else -1: -2 * ((i/2-round(i/2))!=0) + 1
	s = __hmul2(s, sgn);
	
	return s;
}

__device__ __forceinline__ half2 my_h2cospi(half2 x) {
	// It is always on [-0.5, 0.5], so we do not need argument reduction
	
	// cos(pi*x) polinomial approximation for x in [-0.5,0.5]
	half2 x2 = __hmul2(x, x);
	half2 c = __float2half2_rn(-1.229339658587166f);
	c = __hfma2(x2, c, __float2half2_rn(+4.043619929856572f));
	c = __hfma2(x2, c, __float2half2_rn(-4.934120365987677f));
	c = __hfma2(x2, c, __float2half2_rn(+0.999995282317910f));
	
	return c;
}

__device__ __forceinline__ half2 hanning_window_mp(half2 t, half2 Tw_inv) {	
	half2 c = my_h2cospi(__hmul2(Tw_inv, t));
	return __hmul2(c, c);
}

__device__ __forceinline__ half2 my_h2sinc(half2 x) {
	x = __hfma2(__heq2(x, h2zeros), __float2half2_rn(1e-7f), x);
	return __h2div(my_h2sinpi(x), __hmul2(h2pi, x));
	
}

__device__ __forceinline__ half2 image_sample_mp(half2 amp, float tau, float t1, float t2, float Tw_2, half2 Tw_inv) {
	float t1_tau = t1-tau;
	float t2_tau = t2-tau;
	half2 t_tau = __floats2half2_rn(t1_tau, t2_tau);
	if (abs(t1_tau)<Tw_2 || abs(t2_tau)<Tw_2) { // __hble2() is terribly slow
		return __hmul2(hanning_window_mp(t_tau, Tw_inv), __hmul2(amp, my_h2sinc( t_tau )));
	} else return h2zeros;
}

#endif

/******************************************/
/* Lookup table auxiliar device functions */
/******************************************/

__device__ __forceinline__ float image_sample_lut(float amp, float tau, float t, int Tw_2, cudaTextureObject_t sinc_lut, float lut_center) {
	float t_tau = t - tau;
	return (abs(t_tau)<Tw_2)? amp * tex1D<float>(sinc_lut, __fmaf_rz(t_tau,lut_oversamp,lut_center)) : 0.0f;
}

/***********/
/* KERNELS */
/***********/

__global__ void calcAmpTau_kernel(float* g_amp /*[M_src]M_rcv][nb_img_x][nb_img_y][nb_img_z]*/, 
								  float* g_tau /*[M_src]M_rcv][nb_img_x][nb_img_y][nb_img_z]*/, 
								  float* g_tau_dp /*[M_src]M_rcv]*/,
								  float* g_pos_src/*[M_src][3]*/, float* g_pos_rcv/*[M_rcv][3]*/, float* g_orV_src/*[M_src][3]*/, float* g_orV_rcv/*[M_rcv][3]*/,
								  polarPattern spkr_pattern, polarPattern mic_pattern, float room_sz_x, float room_sz_y, float room_sz_z,
								  float beta_x1, float beta_x2, float beta_y1, float beta_y2, float beta_z1, float beta_z2, 
								  int nb_img_x, int nb_img_y, int nb_img_z,
								  int M_src, int M_rcv, float c, float Fs) {
	
	extern __shared__ float sdata[];
	
	int n[3];
	n[0] = blockIdx.x * blockDim.x + threadIdx.x;
	n[1] = blockIdx.y * blockDim.y + threadIdx.y;
	n[2] = blockIdx.z * blockDim.z + threadIdx.z;
	
	int N[3];
	N[0] = nb_img_x;
	N[1] = nb_img_y;
	N[2] = nb_img_z;
	
	float room_sz[3];
	room_sz[0] = room_sz_x;
	room_sz[1] = room_sz_y;
	room_sz[2] = room_sz_z;
	
	float beta[6];
	beta[0] = - beta_x1;
	beta[1] = - beta_x2;
	beta[2] = - beta_y1;
	beta[3] = - beta_y2;
	beta[4] = - beta_z1;
	beta[5] = - beta_z2;
	
	int prodN = N[0]*N[1]*N[2];
	int n_idx = n[0]*N[1]*N[2] + n[1]*N[2] + n[2];
	
	// Copy g_pos_src to shared memory
	float* sh_pos_src = (float*) sdata;
	if (threadIdx.y==0 && threadIdx.z==0)  {
		for (int m=threadIdx.x; m<M_src; m+=blockDim.x) {
			sh_pos_src[m*3  ] = g_pos_src[m*3  ];
			sh_pos_src[m*3+1] = g_pos_src[m*3+1];
			sh_pos_src[m*3+2] = g_pos_src[m*3+2];
		}
	}
	
	// Copy g_pos_rcv to shared memory
	float* sh_pos_rcv = &sh_pos_src[M_src*3];
	if (threadIdx.x==0 && threadIdx.z==0)  {
		for (int m=threadIdx.y; m<M_rcv; m+=blockDim.y) {
			sh_pos_rcv[m*3  ] = g_pos_rcv[m*3  ];
			sh_pos_rcv[m*3+1] = g_pos_rcv[m*3+1];
			sh_pos_rcv[m*3+2] = g_pos_rcv[m*3+2];
		}
	}
	
	// Copy g_orV_rcv to shared memory
	float* sh_orV_rcv = &sh_pos_rcv[M_rcv*3];
	if (threadIdx.x==0 && threadIdx.y==0)  {
		for (int m=threadIdx.z; m<M_rcv; m+=blockDim.z) {
			sh_orV_rcv[m*3  ] = g_orV_rcv[m*3  ];
			sh_orV_rcv[m*3+1] = g_orV_rcv[m*3+1];
			sh_orV_rcv[m*3+2] = g_orV_rcv[m*3+2];
		}
	}

	// Copy g_orV_src to shared memory
	float* sh_orV_src = &sh_orV_rcv[M_rcv*3];
	if (threadIdx.x==0 && threadIdx.y==0)  {
		for (int m=threadIdx.z; m<M_src; m+=blockDim.z) {
			sh_orV_src[m*3  ] = g_orV_src[m*3  ];
			sh_orV_src[m*3+1] = g_orV_src[m*3+1];
			sh_orV_src[m*3+2] = g_orV_src[m*3+2];
		}
	}
	
	// Wait until the copies are completed
	__syncthreads();
	
	if (n[0]<N[0] & n[1]<N[1] & n[2]<N[2]) {
		
		// Common factors for each src and rcv
		float rflx_att = 1;
		float clust_pos[3];
		int clust_idx[3];
		int rflx_idx[3];
		bool direct_path = true;
		for (int d=0; d<3; d++) {
			clust_idx[d] = __float2int_ru((n[d] - N[d]/2) / 2.0f); 
			clust_pos[d] = clust_idx[d] * 2*room_sz[d];
			rflx_idx[d] = abs((n[d] - N[d]/2) % 2); // checking which dimensions are reflected: 1 means reflected in dimension d
			rflx_att *= powf(beta[d*2], abs(clust_idx[d]-rflx_idx[d])) * powf(beta[d*2+1], abs(clust_idx[d]));
			direct_path *= (clust_idx[d]==0)&&(rflx_idx[d]==0);
		}
			
		// Individual factors for each src and rcv
		for (int m_src=0; m_src<M_src; m_src++) {
			for (int m_rcv=0; m_rcv<M_rcv; m_rcv++) {
				float vec[3]; // Vector going from rcv to img src
				float im_src_to_rcv[3]; // for speaker directivity: Vector going from img src to rcv
				float orV_src[3]; // temporary orV_src to prevent overwriting sh_orV_src
				float dist = 0;
				for (int d=0; d<3; d++) {
					// computing the vector going from the rcv to img src
					vec[d] = clust_pos[d] + (1-2*rflx_idx[d]) * sh_pos_src[m_src*3+d] - sh_pos_rcv[m_rcv*3+d]; 
					dist += vec[d] * vec[d]; 
					
					// for speaker directivity
					orV_src[d] = (1-2*rflx_idx[d]) * sh_orV_src[m_src*3+d]; // If rflx_id = 1 => -1 * orV_src. If rflx_id = 0 => 1 * orV_src
					im_src_to_rcv[d] = -vec[d]; // change vector direction (mirror through origin)
				}
				dist = sqrtf(dist);
				float amp = rflx_att / (4*PI*dist);
				amp *= directivity(vec, &sh_orV_rcv[m_rcv], mic_pattern); // apply microphone directivity dampening
				amp *= directivity(im_src_to_rcv, orV_src, spkr_pattern); // apply speaker directivity dampening

				g_amp[m_src*M_rcv*prodN + m_rcv*prodN + n_idx] = amp;
				g_tau[m_src*M_rcv*prodN + m_rcv*prodN + n_idx] = dist / c * Fs;

				if (direct_path){
					g_tau_dp[m_src*M_rcv + m_rcv] = dist / c * Fs;
					// printf("%f,",amp); // For creating polar pattern diagrams
				} 
			}
		}
	}
}

__global__ void generateRIR_kernel(float* initialRIR, float* amp, float* tau, int T, int M, int N, int iniRIR_N, int ini_red, float Tw) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int m = blockIdx.y * blockDim.y + threadIdx.y;
	int n_ini = blockIdx.z * ini_red;
	int n_max = fminf(n_ini + ini_red, N);
	
	if (m<M && t<T) {
		float loc_sum = 0;
		for (int n=n_ini; n<n_max; n++) {
			loc_sum += image_sample(amp[m*N+n], tau[m*N+n], t, Tw);
		}
		initialRIR[m*T*iniRIR_N + t*iniRIR_N + blockIdx.z] = loc_sum;
	}
}

__global__ void reduceRIR_kernel(float* initialRIR, float* intermediateRIR, int M, int T, int N, int intRIR_N) {
	extern __shared__ float sdata[];
	
	int tid = threadIdx.x;
	int n = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	int t = blockIdx.y;
	int m = blockIdx.z;
	
	if (n+blockDim.x < N) sdata[tid] = initialRIR[m*T*N + t*N + n] + initialRIR[m*T*N + t*N + n+blockDim.x];
	else if (n<N) sdata[tid] = initialRIR[m*T*N + t*N + n];
	else sdata[tid] = 0;
	__syncthreads();
	
	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) sdata[tid] += sdata[tid+s];
		__syncthreads();
	}
	
	if (tid==0) {
		intermediateRIR[m*T*intRIR_N + t*intRIR_N + blockIdx.x] = sdata[0];
	}
}

__global__ void envPred_kernel(float* A /*[M_src]M_rcv]*/, float* alpha /*[M_src]M_rcv]*/, 
						float* RIRs_early /*[M_src][M_rcv][nSamples]*/, float* tau_dp, /*[M_src]M_rcv]*/
						int M_src, int M_rcv, int nSamples, float Fs,
						float room_sz_x, float room_sz_y, float room_sz_z,
						float beta_x1, float beta_x2, float beta_y1, float beta_y2, float beta_z1, float beta_z2) {
		
	float w_sz = 10e-3f * Fs; // Maximum window size (samples) to compute the final power of the early RIRs_early
	
	int m_src = blockIdx.x * blockDim.x + threadIdx.x;
	int m_rcv = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (m_src<M_src && m_rcv<M_rcv) {
		int w_start = __float2int_ru( max(nSamples-w_sz, tau_dp[m_src*M_rcv+m_rcv]));
		float w_center = (w_start + (nSamples-w_start)/2.0);
		
		float finalPower = 0.0f;
		for (int t=w_start; t<nSamples; t++) {
			float aux = RIRs_early[m_src*M_rcv*nSamples + m_rcv*nSamples + t];
			finalPower += aux*aux;
		}
		finalPower /= nSamples-w_start;
		
		float T60 = SabineT60(room_sz_x, room_sz_y, room_sz_z, beta_x1, beta_x2, beta_y1, beta_y2, beta_z1, beta_z2);
		float loc_alpha = -13.8155f / (T60 * Fs); //-13.8155 == log(10^(-6))
		
		A[m_src*M_rcv + m_rcv] = finalPower / expf(loc_alpha*(w_center-tau_dp[m_src*M_rcv+m_rcv]));
		alpha[m_src*M_rcv + m_rcv] = loc_alpha;
	}
}

__global__ void diffRev_kernel(float* rir, float* A, float* alpha, float* tau_dp,
							   int M_src, int M_rcv, int nSamplesISM, int nSamplesDiff) {
	
	int sample = blockIdx.x * blockDim.x + threadIdx.x;
	int m_src  = blockIdx.y * blockDim.y + threadIdx.y;
	int m_rcv  = blockIdx.z * blockDim.z + threadIdx.z;
	
	if (sample<nSamplesDiff && m_src<M_src && m_rcv<M_rcv) {
		// Get logistic distribution from uniform distribution
		float uniform = rir[m_src*M_rcv*nSamplesDiff + m_rcv*nSamplesDiff + sample];
		float logistic = 0.551329f * logf(uniform/(1.0f - uniform + 1e-6)); // 0.551329 == sqrt(3)/pi
		
		// Apply power envelope
		float pow_env = A[m_src*M_rcv+m_rcv] * expf(alpha[m_src*M_rcv+m_rcv] * (nSamplesISM+sample-tau_dp[m_src*M_rcv+m_rcv]));
		rir[m_src*M_rcv*nSamplesDiff + m_rcv*nSamplesDiff + sample] = sqrt(pow_env) * logistic;
	}
}

__global__ void complexPointwiseMulAndScale(cufftComplex *signal_segments, cufftComplex *RIRs, int segment_size, int M_rcv, int M_src, float scale) {
    int numThreads_x = blockDim.x * gridDim.x;
    int numThreads_y = blockDim.y * gridDim.y;
    int numThreads_z = blockDim.z * gridDim.z;
	
    int threadID_x = blockIdx.x * blockDim.x + threadIdx.x;
    int threadID_y = blockIdx.y * blockDim.y + threadIdx.y;
    int threadID_z = blockIdx.z * blockDim.z + threadIdx.z;

	for (int m = threadID_z; m < M_src; m += numThreads_z) {
		for (int n = threadID_y; n < M_rcv; n += numThreads_y) {
			for (int i = threadID_x; i < segment_size; i += numThreads_x) {
				RIRs[m*M_rcv*segment_size + n*segment_size + i] = 
					ComplexScale(ComplexMul(RIRs[m*M_rcv*segment_size + n*segment_size + i], 
											signal_segments[m*segment_size + i]), 
								 scale);
			}
		}
	}
}

/***************************/
/* Mixed precision KERNELS */
/***************************/

#if CUDART_VERSION < 9020
__global__ void generateRIR_mp_kernel(half2* initialRIR, float* amp, float* tau, int T, int M, int N, int iniRIR_N, int ini_red, float Fs, float Tw_2, float Tw_inv) {
	half2 h2Tw_inv = __float2half2_rn(Tw_inv);
#else 
__global__ void generateRIR_mp_kernel(half2* initialRIR, float* amp, float* tau, int T, int M, int N, int iniRIR_N, int ini_red, float Fs, float Tw_2, half2 h2Tw_inv) {
#endif
	#if __CUDA_ARCH__ >= 530
		int t = blockIdx.x * blockDim.x + threadIdx.x;
		int m = blockIdx.y * blockDim.y + threadIdx.y;
		int n_ini = blockIdx.z * ini_red;
		int n_max = fminf(n_ini + ini_red, N);
		
		if (m<M && t<T) {
			half2 loc_sum = h2zeros;
			float loc_tim_1 = 2*t;
			float loc_tim_2 = 2*t+1;
			for (int n=n_ini; n<n_max; n++) {
				half2 amp_mp = __float2half2_rn(amp[m*N+n]);
				loc_sum = __hadd2(loc_sum, image_sample_mp(amp_mp, tau[m*N+n], loc_tim_1, loc_tim_2, Tw_2, h2Tw_inv));
			}
			initialRIR[m*T*iniRIR_N + t*iniRIR_N + blockIdx.z] = loc_sum;
		}
	#else
		printf("Mixed precision requires Pascal GPU architecture or higher.\n");
	#endif
}

__global__ void reduceRIR_mp_kernel(half2* initialRIR, half2* intermediateRIR, int M, int T, int N, int intRIR_N) {
	extern __shared__ half2 sdata_mp[];
	#if __CUDA_ARCH__ >= 530
		int tid = threadIdx.x;
		int n = blockIdx.x*(blockDim.x*2) + threadIdx.x;
		int t = blockIdx.y;
		int m = blockIdx.z;

		if (n+blockDim.x < N) sdata_mp[tid] = __hadd2(initialRIR[m*T*N + t*N + n], initialRIR[m*T*N + t*N + n+blockDim.x]);
		else if (n<N) sdata_mp[tid] = initialRIR[m*T*N + t*N + n];
		else sdata_mp[tid] = h2zeros;
		__syncthreads();

		for (int s=blockDim.x/2; s>0; s>>=1) {
			if (tid < s) sdata_mp[tid] = __hadd2(sdata_mp[tid], sdata_mp[tid+s]);
			__syncthreads();
		}

		if (tid==0) {
			intermediateRIR[m*T*intRIR_N + t*intRIR_N + blockIdx.x] = sdata_mp[0];
		}
	#else
		printf("Mixed precision requires Pascal GPU architecture or higher.\n");
	#endif
}

__global__ void h2RIR_to_floatRIR_kernel(half2* h2RIR, float* floatRIR, int M, int T) {
	#if __CUDA_ARCH__ >= 530
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int m = blockIdx.y * blockDim.y + threadIdx.y;

	if (t<T && m<M) {
		floatRIR[m*2*T + 2*t  ] =  __low2float(h2RIR[m*T + t]);
		floatRIR[m*2*T + 2*t+1] = __high2float(h2RIR[m*T + t]);
	}
	#else
		printf("Mixed precision requires Pascal GPU architecture or higher.\n");
	#endif
}

/************************/
/* Lookup table KERNELS */
/************************/

__global__ void generateRIR_kernel_lut(float* initialRIR, float* amp, float* tau, int T, int M, int N, int iniRIR_N, int ini_red, int Tw_2, cudaTextureObject_t sinc_lut, float lut_center) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int m = blockIdx.y * blockDim.y + threadIdx.y;
	int n_ini = blockIdx.z * ini_red;
	int n_max = fminf(n_ini + ini_red, N);
	
	if (m<M && t<T) {
		float loc_sum = 0;
		for (int n=n_ini; n<n_max; n++) {
			loc_sum += image_sample_lut(amp[m*N+n], tau[m*N+n], t, Tw_2, sinc_lut, lut_center);
		}
		initialRIR[m*T*iniRIR_N + t*iniRIR_N + blockIdx.z] = loc_sum;
	}
}

/***************************/
/* Auxiliar host functions */
/***************************/

cudaTextureObject_t create_sinc_texture_lut(cudaArray **cuArrayLut, int Tw, int lut_len) {
	// Create lut in host memory
	int lut_center = lut_len / 2;
	float* sinc_lut_host = (float*)malloc(sizeof(float) * lut_len);
	for (int i=0; i<=lut_center; i++) {
		float x = (float)i / lut_oversamp;
		float sinc = (x==0.0f)? 1.0f : sin(PI*x) / (PI*x);
		float hann = 0.5f * (1.0f + cos(2.0f*PI*x/Tw));
		float y = hann * sinc;
		sinc_lut_host[lut_center+i] = y;
		sinc_lut_host[lut_center-i] = y;
	}
	
	// Copy the lut to a device cudaArray
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray(cuArrayLut, &channelDesc, lut_len);
    cudaMemcpyToArray(*cuArrayLut, 0, 0, sinc_lut_host, sizeof(float)*lut_len,
                      cudaMemcpyHostToDevice);
	
	// Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = *cuArrayLut;

	// Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeBorder;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
	
	// Create texture object
    cudaTextureObject_t texObjLut = 0;
    cudaCreateTextureObject(&texObjLut, &resDesc, &texDesc, NULL);
	
	return texObjLut;
}

void gpuRIR_cuda::cuda_rirGenerator(float* rir, float* amp, float* tau, int M, int N, int T, float Fs) {
	int initialReduction = initialReductionMin;
	while (M * T * ceil((float)N/initialReduction) > 1e9) initialReduction *= 2;
	
	int iniRIR_N = ceil((float)N/initialReduction);
	dim3 threadsPerBlockIni(nThreadsGen_t, nThreadsGen_m, nThreadsGen_n);
	dim3 numBlocksIni(ceil((float)T/threadsPerBlockIni.x), ceil((float)M/threadsPerBlockIni.y), iniRIR_N);
	
	float* initialRIR;
	gpuErrchk( cudaMalloc(&initialRIR, M*T*iniRIR_N*sizeof(float)) );
	
	int Tw = (int) round(8e-3f * Fs); // Window duration [samples]
	
	if (lookup_table) {
		int lut_len = Tw * lut_oversamp;
		lut_len += ((lut_len%2)? 0 : 1); // Must be odd
		cudaArray* cuArrayLut;
		cudaTextureObject_t sinc_lut = create_sinc_texture_lut(&cuArrayLut, Tw, lut_len);
		
		generateRIR_kernel_lut<<<numBlocksIni, threadsPerBlockIni>>>( initialRIR, amp, tau, T, M, N, iniRIR_N, initialReduction, Tw/2, sinc_lut, lut_len/2+0.5 );
		gpuErrchk( cudaDeviceSynchronize() );
		gpuErrchk( cudaPeekAtLastError() );
		
		cudaDestroyTextureObject(sinc_lut);
		cudaFreeArray(cuArrayLut);
	} else {
		generateRIR_kernel<<<numBlocksIni, threadsPerBlockIni>>>( initialRIR, amp, tau, T, M, N, iniRIR_N, initialReduction, Tw );
		gpuErrchk( cudaDeviceSynchronize() );
		gpuErrchk( cudaPeekAtLastError() );
	}
		
	
	dim3 threadsPerBlockRed(nThreadsRed, 1, 1);
	float* intermediateRIR;
	int intRIR_N;
	while (iniRIR_N > 2*nThreadsRed) {		
		intRIR_N = ceil((float)iniRIR_N / (2*nThreadsRed));
		gpuErrchk( cudaMalloc(&intermediateRIR, intRIR_N * T * M * sizeof(float)) );

		dim3 numBlocksRed(intRIR_N, T, M);
		reduceRIR_kernel<<<numBlocksRed, threadsPerBlockRed, nThreadsRed*sizeof(float)>>>(
			initialRIR, intermediateRIR, M, T, iniRIR_N, intRIR_N);
		gpuErrchk( cudaDeviceSynchronize() );
		gpuErrchk( cudaPeekAtLastError() );
		
		gpuErrchk( cudaFree(initialRIR) );
		initialRIR = intermediateRIR;		
		iniRIR_N = intRIR_N;
	}
	
	dim3 numBlocksEnd(1, T, M);
	reduceRIR_kernel<<<numBlocksEnd, threadsPerBlockRed, nThreadsRed*sizeof(float)>>>(
		initialRIR, rir, M, T, iniRIR_N, 1);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaFree(initialRIR) );
}

void cuda_rirGenerator_mp(float* rir, float* amp, float* tau, int M, int N, int T, float Fs) {
	if (cuda_arch >= 530) {
		int initialReduction = initialReductionMin;
		while (M * T/2 * ceil((float)N/initialReduction) > 1e9) initialReduction *= 2;

		int iniRIR_N = ceil((float)N/initialReduction);
		dim3 threadsPerBlockIni(nThreadsGen_t, nThreadsGen_m, nThreadsGen_n);
		dim3 numBlocksIni(ceil((float)T/2/threadsPerBlockIni.x), ceil((float)M/threadsPerBlockIni.y), iniRIR_N);

		half2* initialRIR;
		gpuErrchk( cudaMalloc(&initialRIR, M*(T/2)*iniRIR_N*sizeof(half2)) );

		float Tw_2 = 8e-3f * Fs / 2;
		#if CUDART_VERSION < 9020
			// For CUDA versions older than 9.2 it is nos possible to call from host code __float2half2_rn,
			// but doing it in the kernel is slower
			float Tw_inv = 1.0f / (8e-3f * Fs);
		#else 
			half2 Tw_inv = __float2half2_rn(1.0f / (8e-3f * Fs));
		#endif
		generateRIR_mp_kernel<<<numBlocksIni, threadsPerBlockIni>>>( initialRIR, amp, tau, T/2, M, N, iniRIR_N, initialReduction, Fs, Tw_2, Tw_inv );
		gpuErrchk( cudaDeviceSynchronize() );
		gpuErrchk( cudaPeekAtLastError() );

		dim3 threadsPerBlockRed(nThreadsRed, 1, 1);
		half2* intermediateRIR;
		int intRIR_N;
		while (iniRIR_N > 2*nThreadsRed) {
			intRIR_N = ceil((float)iniRIR_N / (2*nThreadsRed));
			gpuErrchk( cudaMalloc(&intermediateRIR, intRIR_N * T/2 * M * sizeof(half2)) );

			dim3 numBlocksRed(intRIR_N, T/2, M);
			reduceRIR_mp_kernel<<<numBlocksRed, threadsPerBlockRed, nThreadsRed*sizeof(half2)>>>(
				initialRIR, intermediateRIR, M, T/2, iniRIR_N, intRIR_N);
			gpuErrchk( cudaDeviceSynchronize() );
			gpuErrchk( cudaPeekAtLastError() );

			gpuErrchk( cudaFree(initialRIR) );
			initialRIR = intermediateRIR;
			iniRIR_N = intRIR_N;
		}

		gpuErrchk( cudaMalloc(&intermediateRIR, M * T/2 * sizeof(half2)) );
		dim3 numBlocksEnd(1, T/2, M);
		reduceRIR_mp_kernel<<<numBlocksEnd, threadsPerBlockRed, nThreadsRed*sizeof(half2)>>>(
			initialRIR, intermediateRIR, M, T/2, iniRIR_N, 1);
		gpuErrchk( cudaDeviceSynchronize() );
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaFree(initialRIR) );

		dim3 numBlocks(ceil((float)(T/2)/128), M, 1);
		dim3 threadsPerBlock(128, 1, 1);
		h2RIR_to_floatRIR_kernel<<<numBlocks, threadsPerBlock>>>(intermediateRIR, rir, M, T/2);

		gpuErrchk( cudaFree(intermediateRIR) );
	} else {
		printf("Mixed precision requires Pascal GPU architecture or higher.\n");
	}
}

int gpuRIR_cuda::PadData(float *signal, float **padded_signal, int segment_len,
						 float *RIR, float **padded_RIR, int M_src, int M_rcv, int RIR_len) {
				
    int N_fft = pow2roundup(segment_len + RIR_len - 1);

    // Pad signal
    float *new_data = (float *)malloc(sizeof(float) * M_src * (N_fft+2));
	for (int m=0; m<M_src; m++) {
		memcpy(new_data + m*(N_fft+2), signal + m*segment_len, segment_len*sizeof(float));
		memset(new_data + m*(N_fft+2) + segment_len, 0, ((N_fft+2)-segment_len)*sizeof(float));
	}
    *padded_signal = new_data;

    // Pad filter
    new_data = (float *)malloc(sizeof(float) * M_src * M_rcv * (N_fft+2));
	for (int m=0; m<M_src; m++) {
		for (int n=0; n<M_rcv; n++) {
			memcpy(new_data + m*M_rcv*(N_fft+2) + n*(N_fft+2), RIR + m*M_rcv*RIR_len + n*RIR_len, RIR_len*sizeof(float));
			memset(new_data + m*M_rcv*(N_fft+2) + n*(N_fft+2) + RIR_len, 0, ((N_fft+2)-RIR_len)*sizeof(float));
		}
	}
    *padded_RIR = new_data;

    return N_fft;
}

/***********************/
/* Principal functions */
/***********************/

float* gpuRIR_cuda::cuda_simulateRIR(float room_sz[3], float beta[6], float* h_pos_src, int M_src, 
									   float* h_pos_rcv, float* h_orV_src, float* h_orV_rcv, 
									   polarPattern spkr_pattern, polarPattern mic_pattern, int M_rcv, int nb_img[3],
									   float Tdiff, float Tmax, float Fs, float c) {	
	// function float* cuda_simulateRIR(float room_sz[3], float beta[6], float* h_pos_src, int M_src, 
	//								   float* h_pos_rcv, float* h_orV_src, float* h_orV_rcv, 
	//								   polarPattern spkr_pattern, polarPattern mic_pattern, int M_rcv, int nb_img[3],
	//								   float Tdiff, float Tmax, float Fs, float c);
	// Input parameters:
	// 	float room_sz[3]		: Size of the room [m]
	//	float beta[6] 		: Reflection coefficients [beta_x1 beta_x2 beta_y1 beta_y2 beta_z1 beta_z2]
	//	float* h_pos_src 	: M_src x 3 matrix with the positions of the sources [m]
	//	int M_src 				: Number of sources
	//	float* h_pos_rcv 	: M_rcv x 3 matrix with the positions of the receivers [m]
	//	float* h_orV_src 	: M_rcv x 3 matrix with vectors pointing in the same direction than the source
	//	float* h_orV_rcv 	: M_rcv x 3 matrix with vectors pointing in the same direction than the receivers
	//	polarPattern spkr_pattern 	: Polar pattern of the sources (see gpuRIR_cuda.h)
	//	polarPattern mic_pattern 	: Polar pattern of the receivers (see gpuRIR_cuda.h)
	//	int M_rcv 				: Number of receivers
	//	int nb_img[3] 			: Number of sources in each dimension
	//	float Tdiff			: Time when the ISM is replaced by a diffusse reverberation model [s]
	//	float Tmax 			: RIRs length [s]
	//	float Fs				: Sampling frequency [Hz]
	//	float c				: Speed of sound [m/s]
	
	// Copy host memory to GPU
	float *pos_src, *pos_rcv, *orV_src, *orV_rcv;
	gpuErrchk( cudaMalloc(&pos_src, M_src*3*sizeof(float)) );
	gpuErrchk( cudaMalloc(&pos_rcv, M_rcv*3*sizeof(float)) );
	gpuErrchk( cudaMalloc(&orV_src, M_src*3*sizeof(float)) );
	gpuErrchk( cudaMalloc(&orV_rcv, M_rcv*3*sizeof(float)) );
	gpuErrchk( cudaMemcpy(pos_src, h_pos_src, M_src*3*sizeof(float), cudaMemcpyHostToDevice ) );
	gpuErrchk( cudaMemcpy(pos_rcv, h_pos_rcv, M_rcv*3*sizeof(float), cudaMemcpyHostToDevice ) );
	gpuErrchk( cudaMemcpy(orV_src, h_orV_src, M_src*3*sizeof(float), cudaMemcpyHostToDevice ) );
	gpuErrchk( cudaMemcpy(orV_rcv, h_orV_rcv, M_rcv*3*sizeof(float), cudaMemcpyHostToDevice ) );
	
	
	// Use the ISM to calculate the amplitude and delay of each image
	dim3 threadsPerBlockISM(nThreadsISM_x, nThreadsISM_y, nThreadsISM_z);
	dim3 numBlocksISM(ceil((float)nb_img[0] / nThreadsISM_x), 
					  ceil((float)nb_img[1] / nThreadsISM_y), 
					  ceil((float)nb_img[2] / nThreadsISM_z));
	int shMemISM = (2*M_src + 2*M_rcv) * 3 * sizeof(float);
	// printf("shMemISM: %d\n", shMemISM);
	
	float* amp; // Amplitude with which the signals from each image source of each source arrive to each receiver
	gpuErrchk( cudaMalloc(&amp, M_src*M_rcv*nb_img[0]*nb_img[1]*nb_img[2]*sizeof(float)) );
	float* tau; // Delay with which the signals from each image source of each source arrive to each receiver
	gpuErrchk( cudaMalloc(&tau, M_src*M_rcv*nb_img[0]*nb_img[1]*nb_img[2]*sizeof(float)) );
	float* tau_dp; // Direct path delay
	gpuErrchk( cudaMalloc(&tau_dp, M_src*M_rcv*sizeof(float)) );
	
	calcAmpTau_kernel<<<numBlocksISM, threadsPerBlockISM, shMemISM>>> (
		amp, tau, tau_dp,
		pos_src, pos_rcv, orV_src, orV_rcv, spkr_pattern, mic_pattern,
		room_sz[0], room_sz[1], room_sz[2], 
		beta[0], beta[1], beta[2], beta[3], beta[4], beta[5], 
		nb_img[0], nb_img[1], nb_img[2],
		M_src, M_rcv, c, Fs
	);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );

	int nSamplesISM = ceil(Tdiff*Fs);
	nSamplesISM += nSamplesISM%2; // nSamplesISM must be even
	int nSamples = ceil(Tmax*Fs);
	nSamples += nSamples%2; // nSamples must be even
	int nSamplesDiff = nSamples - nSamplesISM;
	
	// Compute the RIRs as a sum of sincs
	int M = M_src * M_rcv;
	int N = nb_img[0] * nb_img[1] * nb_img[2];
	float* rirISM;
	gpuErrchk( cudaMalloc(&rirISM, M*nSamplesISM*sizeof(float)) );
	if (mixed_precision) {
		if (cuda_arch >= 530) {
			cuda_rirGenerator_mp(rirISM, amp, tau, M, N, nSamplesISM, Fs);
		} else {
			printf("The mixed precision requires Pascal GPU architecture or higher.\n");
		}
	} else {
		cuda_rirGenerator(rirISM, amp, tau, M, N, nSamplesISM, Fs);
	}
	
	// Compute the exponential power envelope parammeters of each RIR
	dim3 threadsPerBlockEnvPred(nThreadsEnvPred_x, nThreadsEnvPred_y, nThreadsEnvPred_z);
	dim3 numBlocksEnvPred(ceil((float)M_src / nThreadsEnvPred_x), 
						  ceil((float)M_rcv / nThreadsEnvPred_y), 1);
					  
	float* A; // pow_env = A * exp(alpha * (t-tau_dp))
	gpuErrchk( cudaMalloc(&A, M_src*M_rcv*sizeof(float)) );
	float* alpha;
	gpuErrchk( cudaMalloc(&alpha, M_src*M_rcv*sizeof(float)) );
	
	envPred_kernel<<<numBlocksEnvPred, threadsPerBlockEnvPred>>>(
			A, alpha, rirISM, tau_dp, M_src, M_rcv, nSamplesISM, Fs,
			room_sz[0], room_sz[1], room_sz[2], beta[0], beta[1], beta[2], beta[3], beta[4], beta[5]);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	
	// Generate diffuse reverberation
	float* rirDiff; 
	gpuErrchk( cudaMalloc(&rirDiff, M_src*M_rcv*nSamplesDiff*sizeof(float)) );
	
	if (nSamplesDiff != 0) {
		// Fill rirDiff with random numbers with uniform distribution
		gpuErrchk( curandGenerateUniform(cuRandGenWrap.gen, rirDiff, M_src*M_rcv*nSamplesDiff) );
		gpuErrchk( cudaDeviceSynchronize() );
		gpuErrchk( cudaPeekAtLastError() );
				
		dim3 threadsPerBlockDiff(nThreadsDiff_t, nThreadsDiff_src, nThreadsDiff_rcv);
		dim3 numBlocksDiff(ceil((float)nSamplesDiff / nThreadsDiff_t),
							  ceil((float)M_src / nThreadsDiff_src), 
							  ceil((float)M_rcv / nThreadsDiff_rcv));
		diffRev_kernel<<<numBlocksDiff, threadsPerBlockDiff>>>(
				rirDiff, A, alpha, tau_dp, M_src, M_rcv, nSamplesISM, nSamplesDiff);
		gpuErrchk( cudaDeviceSynchronize() );
		gpuErrchk( cudaPeekAtLastError() );
	}
	
	// Copy GPU memory to host
	int rirSizeISM = M_src * M_rcv * nSamplesISM * sizeof(float);
	int rirSizeDiff = M_src * M_rcv * nSamplesDiff * sizeof(float);
	float* h_rir = (float*) malloc(rirSizeISM+rirSizeDiff);
	
	cudaPitchedPtr h_rir_pitchedPtr = make_cudaPitchedPtr( (void*) h_rir, 
		(nSamplesISM+nSamplesDiff)*sizeof(float), nSamplesISM+nSamplesDiff, M_rcv );
	cudaPitchedPtr rirISM_pitchedPtr = make_cudaPitchedPtr( (void*) rirISM, 
		nSamplesISM*sizeof(float), nSamplesISM, M_rcv );
	cudaPitchedPtr rirDiff_pitchedPtr = make_cudaPitchedPtr( (void*) rirDiff, 
		nSamplesDiff*sizeof(float), nSamplesDiff, M_rcv );
	
	cudaMemcpy3DParms parmsISM = {0};
	parmsISM.srcPtr = rirISM_pitchedPtr;
	parmsISM.dstPtr = h_rir_pitchedPtr;
	parmsISM.extent = make_cudaExtent(nSamplesISM*sizeof(float), M_rcv, M_src);
	parmsISM.kind = cudaMemcpyDeviceToHost;
	gpuErrchk( cudaMemcpy3D(&parmsISM) );
	
	if (nSamplesDiff > 0) {
		cudaMemcpy3DParms parmsDiff = {0};
		parmsDiff.srcPtr = rirDiff_pitchedPtr;
		parmsDiff.dstPtr = h_rir_pitchedPtr;
		parmsDiff.dstPos = make_cudaPos(nSamplesISM*sizeof(float), 0, 0);
		parmsDiff.extent = make_cudaExtent(nSamplesDiff*sizeof(float), M_rcv, M_src);
		parmsDiff.kind = cudaMemcpyDeviceToHost;
		gpuErrchk( cudaMemcpy3D(&parmsDiff) );
	}

	// Free memory
	gpuErrchk( cudaFree(pos_src) );
	gpuErrchk( cudaFree(pos_rcv) );
	gpuErrchk( cudaFree(orV_rcv) );
	gpuErrchk( cudaFree(amp)	 );
	gpuErrchk( cudaFree(tau)	 );
	gpuErrchk( cudaFree(tau_dp)	 );
	gpuErrchk( cudaFree(rirISM)	 );
	gpuErrchk( cudaFree(A)		 );
	gpuErrchk( cudaFree(alpha)	 );
	gpuErrchk( cudaFree(rirDiff) );
	
	return h_rir;
}

float* gpuRIR_cuda::cuda_convolutions(float* source_segments, int M_src, int segment_len,
										float* RIR, int M_rcv, int RIR_len) {	
	// function float* cuda_filterRIR(float* source_segments, int M_src, int segments_len,
	//									 float* RIR, int M_rcv, int RIR_len);
	// Input parameters:
	// 	float* source_segments : Source signal segment for each trajectory point
	//	int M_src 				  : Number of trajectory points
	//	int segment_len 		  : Length of the segments [samples]
	//	float* RIR		 	  : 3D array with the RIR from each point of the trajectory to each receiver
	//	int M_rcv 				  : Number of receivers
	//	int RIR_len 			  : Length of the RIRs [samples]

	// Size of the FFT needed to avoid circular convolution effects
	int N_fft = pow2roundup(segment_len + RIR_len - 1);
	
	// Copy the signal segments with zero padding
    int mem_size_signal = sizeof(float) * M_src * (N_fft+2);
    cufftComplex *d_signal;
    gpuErrchk( cudaMalloc((void **)&d_signal, mem_size_signal) );
	gpuErrchk( cudaMemcpy2D((void *)d_signal, (N_fft+2)*sizeof(float), 
		(void *)source_segments, segment_len*sizeof(float),
		segment_len*sizeof(float), M_src, cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemset2D((void *)((float *)d_signal + segment_len), (N_fft+2)*sizeof(float),
		0, (N_fft+2-segment_len)*sizeof(float), M_src ) );
	
	// Copy the RIRs with zero padding
	cudaPitchedPtr h_RIR_pitchedPtr = make_cudaPitchedPtr( (void*) RIR, 
		RIR_len*sizeof(float), RIR_len, M_rcv );
    int mem_size_RIR = sizeof(float) * M_src * M_rcv * (N_fft+2);
	cufftComplex *d_RIR;
	gpuErrchk( cudaMalloc((void **)&d_RIR, mem_size_RIR) );
	cudaPitchedPtr d_RIR_pitchedPtr = make_cudaPitchedPtr( (void*) d_RIR, 
		(N_fft+2)*sizeof(float), (N_fft+2), M_rcv );
	cudaMemcpy3DParms parmsCopySignal = {0};
	parmsCopySignal.srcPtr = h_RIR_pitchedPtr;
	parmsCopySignal.dstPtr = d_RIR_pitchedPtr;
	parmsCopySignal.extent = make_cudaExtent(RIR_len*sizeof(float), M_rcv, M_src);
	parmsCopySignal.kind = cudaMemcpyHostToDevice;
	gpuErrchk( cudaMemcpy3D(&parmsCopySignal) );
	gpuErrchk( cudaMemset2D((void *)((float *)d_RIR + RIR_len), (N_fft+2)*sizeof(float),
		0, (N_fft+2-RIR_len)*sizeof(float), M_rcv*M_src ) );
	
	// CUFFT plans
    cufftHandle plan_signal, plan_RIR, plan_RIR_inv;
    gpuErrchk( cufftPlan1d(&plan_signal,  N_fft, CUFFT_R2C, M_src) );
    gpuErrchk( cufftPlan1d(&plan_RIR,     N_fft, CUFFT_R2C, M_src * M_rcv) );
    gpuErrchk( cufftPlan1d(&plan_RIR_inv, N_fft, CUFFT_C2R, M_src * M_rcv) );
	
	// Transform signal and RIR
    gpuErrchk( cufftExecR2C(plan_signal, (cufftReal *)d_signal, (cufftComplex *)d_signal) );
    gpuErrchk( cufftExecR2C(plan_RIR,    (cufftReal *)d_RIR,    (cufftComplex *)d_RIR   ) );
	
	// Multiply the coefficients together and normalize the result
	dim3 threadsPerBlock(nThreadsConv_x, nThreadsConv_y, nThreadsConv_z);
	int numBlocks_x = (int) ceil((float)(N_fft/2+1)/nThreadsConv_x);
	int numBlocks_y = (int) ceil((float)M_rcv/nThreadsConv_y);
	int numBlocks_z = (int) ceil((float)M_src/nThreadsConv_z);
	dim3 numBlocks(numBlocks_x, numBlocks_y, numBlocks_z);
    complexPointwiseMulAndScale<<<numBlocks, threadsPerBlock>>>
			(d_signal, d_RIR, (N_fft/2+1), M_rcv, M_src, 1.0f/N_fft);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	
	// Transform signal back
    gpuErrchk( cufftExecC2R(plan_RIR_inv, (cufftComplex *)d_RIR, (cufftReal *)d_RIR) );
	
	// Copy device memory to host
	int conv_len = segment_len + RIR_len - 1;
    float *convolved_segments = (float *)malloc(sizeof(float)*M_src*M_rcv*conv_len);
	cudaPitchedPtr d_convolved_segments_pitchedPtr = make_cudaPitchedPtr( (void*) d_RIR, 
		(N_fft+2)*sizeof(float), conv_len, M_rcv );
	cudaPitchedPtr h_convolved_segments_pitchedPtr = make_cudaPitchedPtr( (void*) convolved_segments, 
		conv_len*sizeof(float), conv_len, M_rcv );
	cudaMemcpy3DParms parmsCopy = {0};
	parmsCopy.srcPtr = d_convolved_segments_pitchedPtr;
	parmsCopy.dstPtr = h_convolved_segments_pitchedPtr;
	parmsCopy.extent = make_cudaExtent(conv_len*sizeof(float), M_rcv, M_src);
	parmsCopy.kind = cudaMemcpyDeviceToHost;
	gpuErrchk( cudaMemcpy3D(&parmsCopy) );

	//Destroy CUFFT context
    gpuErrchk( cufftDestroy(plan_signal) );
    gpuErrchk( cufftDestroy(plan_RIR) );
    gpuErrchk( cufftDestroy(plan_RIR_inv) );

    // cleanup memory
    gpuErrchk( cudaFree(d_signal) );
    gpuErrchk( cudaFree(d_RIR) );
	
	return convolved_segments;
}

gpuRIR_cuda::gpuRIR_cuda(bool mPrecision, bool lut) {
	// Get CUDA architecture
	cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
	cuda_arch = prop.major*100 + prop.minor*10;

	// Activate mixed precision and lut if selected
	activate_mixed_precision(mPrecision);
	activate_lut(lut);
	
	// Initiate CUDA runtime API
	float* memPtr_warmup;
	gpuErrchk( cudaMalloc(&memPtr_warmup, 1*sizeof(float)) );
	gpuErrchk( cudaFree(memPtr_warmup) );
	
	// Initiate cuFFT library
	cufftHandle plan_warmup;
	gpuErrchk( cufftPlan1d(&plan_warmup,  1024, CUFFT_R2C, 1) );
	gpuErrchk( cufftDestroy(plan_warmup) );

	// Initialize cuRAND generator
	gpuErrchk( curandCreateGenerator(&cuRandGenWrap.gen, CURAND_RNG_PSEUDO_DEFAULT) );
	gpuErrchk( curandSetPseudoRandomGeneratorSeed(cuRandGenWrap.gen, 1234ULL) );
}

bool gpuRIR_cuda::activate_mixed_precision(bool activate) {
	if (cuda_arch >= 530) {
		if (activate && lookup_table) {
			printf("The mixed precision implementation is not compatible with the lookup table.\n");
			printf("Disabling the lookup table.\n");
			lookup_table = false;
		}
		mixed_precision = activate;
	} else {
		if (activate) printf("This feature requires Pascal GPU architecture or higher.\n");
		mixed_precision = false;
	}
	return mixed_precision;
}

bool gpuRIR_cuda::activate_lut(bool activate) {
	if (activate && mixed_precision) {
		printf("The lookup table is not compatible with the mixed precision implementation.\n");
		printf("Disabling the mixed precision implementation.\n");
		mixed_precision = false;
	}
	lookup_table = activate;
	return lookup_table;
}
