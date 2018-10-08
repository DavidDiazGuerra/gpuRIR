#include <iostream>
#include <stdio.h>

#include "/usr/local/cuda/include/cuda.h"
#include "/usr/local/cuda/include/cuda_runtime.h"
#include <curand.h>

#include <vector>

#define PI 3.141592654f

typedef float scalar_t;

/******************************/
/* Parallelization parameters */
/******************************/

// Image Source Method
const int nThreadsISM_x = 4;
const int nThreadsISM_y = 4;
const int nThreadsISM_z = 4;

// Time vector generation
const int nThreadsTime = 128;

// RIR computation
const int initialReductionMin = 512;
const int nThreadsGen_t = 32;
const int nThreadsGen_m = 4;
const int nThreadsGen_n = 1; // Don't change it
const int nThreadsRed = 128;

// Power envelope prediction
const int nThreadsEnvPred_x = 4;
const int nThreadsEnvPred_y = 4;
const int nThreadsEnvPred_z = 1; // Don't change it

// Generate diffuse reverberation
const int nThreadsDiff_t = 16;
const int nThreadsDiff_src = 4;
const int nThreadsDiff_rcv = 2;

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
      fprintf(stderr,"GPUassert: %s %s %d\n", code, file, line);
      if (abort) exit(code);
   }
}


/*****************************/
/* Auxiliar device functions */
/*****************************/

__device__ __forceinline__ scalar_t hanning_window(scalar_t t, scalar_t Tw) {
	return 0.5f * (1.0f + __cosf(2.0f*PI*t/Tw));
}

__device__ __forceinline__ scalar_t sinc(scalar_t x) {
	return (x==0)? 1 : sinf(x)/x; 
}

__device__ __forceinline__ scalar_t image_sample(scalar_t amp, scalar_t tau, scalar_t t, scalar_t Fs) {
	scalar_t Tw = 8e-3f; // Window duration [s]
	return (abs(t-tau)<Tw/2)? hanning_window(t-tau, Tw) * amp * sinc( (t - tau) * Fs * PI ) : 0.0f;
}

__device__ __forceinline__ scalar_t SabineT60( scalar_t room_sz_x, scalar_t room_sz_y, scalar_t room_sz_z,
							scalar_t beta_x1, scalar_t beta_x2, scalar_t beta_y1, scalar_t beta_y2, scalar_t beta_z1, scalar_t beta_z2 ) {
	scalar_t Sa = ((1.0f-beta_x1*beta_x1) + (1.0f-beta_x2*beta_x2)) * room_sz_y * room_sz_z +
				  ((1.0f-beta_y1*beta_y1) + (1.0f-beta_y2*beta_y2)) * room_sz_x * room_sz_z +
				  ((1.0f-beta_z1*beta_z1) + (1.0f-beta_z2*beta_z2)) * room_sz_x * room_sz_y;
	scalar_t V = room_sz_x * room_sz_y * room_sz_z;
	return 0.161f * V / Sa;
}

/***********/
/* KERNELS */
/***********/

__global__ void calcAmpTau_kernel(scalar_t* g_amp /*[M_src]M_rcv][nb_img_x][nb_img_y][nb_img_z]*/, 
								  scalar_t* g_tau /*[M_src]M_rcv][nb_img_x][nb_img_y][nb_img_z]*/, 
								  scalar_t* g_tau_dp /*[M_src]M_rcv]*/,
								  scalar_t* g_pos_src/*[M_src][3]*/, scalar_t* g_pos_rcv/*[M_rcv][3]*/, 
								  scalar_t room_sz_x, scalar_t room_sz_y, scalar_t room_sz_z,
								  scalar_t beta_x1, scalar_t beta_x2, scalar_t beta_y1, scalar_t beta_y2, scalar_t beta_z1, scalar_t beta_z2, 
								  int nb_img_x, int nb_img_y, int nb_img_z,
								  int M_src, int M_rcv, scalar_t c) {	
	
	extern __shared__ scalar_t sdata[];
		
	int n[3];
	n[0] = blockIdx.x * blockDim.x + threadIdx.x;
	n[1] = blockIdx.y * blockDim.y + threadIdx.y;
	n[2] = blockIdx.z * blockDim.z + threadIdx.z;
	
	int N[3];
	N[0] = nb_img_x;
	N[1] = nb_img_y;
	N[2] = nb_img_z;
	
	scalar_t room_sz[3];
	room_sz[0] = room_sz_x;
	room_sz[1] = room_sz_y;
	room_sz[2] = room_sz_z;
	
	scalar_t beta[6];
	beta[0] = - beta_x1;
	beta[1] = - beta_x2;
	beta[2] = - beta_y1;
	beta[3] = - beta_y2;
	beta[4] = - beta_z1;
	beta[5] = - beta_z2;
	
	int prodN = N[0]*N[1]*N[2];
	int n_idx = n[0]*N[1]*N[2] + n[1]*N[2] + n[2];
	
	// Copy g_pos_src to shared memory
	scalar_t* sh_pos_src = (scalar_t*) sdata;
	if (threadIdx.y==0 && threadIdx.z==0)  {
		for (int m=threadIdx.x; m<M_src; m+=blockDim.x) {
			sh_pos_src[m*3  ] = g_pos_src[m*3  ];
			sh_pos_src[m*3+1] = g_pos_src[m*3+1];
			sh_pos_src[m*3+2] = g_pos_src[m*3+2];
		}
	}
	
	// Copy g_pos_rcv to shared memory
	scalar_t* sh_pos_rcv = &sh_pos_src[M_src*3];
	if (threadIdx.x==0 && threadIdx.z==0)  {
		for (int m=threadIdx.y; m<M_rcv; m+=blockDim.y) {
			sh_pos_rcv[m*3  ] = g_pos_rcv[m*3  ];
			sh_pos_rcv[m*3+1] = g_pos_rcv[m*3+1];
			sh_pos_rcv[m*3+2] = g_pos_rcv[m*3+2];
		}
	}
	
	// Wait until the copies are completed
	__syncthreads();
	
	if (n[0]<N[0] & n[1]<N[1] & n[2]<N[2]) {
		
		// Common factors for each src and rcv
		scalar_t rflx_att = 1;
		scalar_t clust_pos[3];
		int clust_idx[3];
		int rflx_idx[3];
		bool direct_path = true;
		for (int d=0; d<3; d++) {
			clust_idx[d] = __float2int_ru((n[d] - N[d]/2) / 2.0f); 
			clust_pos[d] = clust_idx[d] * 2*room_sz[d];
			rflx_idx[d] = abs((n[d] - N[d]/2) % 2); // 1 means reflected in dimension d
			rflx_att *= powf(beta[d*2], abs(clust_idx[d]-rflx_idx[d])) * powf(beta[d*2+1], abs(clust_idx[d]));
			direct_path *= (clust_idx[d]==0)&&(rflx_idx[d]==0);
		}
			
		// Individual factors for each src and rcv
		for (int m_src=0; m_src<M_src; m_src++) {
			for (int m_rcv=0; m_rcv<M_rcv; m_rcv++) {
				scalar_t vec[3];
				scalar_t dist = 0;
				for (int d=0; d<3; d++) {
					vec[d] = clust_pos[d] + (1-2*rflx_idx[d]) * sh_pos_src[m_src*3+d] - sh_pos_rcv[m_rcv*3+d];
					dist += vec[d] * vec[d];
				}
				dist = sqrtf(dist);
				//printf("%d, \n", m_src*M_rcv*prodN + m_rcv*prodN + n_idx);
				g_amp[m_src*M_rcv*prodN + m_rcv*prodN + n_idx] = rflx_att / (4*PI*dist);
				g_tau[m_src*M_rcv*prodN + m_rcv*prodN + n_idx] = dist / c;

				if (direct_path) g_tau_dp[m_src*M_rcv + m_rcv] = dist / c;
				
				//if (n[0] == 3) printf("%d %d %d %f %f\n", n[0], n[1], n[2], dist / c, rflx_att / (4*PI*dist));
			}
		}
	}
}

__global__ void generateTime_kernel(scalar_t* t, scalar_t Fs, int nSamples) {
	int sample = blockIdx.x * blockDim.x + threadIdx.x;
	if (sample<nSamples) {t[sample] = sample/Fs; /* printf("%d %f \n", sample, t[sample]); */} 
}

__global__ void generateRIR_kernel(scalar_t* initialRIR, scalar_t* tim, scalar_t* amp, scalar_t* tau, int T, int M, int N, int iniRIR_N, int ini_red, scalar_t Fs) {	
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int m = blockIdx.y * blockDim.y + threadIdx.y;
	int n_ini = blockIdx.z * ini_red;
	int n_max = fminf(n_ini + ini_red, N);
	
	if (m<M && t<T) {
		scalar_t loc_sum = 0;
		scalar_t loc_tim = tim[t];		
		for (int n=n_ini; n<n_max; n++) {
			loc_sum += image_sample(amp[m*N+n], tau[m*N+n], loc_tim, Fs);
			//if (t==19685) printf("%d %f %f %f %f\n", n, loc_tim, tau[m*N+n], amp[m*N+n], image_sample(amp[m*N+n], tau[m*N+n], loc_tim, Fs));
		}
		initialRIR[m*T*iniRIR_N + t*iniRIR_N + blockIdx.z] = loc_sum;
	}
}

__global__ void reduceRIR_kernel(scalar_t* initialRIR, scalar_t* intermediateRIR, int M, int T, int N, int intRIR_N) {
	extern __shared__ scalar_t sdata[];
	
	int tid = threadIdx.x;
	int n = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	int t = blockIdx.y; //*blockDim.y + threadIdx.y;
	int m = blockIdx.z; //*blockDim.z + threadIdx.z;
	
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

__global__ void envPred_kernel(scalar_t* A /*[M_src]M_rcv]*/, scalar_t* alpha /*[M_src]M_rcv]*/, 
						scalar_t* RIRs_early /*[M_src][M_rcv][nSamples]*/, scalar_t* tau_dp, /*[M_src]M_rcv]*/
						int M_src, int M_rcv, int nSamples, scalar_t fs,
						scalar_t room_sz_x, scalar_t room_sz_y, scalar_t room_sz_z,
						scalar_t beta_x1, scalar_t beta_x2, scalar_t beta_y1, scalar_t beta_y2, scalar_t beta_z1, scalar_t beta_z2) {
		
	scalar_t w_sz = 10e-3f; // Maximum window size (s) to compute the final power of the early RIRs_early
	
	int m_src = blockIdx.x * blockDim.x + threadIdx.x;
	int m_rcv = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (m_src<M_src && m_rcv<M_rcv) {
		int w_start = __float2int_ru( max(nSamples/fs-w_sz, tau_dp[m_src*M_rcv+m_rcv]) * fs );
		scalar_t w_center = (w_start + (nSamples-w_start)/2.0) / fs;
		
		scalar_t finalPower = 0.0f;
		for (int t=w_start; t<nSamples; t++) {
			scalar_t aux = RIRs_early[m_src*M_rcv*nSamples + m_rcv*nSamples + t];
			finalPower += aux*aux;
		}
		finalPower /= nSamples-w_start;
		
		scalar_t T60 = SabineT60(room_sz_x, room_sz_y, room_sz_z, beta_x1, beta_x2, beta_y1, beta_y2, beta_z1, beta_z2);
		scalar_t loc_alpha = -13.8155f / T60; //-13.8155 == log(10^(-6))
		
		A[m_src*M_rcv + m_rcv] = finalPower / expf(loc_alpha*(w_center-tau_dp[m_src*M_rcv+m_rcv]));
		alpha[m_src*M_rcv + m_rcv] = loc_alpha;
		
		//printf("T60[%d][%d] = %f\n", m_src, m_rcv, T60);
		//printf("A[%d][%d] = %f\n", m_src, m_rcv, A[m_src*M_rcv + m_rcv]);
		//printf("alpha[%d][%d] = %f\n", m_src, m_rcv, alpha[m_src*M_rcv + m_rcv]);
	}
}

__global__ void diffRev_kernel(scalar_t* rir, scalar_t* tim, scalar_t* A, scalar_t* alpha, scalar_t* tau_dp, 
							   int M_src, int M_rcv, int nSamples) {
	
	int sample = blockIdx.x * blockDim.x + threadIdx.x;
	int m_src  = blockIdx.y * blockDim.y + threadIdx.y;
	int m_rcv  = blockIdx.z * blockDim.z + threadIdx.z;
	
	if (sample<nSamples && m_src<M_src && m_rcv<M_rcv) {
		// Get logistic distribution from uniform distribution
		scalar_t uniform = rir[m_src*M_rcv*nSamples + m_rcv*nSamples + sample];
		scalar_t logistic = 0.551329f * logf(uniform/(1.0f - uniform)); // 0.551329 == sqrt(3)/pi
		
		// Apply power envelope
		scalar_t pow_env = A[m_src*M_rcv+m_rcv] * expf(alpha[m_src*M_rcv+m_rcv] * (tim[sample]-tau_dp[m_src*M_rcv+m_rcv]));
		rir[m_src*M_rcv*nSamples + m_rcv*nSamples + sample] = sqrt(pow_env) * logistic;
		//if (sample==0) printf("A[%d][%d] = %f\n", m_src, m_rcv, A[m_src*M_rcv + m_rcv]);
	}
}

/***************************/
/* Auxiliar host functions */
/***************************/

scalar_t* cuda_rirGenerator(scalar_t* rir, scalar_t* x, scalar_t* amp, scalar_t* tau, int M, int N, int T, scalar_t Fs) {
	int initialReduction = initialReductionMin;
	while (M * T * ceil((float)N/initialReduction) > 1e9) initialReduction *= 2;
	
	int iniRIR_N = ceil((float)N/initialReduction);
	dim3 threadsPerBlockIni(nThreadsGen_t, nThreadsGen_m, nThreadsGen_n);
	dim3 numBlocksIni(ceil((float)T/threadsPerBlockIni.x), ceil((float)M/threadsPerBlockIni.y), iniRIR_N);
	
	scalar_t* initialRIR;
	gpuErrchk( cudaMalloc(&initialRIR, M*T*iniRIR_N*sizeof(scalar_t)) );
	
	//printf("initialReduction = %d\n", initialReduction);
	//printf("generateRIR_kernel<<<(%d, %d, %d), (%d, %d, %d)>>>\n", numBlocksIni.x, numBlocksIni.y, numBlocksIni.z, threadsPerBlockIni.x, threadsPerBlockIni.y, threadsPerBlockIni.z);
	generateRIR_kernel<<<numBlocksIni, threadsPerBlockIni>>>( initialRIR, x, amp, tau, T, M, N, iniRIR_N, initialReduction, Fs );
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	
	dim3 threadsPerBlockRed(nThreadsRed, 1, 1);
	scalar_t* intermediateRIR;
	int intRIR_N;
	while (iniRIR_N > 2*nThreadsRed) {		
		intRIR_N = ceil((float)iniRIR_N / (2*nThreadsRed));
		gpuErrchk( cudaMalloc(&intermediateRIR, intRIR_N * T * M * sizeof(scalar_t)) );

		dim3 numBlocksRed(intRIR_N, T, M);
		reduceRIR_kernel<<<numBlocksRed, threadsPerBlockRed, nThreadsRed*sizeof(scalar_t)>>>(
			initialRIR, intermediateRIR, M, T, iniRIR_N, intRIR_N);
		gpuErrchk( cudaDeviceSynchronize() );
		gpuErrchk( cudaPeekAtLastError() );
		
		gpuErrchk( cudaFree(initialRIR) );
		initialRIR = intermediateRIR;		
		iniRIR_N = intRIR_N;
	}
	
	dim3 numBlocksEnd(1, T, M);
	reduceRIR_kernel<<<numBlocksEnd, threadsPerBlockRed, nThreadsRed*sizeof(scalar_t)>>>(
		initialRIR, rir, M, T, iniRIR_N, 1);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaFree(initialRIR) );
	
	return rir;
}

/**********************/
/* Principal function */
/**********************/

scalar_t* cuda_simulateRIR(scalar_t room_sz[3], scalar_t beta[6], scalar_t* h_pos_src, int M_src, scalar_t* h_pos_rcv, int M_rcv, 
						   int nb_img[3], scalar_t Tdiff, scalar_t Tmax, scalar_t Fs=16000.0f, scalar_t c=343.0f) {	
	// Copy host memory to GPU
	scalar_t *pos_src, *pos_rcv;
	gpuErrchk( cudaMalloc(&pos_src, M_src*3*sizeof(scalar_t)) );
	gpuErrchk( cudaMalloc(&pos_rcv, M_rcv*3*sizeof(scalar_t)) );
	gpuErrchk( cudaMemcpy(pos_src, h_pos_src, M_src*3*sizeof(scalar_t), cudaMemcpyHostToDevice ) );
	gpuErrchk( cudaMemcpy(pos_rcv, h_pos_rcv, M_rcv*3*sizeof(scalar_t), cudaMemcpyHostToDevice ) );
	
	
	// Use the ISM to calculate the amplitude and delay of each image
	dim3 threadsPerBlockISM(nThreadsISM_x, nThreadsISM_y, nThreadsISM_z);
	dim3 numBlocksISM(ceil((float)nb_img[0] / nThreadsISM_x), 
					  ceil((float)nb_img[1] / nThreadsISM_y), 
					  ceil((float)nb_img[2] / nThreadsISM_z));
	int shMemISM = (M_src + M_rcv) * 3 * sizeof(scalar_t);
	
	scalar_t* amp;
	gpuErrchk( cudaMalloc(&amp, M_src*M_rcv*nb_img[0]*nb_img[1]*nb_img[2]*sizeof(scalar_t)) );
	scalar_t* tau;
	gpuErrchk( cudaMalloc(&tau, M_src*M_rcv*nb_img[0]*nb_img[1]*nb_img[2]*sizeof(scalar_t)) );
	scalar_t* tau_dp; // Direct path delay
	gpuErrchk( cudaMalloc(&tau_dp, M_src*M_rcv*sizeof(scalar_t)) );
	
	calcAmpTau_kernel<<<numBlocksISM, threadsPerBlockISM, shMemISM>>> (
		amp, tau, tau_dp,
		pos_src, pos_rcv, 
		room_sz[0], room_sz[1], room_sz[2], 
		beta[0], beta[1], beta[2], beta[3], beta[4], beta[5], 
		nb_img[0], nb_img[1], nb_img[2],
		M_src, M_rcv, c
	);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	
	// Generate a vector with the time instant of each sample
	int nSamplesISM = ceil(Tdiff*Fs);
	int nSamples = ceil(Tmax*Fs);
	int nSamplesDiff = nSamples - nSamplesISM;
	//printf("nSamplesISM = %d\nnSamplesDiff = %d\nnSamples = %d\n", nSamplesISM, nSamplesDiff, nSamples);
	scalar_t* time;
	gpuErrchk( cudaMalloc(&time, nSamples*sizeof(scalar_t)) );
	generateTime_kernel<<<ceil((float)nSamples/nThreadsTime), nThreadsTime>>>(time, Fs, nSamples);
	
	// Compute the RIRs as a sum of sincs
	int M = M_src * M_rcv;
	int N = nb_img[0] * nb_img[1] * nb_img[2];
	scalar_t* rirISM;
	gpuErrchk( cudaMalloc(&rirISM, M*nSamplesISM*sizeof(scalar_t)) );
	cuda_rirGenerator(rirISM, time, amp, tau, M, N, nSamplesISM, Fs);
	
	// Compute the exponential power envelope parammeters of each RIR
	dim3 threadsPerBlockEnvPred(nThreadsEnvPred_x, nThreadsEnvPred_y, nThreadsEnvPred_z);
	dim3 numBlocksEnvPred(ceil((float)M_src / nThreadsEnvPred_x), 
						  ceil((float)M_rcv / nThreadsEnvPred_y), 1);
					  
	scalar_t* A; // pow_env = A * exp(alpha * (t-tau_dp))
	gpuErrchk( cudaMalloc(&A, M_src*M_rcv*sizeof(scalar_t)) );
	scalar_t* alpha;
	gpuErrchk( cudaMalloc(&alpha, M_src*M_rcv*sizeof(scalar_t)) );
	
	envPred_kernel<<<numBlocksEnvPred, threadsPerBlockEnvPred>>>(
			A, alpha, rirISM, tau_dp, M_src, M_rcv, nSamplesISM, Fs,
			room_sz[0], room_sz[1], room_sz[2], beta[0], beta[1], beta[2], beta[3], beta[4], beta[5]);
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
	
	// Generate diffuse reverberation
	scalar_t* rirDiff; 
	gpuErrchk( cudaMalloc(&rirDiff, M_src*M_rcv*nSamplesDiff*sizeof(scalar_t)) );
	
	if (nSamplesDiff != 0) {
		curandGenerator_t gen; // Fill rirDiff with random numbers with uniform distribution
		gpuErrchk( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
		gpuErrchk( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
		gpuErrchk( curandGenerateUniform(gen, rirDiff, M_src*M_rcv*nSamplesDiff) );
		gpuErrchk( cudaDeviceSynchronize() );
		gpuErrchk( cudaPeekAtLastError() );
		
		dim3 threadsPerBlockDiff(nThreadsDiff_t, nThreadsDiff_src, nThreadsDiff_rcv);
		dim3 numBlocksDiff(ceil((float)nSamplesDiff / nThreadsDiff_t),
							  ceil((float)M_src / nThreadsDiff_src), 
							  ceil((float)M_rcv / nThreadsDiff_rcv));
		diffRev_kernel<<<numBlocksDiff, threadsPerBlockDiff>>>(
				rirDiff, &time[nSamplesISM], A, alpha, tau_dp, M_src, M_rcv, nSamplesDiff);
		gpuErrchk( cudaDeviceSynchronize() );
		gpuErrchk( cudaPeekAtLastError() );
	}
	
	// Copy GPU memory to host
	int rirSizeISM = M_src * M_rcv * nSamplesISM * sizeof(scalar_t);
	int rirSizeDiff = M_src * M_rcv * nSamplesDiff * sizeof(scalar_t);
	scalar_t* h_rir = (scalar_t*) malloc(rirSizeISM+rirSizeDiff);
	
	cudaPitchedPtr h_rir_pitchedPtr = make_cudaPitchedPtr( (void*) h_rir, 
		(nSamplesISM+nSamplesDiff)*sizeof(scalar_t), nSamplesISM+nSamplesDiff, M_rcv );
	cudaPitchedPtr rirISM_pitchedPtr = make_cudaPitchedPtr( (void*) rirISM, 
		nSamplesISM*sizeof(scalar_t), nSamplesISM, M_rcv );
	cudaPitchedPtr rirDiff_pitchedPtr = make_cudaPitchedPtr( (void*) rirDiff, 
		nSamplesDiff*sizeof(scalar_t), nSamplesDiff, M_rcv );
	
	cudaMemcpy3DParms parmsISM = {0};
	parmsISM.srcPtr = rirISM_pitchedPtr;
	parmsISM.dstPtr = h_rir_pitchedPtr;
	parmsISM.extent = make_cudaExtent(nSamplesISM*sizeof(scalar_t), M_rcv, M_src);
	parmsISM.kind = cudaMemcpyDeviceToHost;
	gpuErrchk( cudaMemcpy3D(&parmsISM) );
	
	if (nSamplesDiff > 0) {
		cudaMemcpy3DParms parmsDiff = {0};
		parmsDiff.srcPtr = rirDiff_pitchedPtr;
		parmsDiff.dstPtr = h_rir_pitchedPtr;
		parmsDiff.dstPos = make_cudaPos(nSamplesISM*sizeof(scalar_t), 0, 0);
		parmsDiff.extent = make_cudaExtent(nSamplesDiff*sizeof(scalar_t), M_rcv, M_src);
		parmsDiff.kind = cudaMemcpyDeviceToHost;
		gpuErrchk( cudaMemcpy3D(&parmsDiff) );
	}

	// Free memory
	gpuErrchk( cudaFree(pos_src) );
	gpuErrchk( cudaFree(pos_rcv) );
	gpuErrchk( cudaFree(amp)	 );
	gpuErrchk( cudaFree(tau)	 );
	gpuErrchk( cudaFree(tau_dp)	 );
	gpuErrchk( cudaFree(time)	 );
	gpuErrchk( cudaFree(rirISM)	 );
	gpuErrchk( cudaFree(A)		 );
	gpuErrchk( cudaFree(alpha)	 );
	gpuErrchk( cudaFree(rirDiff) );
	
	return h_rir;
}
