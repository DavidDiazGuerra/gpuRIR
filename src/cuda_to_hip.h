//********************************************************//
// CUDA-to-HIP compatibility shim for gpuRIR              //
//                                                        //
// Minimal-footprint port: gpuRIR_cuda.cu                  //
// keeps its plain CUDA spelling. On AMD this header pulls  //
// in the HIP runtime, hipFFT and hipRAND, and #defines     //
// the CUDA/cuFFT/cuRAND symbols the project uses to their   //
// HIP equivalents. On NVIDIA it is a no-op that pulls in   //
// the original CUDA headers.                              //
//                                                        //
// Symbol names follow PyTorch's authoritative hipify map: //
// torch/utils/hipify/cuda_to_hip_mappings.py             //
//********************************************************//

#ifndef GPURIR_CUDA_TO_HIP_H
#define GPURIR_CUDA_TO_HIP_H

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

// <cstring>/<cstdlib> must precede hip_runtime so the host libc memcpy/memset
// (used by PadData) are in scope; otherwise overload resolution in host code
// picks HIP's __device__ memcpy/memset and fails to compile.
#include <cstring>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipfft/hipfft.h>
#include <hiprand/hiprand.h>

// ---- Error handling / sync ----
#define cudaError_t            hipError_t
#define cudaSuccess            hipSuccess
#define cudaGetErrorString     hipGetErrorString
#define cudaPeekAtLastError    hipPeekAtLastError
#define cudaDeviceSynchronize  hipDeviceSynchronize

// ---- Device management ----
#define cudaGetDeviceProperties  hipGetDeviceProperties
#define cudaDeviceProp           hipDeviceProp_t

// ---- Linear / pitched memory ----
#define cudaMalloc    hipMalloc
#define cudaFree      hipFree
#define cudaMemcpy    hipMemcpy
#define cudaMemset    hipMemset
#define cudaMemcpy2D  hipMemcpy2D
#define cudaMemset2D  hipMemset2D

// ---- 3D pitched copies ----
#define cudaMemcpy3D          hipMemcpy3D
#define cudaMemcpy3DParms     hipMemcpy3DParms
#define cudaPitchedPtr        hipPitchedPtr
#define cudaExtent            hipExtent
#define cudaPos               hipPos
#define make_cudaPitchedPtr   make_hipPitchedPtr
#define make_cudaExtent       make_hipExtent
#define make_cudaPos          make_hipPos

// ---- memcpy kinds ----
#define cudaMemcpyHostToDevice    hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost    hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice  hipMemcpyDeviceToDevice

// ---- cuFFT -> hipFFT ----
#define cufftHandle      hipfftHandle
#define cufftComplex     hipfftComplex
#define cufftReal        hipfftReal
#define cufftResult_t    hipfftResult_t
#define cufftPlan1d      hipfftPlan1d
#define cufftExecR2C     hipfftExecR2C
#define cufftExecC2R     hipfftExecC2R
#define cufftDestroy     hipfftDestroy
#define CUFFT_SUCCESS    HIPFFT_SUCCESS
#define CUFFT_R2C        HIPFFT_R2C
#define CUFFT_C2R        HIPFFT_C2R

// ---- cuRAND -> hipRAND ----
#define curandGenerator_t                    hiprandGenerator_t
#define curandStatus_t                       hiprandStatus_t
#define CURAND_STATUS_SUCCESS                HIPRAND_STATUS_SUCCESS
#define CURAND_RNG_PSEUDO_DEFAULT            HIPRAND_RNG_PSEUDO_DEFAULT
#define curandCreateGenerator                hiprandCreateGenerator
#define curandSetPseudoRandomGeneratorSeed   hiprandSetPseudoRandomGeneratorSeed
#define curandGenerateUniform                hiprandGenerateUniform

// ---- Device intrinsics without a 1:1 HIP spelling ----
// HIP has no round-toward-zero FMA. The only use computes a LUT index
// coordinate that is then interpolated, so rounding mode is immaterial;
// the round-to-nearest FMA is the faithful HIP equivalent.
#define __fmaf_rz(a, b, c) __fmaf_rn((a), (b), (c))

#else // NVIDIA / CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuda_fp16.h>

#endif

#endif // GPURIR_CUDA_TO_HIP_H
