# https://stackoverflow.com/questions/51907755/building-a-pybind11-module-with-cpp-and-cuda-sources-using-cmake

cmake_minimum_required(VERSION 3.12)

project(gpuRIR LANGUAGES CXX CUDA)
find_package(CUDA)

# Select several CUDA architectures
include(FindCUDA)
cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS Common) # Change Common by Auto for autodetect
string(REPLACE ";" " " CUDA_ARCH_FLAGS "${CUDA_ARCH_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS}")

set(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -D_FORCE_INLINES")
# set(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -lineinfo")

add_library(gpuRIRcu STATIC
    src/gpuRIR_cuda.cu
)

set_target_properties(gpuRIRcu PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    # CUDA_SEPARABLE_COMPILATION ON
)

add_library(gpuRIR_bind MODULE
    src/python_bind.cpp
)

set(PYBIND11_CPP_STANDARD -std=c++11)
add_subdirectory(third_party/pybind11)

set_target_properties(gpuRIR_bind PROPERTIES 
    CXX_VISIBILITY_PRESET "hidden"
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# Optional IPO. Do not use IPO if it's not supported by compiler.
include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if(result)
    set_target_properties(gpuRIR_bind PROPERTIES 
        INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(WARNING "IPO is not supported: ${output}")
endif()

target_link_libraries(gpuRIR_bind PRIVATE ${CUDA_curand_LIBRARY})
target_link_libraries(gpuRIR_bind PRIVATE ${CUDA_CUFFT_LIBRARIES})
target_link_libraries(gpuRIR_bind PRIVATE gpuRIRcu)
target_link_libraries(gpuRIR_bind PRIVATE pybind11::module)
