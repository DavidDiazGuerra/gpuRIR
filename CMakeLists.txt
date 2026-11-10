# https://stackoverflow.com/questions/51907755/building-a-pybind11-module-with-cpp-and-cuda-sources-using-cmake

cmake_minimum_required(VERSION 3.23)

option(USE_HIP "Build with HIP for AMD GPUs instead of CUDA" OFF)

if(USE_HIP)
    # The existing .cu file is compiled as HIP and sees the HIP toolchain via a
    # single force-included compatibility header. Host C++ (python_bind.cpp,
    # pybind11) is untouched.
    # project(... LANGUAGES ... HIP) enables HIP, which honors an explicit
    # -DCMAKE_HIP_ARCHITECTURES, otherwise auto-detects the host GPU(s), and
    # errors on a no-GPU build host.
    project(gpuRIR LANGUAGES CXX HIP)
    enable_language(HIP)
    find_package(hipfft REQUIRED)
    find_package(hiprand REQUIRED)
else()
    project(gpuRIR LANGUAGES CXX CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Select several CUDA architectures
    #include(FindCUDA)
    #cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS Common) # Change Common by Auto for autodetect
    #string(REPLACE ";" " " CUDA_ARCH_FLAGS "${CUDA_ARCH_FLAGS}")
    #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS}")

    set(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -D_FORCE_INLINES")
    # set(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()

add_library(gpuRIRcu STATIC
    src/gpuRIR_cuda.cu
)

if(USE_HIP)
    set_source_files_properties(src/gpuRIR_cuda.cu PROPERTIES LANGUAGE HIP)
    set_target_properties(gpuRIRcu PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    target_compile_definitions(gpuRIRcu PRIVATE USE_HIP)
    # Force-include the compat header so the .cu's plain CUDA spelling resolves
    # to HIP without editing every include.
    target_compile_options(gpuRIRcu PRIVATE
        -include ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_to_hip.h)
else()
    set_property(TARGET gpuRIRcu PROPERTY CUDA_ARCHITECTURES all-major)
endif()

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
# Skip IPO for the HIP build: LTO bitcode from the pybind11 module is not
# finalized by the HIP link step, producing an .so with no PyInit symbol.
if(NOT USE_HIP)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if(result)
        set_target_properties(gpuRIR_bind PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

if(USE_HIP)
    target_link_libraries(gpuRIRcu PUBLIC
        hip::hipfft
        hip::hiprand
    )
else()
    target_link_libraries(gpuRIRcu PUBLIC
        CUDA::cufft
        CUDA::curand
    )
endif()
target_link_libraries(gpuRIR_bind PRIVATE
    gpuRIRcu
    pybind11::module
)
