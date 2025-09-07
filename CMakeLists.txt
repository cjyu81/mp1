cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Option to skip GPU build
option(CPU_ONLY "Build only CPU implementation" OFF)

if(CPU_ONLY)
    project(cs521_mp1 LANGUAGES CXX)
else()
    project(cs521_mp1 LANGUAGES CUDA CXX)
endif()

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Only set architectures if CUDA is enabled
if(NOT CPU_ONLY)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(FATAL_ERROR "Need OpenMP")
endif()

# CPU target
add_executable(mp1_cpu ../cpu/gemm_cpu.cpp)

# GPU target only if CUDA is enabled
if(NOT CPU_ONLY)
    find_package(CUDAToolkit REQUIRED)
    add_executable(mp1_gpu ../gpu/gemm_gpu.cu)
    target_link_libraries(mp1_gpu PRIVATE CUDA::cublas CUDA::cublasLt)
endif()
