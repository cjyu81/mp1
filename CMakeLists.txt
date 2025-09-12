cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Option to skip GPU build
option(CPU_ONLY "Build only CPU implementation" OFF)

# Option to toggle FMA
option(USE_FMA "Enable fused-multiply-add vectorization" ON)

if(CPU_ONLY)
    project(cs521_mp1 LANGUAGES CXX)
else()
    project(cs521_mp1 LANGUAGES CUDA CXX)
endif()

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CPU_ONLY)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# CPU target
add_executable(mp1_cpu ../cpu/gemm_cpu.cpp)

# Apply OpenMP flags
if (OpenMP_CXX_FOUND)
    if (MSVC)
        target_compile_options(mp1_cpu PRIVATE /openmp:experimental)
    else()
        target_compile_options(mp1_cpu PRIVATE ${OpenMP_CXX_FLAGS})
        target_link_libraries(mp1_cpu PRIVATE ${OpenMP_CXX_FLAGS})
    endif()
else()
    message(FATAL_ERROR "Need OpenMP")
endif()

# Apply optimization + FMA flags depending on USE_FMA
if (MSVC)
    if (USE_FMA)
        message(STATUS "FMA enabled")
        target_compile_options(mp1_cpu PRIVATE
            $<$<CONFIG:Release>:/O2>
            $<$<CONFIG:Release>:/fp:fast>
            $<$<CONFIG:Release>:/arch:AVX2>
        )
    else()
        message(STATUS "FMA disabled")
        target_compile_options(mp1_cpu PRIVATE
            $<$<CONFIG:Release>:/O2>
            $<$<CONFIG:Release>:/fp:precise>
            $<$<CONFIG:Release>:/arch:AVX2>
        )
    endif()
endif()

# GPU target only if CUDA is enabled
if(NOT CPU_ONLY)
    find_package(CUDAToolkit REQUIRED)
    add_executable(mp1_gpu ../gpu/gemm_gpu.cu)
    target_link_libraries(mp1_gpu PRIVATE CUDA::cublas CUDA::cublasLt)
endif()
