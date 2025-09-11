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

if(NOT CPU_ONLY)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# ---------------------
# Find and enable OpenMP
# ---------------------
find_package(OpenMP REQUIRED)
if (NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "Need OpenMP")
endif()

# ---------------------
# CPU Target
# ---------------------
add_executable(mp1_cpu ../cpu/gemm_cpu.cpp)

if (OpenMP_CXX_FOUND)
    if (MSVC)
        # Enable SIMD pragmas with MSVC
        message(STATUS "Detected MSVC: enabling /openmp:experimental")
        target_compile_options(mp1_cpu PRIVATE /openmp:experimental)
    else()
        target_compile_options(mp1_cpu PRIVATE ${OpenMP_CXX_FLAGS})
        target_link_libraries(mp1_cpu PRIVATE ${OpenMP_CXX_FLAGS})
    endif()
endif()

# ---------------------
# MSVC-specific optimizations and assembly output
# ---------------------
if (MSVC)
    # Where to write .cod files
    set(ASM_OUTPUT_DIR ${CMAKE_BINARY_DIR}/asm)
    file(MAKE_DIRECTORY ${ASM_OUTPUT_DIR})

    target_compile_options(mp1_cpu PRIVATE
        $<$<CONFIG:Release>:/O2>            # optimize for speed
        $<$<CONFIG:Release>:/fp:fast>       # allow FMA contractions
        $<$<CONFIG:Release>:/arch:AVX2>     # enable AVX2 (vectorization)
        $<$<CONFIG:Release>:/FAcs>          # produce assembly with source+machine code
        $<$<CONFIG:Release>:/Fa${ASM_OUTPUT_DIR}/> # put .cod in build/asm/
    )
endif()

# ---------------------
# GPU Target (if enabled)
# ---------------------
if(NOT CPU_ONLY)
    find_package(CUDAToolkit REQUIRED)
    add_executable(mp1_gpu ../gpu/gemm_gpu.cu)
    target_link_libraries(mp1_gpu PRIVATE CUDA::cublas CUDA::cublasLt)
endif()
