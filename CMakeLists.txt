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

# Force MSVC to emit assembly in Release builds
if (MSVC)
    target_compile_options(mp1_cpu PRIVATE 
        $<$<CONFIG:Release>:/FAcs>   # produce .asm with source + machine code
        $<$<CONFIG:Release>:/O2>     # optimization
        $<$<CONFIG:Release>:/fp:fast> # allow FMA contractions
    )
endif()

# GPU target only if CUDA is enabled
if(NOT CPU_ONLY)
    find_package(CUDAToolkit REQUIRED)
    add_executable(mp1_gpu ../gpu/gemm_gpu.cu)
    target_link_libraries(mp1_gpu PRIVATE CUDA::cublas CUDA::cublasLt)
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    if (MSVC)
        # Use experimental OpenMP for SIMD
        target_compile_options(mp1_cpu PRIVATE /openmp:experimental)
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
else()
    message(FATAL_ERROR "Need OpenMP")
endif()

if (MSVC)
    # Generate assembly into the build/asm directory
    set(ASM_OUTPUT_DIR ${CMAKE_BINARY_DIR}/asm)

    # Ensure directory exists
    file(MAKE_DIRECTORY ${ASM_OUTPUT_DIR})

    target_compile_options(mp1_cpu PRIVATE 
        $<$<CONFIG:Release>:/FAcs>             # emit asm with source + machine code
        $<$<CONFIG:Release>:/Fa${ASM_OUTPUT_DIR}/> # put asm into build/asm/
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/fp:fast>
    )
endif()
