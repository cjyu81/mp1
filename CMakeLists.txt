cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Option to skip GPU build
option(CPU_ONLY "Build only CPU implementation" OFF)

if(CPU_ONLY)
    project(cs521_mp1 LANGUAGES CXX)
else()
    project(cs521_mp1 LANGUAGES CUDA CXX)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# ------------------------------
# Find and configure OpenMP
# ------------------------------
find_package(OpenMP REQUIRED)

if (OpenMP_CXX_FOUND)
    if (MSVC)
        # Enable SIMD directives (like '#pragma omp simd')
        add_compile_options(/openmp:experimental)
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
else()
    message(FATAL_ERROR "Need OpenMP")
endif()

# ------------------------------
# CPU target
# ------------------------------
add_executable(mp1_cpu ../cpu/gemm_cpu.cpp)

if (MSVC)
    # Create folder for assembly outputs
    set(ASM_OUTPUT_DIR ${CMAKE_BINARY_DIR}/asm)
    file(MAKE_DIRECTORY ${ASM_OUTPUT_DIR})

    # Force vectorization and generate assembly listings
    target_compile_options(mp1_cpu PRIVATE 
        $<$<CONFIG:Release>:/O2>                  # optimize
        $<$<CONFIG:Release>:/fp:fast>             # allow FMA contraction
        $<$<CONFIG:Release>:/arch:AVX2>            # enable AVX2
        $<$<CONFIG:Release>:/FAcs>                # output assembly+source+machine code
        $<$<CONFIG:Release>:/Fa${ASM_OUTPUT_DIR}/> # put assembly files in build/asm
    )
endif()

# ------------------------------
# GPU target (only if not CPU_ONLY)
# ------------------------------
if(NOT CPU_ONLY)
    find_package(CUDAToolkit REQUIRED)
    add_executable(mp1_gpu ../gpu/gemm_gpu.cu)
    target_link_libraries(mp1_gpu PRIVATE CUDA::cublas CUDA::cublasLt)
endif()
