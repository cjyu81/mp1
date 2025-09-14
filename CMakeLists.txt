cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Option to skip GPU build
option(CPU_ONLY "Build only CPU implementation" OFF)

# Option to toggle FMA
option(USE_FMA "Enable fused-multiply-add vectorization" ON)

# Project
if(CPU_ONLY)
    project(cs521_mp1 LANGUAGES CXX)
else()
    project(cs521_mp1 LANGUAGES CUDA CXX)
endif()

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CPU_ONLY)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# -------------------
# CPU target
# -------------------
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
    message(FATAL_ERROR "OpenMP required for CPU target")
endif()

# CPU optimization + FMA flags
if (MSVC)
    if (USE_FMA)
        target_compile_options(mp1_cpu PRIVATE
            $<$<CONFIG:Release>:/O2>
            $<$<CONFIG:Release>:/fp:fast>
            $<$<CONFIG:Release>:/arch:AVX2>
        )
    else()
        target_compile_options(mp1_cpu PRIVATE
            $<$<CONFIG:Release>:/O2>
            $<$<CONFIG:Release>:/fp:precise>
            $<$<CONFIG:Release>:/arch:AVX2>
        )
    endif()
endif()

# -------------------
# GPU target
# -------------------
if(NOT CPU_ONLY)
    find_package(CUDAToolkit REQUIRED)
    add_executable(mp1_gpu ../gpu/gemm_gpu.cu)
    target_link_libraries(mp1_gpu PRIVATE CUDA::cublas CUDA::cublasLt)

    # Explicitly compile for Tesla T4
    set_target_properties(mp1_gpu PROPERTIES
        CUDA_ARCHITECTURES "75"
        CUDA_STANDARD 14
        CUDA_SEPARABLE_COMPILATION ON
    )

    # Ensure nvcc generates code only for T4
    set_target_properties(mp1_gpu PROPERTIES
        CUDA_FLAGS "--generate-code=arch=compute_75,code=sm_75"
    )
endif()
