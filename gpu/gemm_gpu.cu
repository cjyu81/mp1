#include "../include/utils.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define NUM_RUNS 10

// Existing CUDA_CHECK macro for CUDA runtime API
#define CUDA_CHECK(func)                                                       \
    do {                                                                       \
        cudaError_t status = (func);                                           \
        if (status != cudaSuccess) {                                           \
            printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, \
                   cudaGetErrorString(status), status);                        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// New CUBLAS_CHECK macro for cuBLAS API
#define CUBLAS_CHECK(func)                                                     \
    do {                                                                       \
        cublasStatus_t status = (func);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            printf("cuBLAS API failed at line %d with error: %d\n", __LINE__,  \
                   status);                                                    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHECK(name) \
    float *d_Aref_ ## name, *d_Bref_ ## name, *d_Cref_ ## name; \
    std::cerr << "checking " << #name << std::endl; \
    CUDA_CHECK(cudaMalloc(&d_Aref_ ## name, Ref::M * Ref::K * sizeof(float))); \
    CUDA_CHECK(cudaMalloc(&d_Bref_ ## name, Ref::K * Ref::N * sizeof(float))); \
    CUDA_CHECK(cudaMalloc(&d_Cref_ ## name, Ref::M * Ref::N * sizeof(float))); \
    CUDA_CHECK(cudaMemcpy(d_Aref_ ## name, ref.A, Ref::M * Ref::K * sizeof(float), cudaMemcpyHostToDevice)); \
    CUDA_CHECK(cudaMemcpy(d_Bref_ ## name, ref.B, Ref::K * Ref::N * sizeof(float), cudaMemcpyHostToDevice)); \
    float* d_Cref_INI_ ## name = new float[Ref::M * Ref::N](); \
    for (int i = 0; i < Ref::M; i++) { \
        for (int j = 0; j < Ref::N; j++) { \
            d_Cref_INI_ ## name[i * Ref::N + j] = 0; \
        } \
    } \
    CUDA_CHECK(cudaMemcpy(d_Cref_ ## name, d_Cref_INI_ ## name, Ref::M * Ref::N * sizeof(float), cudaMemcpyHostToDevice)); \
    name(d_Aref_ ## name, d_Bref_ ## name, d_Cref_ ## name, Ref::M, Ref::N, Ref::K); \
    cudaError_t err_c_ ## name = cudaGetLastError(); \
    if (err_c_ ## name != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err_c_ ## name) << std::endl; \
    } \
    CUDA_CHECK(cudaMemcpy(refC, d_Cref_ ## name, Ref::M * Ref::N * sizeof(float), cudaMemcpyDeviceToHost)); \
    if (!ref.checkRef(refC)){ \
        std::cerr << "check ref failed!" << std::endl; \
    };

#define TIME(name) \
    float *d_A_ ## name, *d_B_ ## name, *d_C_ ## name; \
    CUDA_CHECK(cudaMalloc(&d_A_ ## name, M * K * sizeof(float))); \
    CUDA_CHECK(cudaMalloc(&d_B_ ## name, K * N * sizeof(float))); \
    CUDA_CHECK(cudaMalloc(&d_C_ ## name, M * N * sizeof(float))); \
    CUDA_CHECK(cudaMemcpy(d_A_ ## name, A, M * K * sizeof(float), cudaMemcpyHostToDevice)); \
    CUDA_CHECK(cudaMemcpy(d_B_ ## name, B, K * N * sizeof(float), cudaMemcpyHostToDevice)); \
    cudaEvent_t start_ ## name, end_ ## name; \
    CUDA_CHECK(cudaEventCreate(&start_ ## name)); \
    CUDA_CHECK(cudaEventCreate(&end_ ## name)); \
    float* d_C_INI_ ## name = new float[M * N](); \
    for (int i = 0; i < M; i++) { \
        for (int j = 0; j < N; j++) { \
            d_C_INI_ ## name[i * N + j] = 0; \
        } \
    } \
    for (int i = 0; i < 2; i++) { \
        CUDA_CHECK(cudaMemcpy(d_C_ ## name, d_C_INI_ ## name, M * N * sizeof(float), cudaMemcpyHostToDevice)); \
        name(d_A_ ## name, d_B_ ## name, d_C_ ## name, M, N, K); \
    } \
    cudaError_t err_t_ ## name = cudaGetLastError(); \
    if (err_t_ ## name != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err_t_ ## name) << std::endl; \
    } \
    float milliseconds_ ## name = 0; \
    for (int i = 0; i < NUM_RUNS; i++) { \
        CUDA_CHECK(cudaMemcpy(d_C_ ## name, d_C_INI_ ## name, M * N * sizeof(float), cudaMemcpyHostToDevice)); \
        CUDA_CHECK(cudaDeviceSynchronize()); \
        CUDA_CHECK(cudaEventRecord(start_ ## name)); \
        name(d_A_ ## name, d_B_ ## name, d_C_ ## name, M, N, K); \
        CUDA_CHECK(cudaEventRecord(end_ ## name)); \
        CUDA_CHECK(cudaEventSynchronize(end_ ## name)); \
        float milliseconds_ ## i = 0; \
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds_ ## i, start_ ## name, end_ ## name)); \
        milliseconds_ ## name += milliseconds_ ## i; \
    } \
    CUDA_CHECK(cudaMemcpy(C, d_C_ ## name, M * N * sizeof(float), cudaMemcpyDeviceToHost)); \
    std::cout << "Time taken for GEMM (GPU, " << #name <<"): " << milliseconds_ ## name / (float)NUM_RUNS << "ms" << std::endl; \
    CUDA_CHECK(cudaFree(d_A_ ## name)); \
    CUDA_CHECK(cudaFree(d_B_ ## name)); \
    CUDA_CHECK(cudaFree(d_C_ ## name)); \
    delete[] d_C_INI_ ## name;

// Existing kernel definitions (o0, o1, o2, o3_8, o3_16, o3_32)
__global__ void gemm_gpu_o0_kernel(float* A, float* B, float *C, int M, int N, int K) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < K; k++) {
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                }
            }
        }
    }
}

void gemm_gpu_o0(float* A, float* B, float* C, int M, int N, int K) {
	return;
    dim3 blockSize(1);
    dim3 gridSize(1);
    gemm_gpu_o0_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

__global__ void gemm_gpu_o1_kernel(float* A, float* B, float *C, int M, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < M && i < N) {
        float value = 0.0f;
        for (int k = 0; k < K; k++) {
            value += A[j * K + k] * B[k * N + i];
        }
        C[j * N + i] = value;
    }
}

void gemm_gpu_o1(float* A, float* B, float* C, int M, int N, int K) {
    const int blockdim = 16;
    dim3 blockSize(blockdim, blockdim);
    dim3 gridSize((N + blockdim - 1) / blockdim, (M + blockdim - 1) / blockdim);
    gemm_gpu_o1_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

template <int TILE>
__global__ void gemm_gpu_o2_o3_kernel(float* A, float* B, float *C, int M, int N, int K) {
    __shared__ float sharedA[TILE][TILE];
    __shared__ float sharedB[TILE][TILE];
    int i = blockIdx.x * TILE + threadIdx.x; // col
    int j = blockIdx.y * TILE + threadIdx.y; // row
    float sum = 0.0f;

    for (int a = 0; a < (K + TILE - 1) / TILE; a++) {
        int Acol = a * TILE + threadIdx.x;
        int Brow = a * TILE + threadIdx.y;
        if (j < M && Acol < K) {
            sharedA[threadIdx.y][threadIdx.x] = A[j * K + Acol];
        } else {
            sharedA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (Brow < K && i < N) {
            sharedB[threadIdx.y][threadIdx.x] = B[Brow * N + i];
        } else {
            sharedB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        for (int b = 0; b < TILE; b++) {
            sum += sharedA[threadIdx.y][b] * sharedB[b][threadIdx.x];
        }
        __syncthreads();
    }
    if (j < M && i < N) {
        C[j * N + i] = sum;
    }
}

void gemm_gpu_o2(float* A, float* B, float* C, int M, int N, int K) {
    const int tilesize = 8;
    dim3 blockSize(tilesize, tilesize);
    dim3 gridSize((N + tilesize - 1) / tilesize, (M + tilesize - 1) / tilesize);
    gemm_gpu_o2_o3_kernel<8><<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

void gemm_gpu_o3_8(float* A, float* B, float* C, int M, int N, int K) {
    const int tilesize = 8;
    dim3 blockSize(tilesize, tilesize);
    dim3 gridSize((N + tilesize - 1) / tilesize, (M + tilesize - 1) / tilesize);
    gemm_gpu_o2_o3_kernel<8><<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

void gemm_gpu_o3_16(float* A, float* B, float* C, int M, int N, int K) {
    const int tilesize = 16;
    dim3 blockSize(tilesize, tilesize);
    dim3 gridSize((N + tilesize - 1) / tilesize, (M + tilesize - 1) / tilesize);
    gemm_gpu_o2_o3_kernel<16><<<gridSize, blockSize>>>(A, B, C, M, N, K);
}

void gemm_gpu_o3_32(float* A, float* B, float* C, int M, int N, int K) {
    const int tilesize = 32;
    dim3 blockSize(tilesize, tilesize);
    dim3 gridSize((N + tilesize - 1) / tilesize, (M + tilesize - 1) / tilesize);
    gemm_gpu_o2_o3_kernel<32><<<gridSize, blockSize>>>(A, B, C, M, N, K);
}


void gemm_gpu_o3_64(float* A, float* B, float* C, int M, int N, int K) {
    const int tilesize = 64;
    dim3 blockSize(tilesize, tilesize);
    dim3 gridSize((N + tilesize - 1) / tilesize, (M + tilesize - 1) / tilesize);
    gemm_gpu_o2_o3_kernel<64><<<gridSize, blockSize>>>(A, B, C, M, N, K);
}


// Modified gemm_gpu_cublas function
void gemm_gpu_cublas(float* d_A, float* d_B, float* d_C, int M, int N, int K) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float alpha = 1.0f;
    float beta = 0.0f;
    // cuBLAS uses column-major order, so we compute C^T = B^T * A^T to get row-major C
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    CUBLAS_CHECK(cublasDestroy(handle));
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: mp1 <M> <N> <K>" << std::endl;
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    float* A = new float[M * K]();
    float* B = new float[K * N]();
    float* C = new float[M * N]();

    fillRandom(A, M * K);
    fillRandom(B, K * N);

    // Check implementations
    auto ref = Ref();
    float* refC = new float[Ref::M * Ref::N]();
    CHECK(gemm_gpu_o0)
    CHECK(gemm_gpu_o1)
    CHECK(gemm_gpu_o2)
    CHECK(gemm_gpu_o3_8)
    CHECK(gemm_gpu_o3_16)
    CHECK(gemm_gpu_o3_32)
    CHECK(gemm_gpu_o3_64)
    CHECK(gemm_gpu_cublas)

    // Time implementations
    TIME(gemm_gpu_o0)
    TIME(gemm_gpu_o1)
    TIME(gemm_gpu_o2)
    TIME(gemm_gpu_o3_8)
    TIME(gemm_gpu_o3_16)
    TIME(gemm_gpu_o3_32)
    TIME(gemm_gpu_o3_64)
    TIME(gemm_gpu_cublas)

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] refC;

    return 0;
}