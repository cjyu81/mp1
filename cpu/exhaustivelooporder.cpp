// Generated with AI
#include <chrono>
#include <algorithm>
#include "../include/utils.h"

#define NUM_RUNS 2

#define CHECK(name) \
  std::cout << "checking " << #name << std::endl;		\
  initialize(refC, Ref::M * Ref::N);				\
  name(ref.A, ref.B, refC, Ref::M, Ref::N, Ref::K);		\
  if (!ref.checkRef(refC)){					\
    std::cerr << #name << ": check ref failed!" << std::endl;	\
  };								
  
#define TIME(name) \
  for (int i = 0; i < 1; i++) { \
      name(A, B, C, M, N, K); \
  } \
  std::chrono::duration<double, std::milli> time_ ## name(0); \
  for (int i = 0; i < NUM_RUNS; i++) { \
      initialize(C, M * N); \
      auto start_time_ ## name = std::chrono::high_resolution_clock::now(); \
      name(A, B, C, M, N, K); \
      auto end_time_ ## name = std::chrono::high_resolution_clock::now(); \
      time_ ## name += end_time_ ## name - start_time_ ## name; \
  } \
  std::chrono::duration<double, std::milli> duration_ ## name = time_ ## name / float(NUM_RUNS); \
  std::cout << "Time taken for GEMM (CPU," << #name << "): " << duration_ ## name.count() << "ms" << std::endl;

// Loop order: j-i-k (original naive implementation)
void gemm_cpu_jik(float* A, float* B, float *C, int M, int N, int K) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      for (int k = 0; k < K; k++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

// Loop order: j-k-i
void gemm_cpu_jki(float* A, float* B, float *C, int M, int N, int K) {
  for (int j = 0; j < N; j++) {
    for (int k = 0; k < K; k++) {
      for (int i = 0; i < M; i++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

// Loop order: k-j-i
void gemm_cpu_kji(float* A, float* B, float *C, int M, int N, int K) {
  for (int k = 0; k < K; k++) {
    for (int j = 0; j < N; j++) {
      for (int i = 0; i < M; i++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

// Loop order: k-i-j
void gemm_cpu_kij(float* A, float* B, float *C, int M, int N, int K) {
  for (int k = 0; k < K; k++) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

// Loop order: i-k-j
void gemm_cpu_ikj(float* A, float* B, float *C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

// Loop order: i-j-k
void gemm_cpu_ijk(float* A, float* B, float *C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

// Method to test all loop orderings
void test_loop_orders(float* A, float* B, float* C, int M, int N, int K) {
  float* refC = new float[Ref::M * Ref::N]();
  auto ref = Ref();

  // Check correctness for each loop ordering
  std::cout << "Checking correctness of all loop orderings..." << std::endl;
  CHECK(gemm_cpu_jik)
  CHECK(gemm_cpu_jki)
  CHECK(gemm_cpu_kji)
  CHECK(gemm_cpu_kij)
  CHECK(gemm_cpu_ikj)
  CHECK(gemm_cpu_ijk)
  delete[] refC;

  // Time each loop ordering
  std::cout << "\nTiming all loop orderings..." << std::endl;
  TIME(gemm_cpu_jik)
  TIME(gemm_cpu_jki)
  TIME(gemm_cpu_kji)
  TIME(gemm_cpu_kij)
  TIME(gemm_cpu_ikj)
  TIME(gemm_cpu_ijk)
}

// Original functions (kept for compatibility with main)
void gemm_cpu_o0(float* A, float* B, float *C, int M, int N, int K) {
  gemm_cpu_jik(A, B, C, M, N, K); // Use j-i-k as the reference
}

void gemm_cpu_o1(float* A, float* B, float *C, int M, int N, int K) {
  gemm_cpu_jki(A, B, C, M, N, K); // Use j-k-i as provided
}

void gemm_cpu_o2(float* A, float* B, float *C, int M, int N, int K) {
  gemm_cpu_kji(A, B, C, M, N, K); // Use k-j-i as provided
}

void gemm_cpu_o3(float* A, float* B, float *C, int M, int N, int K) {
  gemm_cpu_kji(A, B, C, M, N, K); // Reuse k-j-i (placeholder for o3)
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

  // Test all loop orderings
  test_loop_orders(A, B, C, M, N, K);

  // Optionally, test original functions
  std::cout << "\nTesting original functions..." << std::endl;
  float* refC = new float[Ref::M * Ref::N]();
  auto ref = Ref();
  CHECK(gemm_cpu_o0)
  CHECK(gemm_cpu_o1)
  CHECK(gemm_cpu_o2)
  CHECK(gemm_cpu_o3)
  delete[] refC;

  TIME(gemm_cpu_o0)
  TIME(gemm_cpu_o1)
  TIME(gemm_cpu_o2)
  TIME(gemm_cpu_o3)

  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}