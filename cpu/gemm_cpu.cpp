#include <chrono>
#include "../include/utils.h"

#define NUM_RUNS 2

#define CHECK(name)                                           \
  std::cout << "checking " << #name << std::endl;             \
  initialize(refC, Ref::M *Ref::N);                           \
  name(ref.A, ref.B, refC, Ref::M, Ref::N, Ref::K);           \
  if (!ref.checkRef(refC))                                    \
  {                                                           \
    std::cerr << #name << ": check ref failed!" << std::endl; \
  };

#define TIME(name)                                                                           \
  for (int i = 0; i < 1; i++)                                                                \
  {                                                                                          \
    name(A, B, C, M, N, K);                                                                  \
  }                                                                                          \
  std::chrono::duration<double, std::milli> time_##name(0);                                  \
  for (int i = 0; i < NUM_RUNS; i++)                                                         \
  {                                                                                          \
    initialize(C, M * N);                                                                    \
    auto start_time_##name = std::chrono::high_resolution_clock::now();                      \
    name(A, B, C, M, N, K);                                                                  \
    auto end_time_##name = std::chrono::high_resolution_clock::now();                        \
    time_##name += end_time_##name - start_time_##name;                                      \
  }                                                                                          \
  std::chrono::duration<double, std::milli> duration_##name = time_##name / float(NUM_RUNS); \
  std::cout << "Time taken for GEMM (CPU," << #name << "): " << duration_##name.count() << "ms" << std::endl;

// reference CPU implementation of the GEMM kernel
// note that this implementation is naive and will run for longer for larger
// graphs
void gemm_cpu_o0(float *A, float *B, float *C, int M, int N, int K)
{
  return;
  for (int j = 0; j < N; j++)
  {
    for (int i = 0; i < M; i++)
    {
      for (int k = 0; k < K; k++)
      {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

// Your optimized implementations go here
// note that for o4 you don't have to change the code, but just the compiler flags. So, you can use o3's code for that part
void gemm_cpu_o1(float *A, float *B, float *C, int M, int N, int K)
{
  return;
  for (int i = 0; i < M; i++)
  {
    for (int k = 0; k < K; k++)
    {
      for (int j = 0; j < N; j++)
      {
        C[i * N + j] += A[i * K + k] * B[k * N + j];
      }
    }
  }
}

// 640 KB L1 Cache
void gemm_cpu_o2(float *A, float *B, float *C, int M, int N, int K)
{
  return;
    const int STRIDE = 32;
    for (int i = 0; i < M; i += STRIDE)
    {
        for (int k = 0; k < K; k += STRIDE)
        {
            for (int j = 0; j < N; j += STRIDE)
            {
                for (int ii = i; ii < i + STRIDE; ii++)
                {
                    if (ii >= M) break;
                    for (int kk = k; kk < k + STRIDE; kk++)
                    {
                        if (kk >= K) break;
                        float a_ik = A[ii * K + kk];
                        for (int jj = j; jj < j + STRIDE; jj++)
                        {
                            if (jj >= N) break;
                            C[ii * N + jj] += a_ik * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
}

void gemm_cpu_o3(float *A, float *B, float *C, int M, int N, int K)
{
    const int STRIDE = 32;
    
    #pragma omp parallel for
    for (int i = 0; i < M; i += STRIDE)
    {
        for (int k = 0; k < K; k += STRIDE)
        {
            for (int j = 0; j < N; j += STRIDE)
            {
                // Get boundaries manually instea dof if statement to satisfy compiler requirements
                int i_end = (i + STRIDE < M) ? i + STRIDE : M;
                int k_end = (k + STRIDE < K) ? k + STRIDE : K;
                int j_end = (j + STRIDE < N) ? j + STRIDE : N;
                
                for (int ii = i; ii < i_end; ii++)
                {
                    for (int kk = k; kk < k_end; kk++)
                    {
                        float a_ik = A[ii * K + kk];
                        #pragma omp simd
                        for (int jj = j; jj < j_end; jj++)
                        {
                            C[ii * N + jj] += a_ik * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
  if (argc < 3)
  {
    std::cout << "Usage: mp1 <M> <N> <K>" << std::endl;
    return 1;
  }

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);

  float *A = new float[M * K]();
  float *B = new float[K * N]();
  float *C = new float[M * N]();

  fillRandom(A, M * K);
  fillRandom(B, K * N);

  // Check if the kernel results are correct
  // note that even if the correctness check fails all optimized kernels will run.
  // We are not exiting the program at failure at this point.
  // It is a good idea to add more correctness checks to your code.
  // We may (at discretion) verify that your code is correct.
  float *refC = new float[Ref::M * Ref::N]();
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
