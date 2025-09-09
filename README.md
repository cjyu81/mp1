# mp1
Machine Project 1
Charlie Jyu 9/4/25

Commands (from debug):
cd C:\Users\charl\Documents\GitHub\mp1\build
cmake --build .
cd Debug
.\mp1_cpu.exe 5000 5000 5000

Results
PS C:\Users\charl\Documents\GitHub\mp1\build\Debug> .\mp1_cpu.exe 5000 5000 5000
checking gemm_cpu_o0
checking gemm_cpu_o1
checking gemm_cpu_o2
checking gemm_cpu_o3
Time taken for GEMM (JIK): 768286ms
Time taken for GEMM (IJK): 1.31334e+06ms
Time taken for GEMM (IKJ): 1.01319e+06ms
Time taken for GEMM (KIJ): 725338ms
Time taken for GEMM (JKI): 1.59646e+06ms
Time taken for GEMM (KJI): 2.53341e+06ms

Timing all loop orderings...
Time taken for GEMM (CPU,gemm_cpu_jik): 656420ms
Time taken for GEMM (CPU,gemm_cpu_jki): 1.47879e+06ms
Time taken for GEMM (CPU,gemm_cpu_kji): 1.4636e+06ms
Time taken for GEMM (CPU,gemm_cpu_kij): 476490ms
Time taken for GEMM (CPU,gemm_cpu_ikj): 436473ms
Time taken for GEMM (CPU,gemm_cpu_ijk): 905760ms


Timing all loop orderings...
Time taken for GEMM (CPU,gemm_cpu_jik): 1.21744e+06ms
Time taken for GEMM (CPU,gemm_cpu_jki): 2.67237e+06ms
Time taken for GEMM (CPU,gemm_cpu_kji): 1.54729e+06ms
Time taken for GEMM (CPU,gemm_cpu_kij): 360055ms
Time taken for GEMM (CPU,gemm_cpu_ikj): 380022ms
Time taken for GEMM (CPU,gemm_cpu_ijk): 924629ms