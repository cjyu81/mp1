# mp1
Machine Project 1
Charlie Jyu 9/4/25

Commands (from debug):
cd C:\Users\charl\Documents\GitHub\mp1\build
cmake --build .
cd Debug
.\mp1_cpu.exe 5000 5000 5000


Commands to clean build(from mp1):
rm -r -fo build
mkdir build
cd build
cmake .. -DCPU_ONLY=ON
cmake --build .
cd Debug
.\mp1_cpu.exe 5000 5000 5000


Build with FMA:
cd ..
rm -r -fo build
mkdir build
cd build
cmake .. -DCPU_ONLY=ON -DUSE_FMA=ON
cmake --build . --config Release
.\Release\mp1_cpu.exe 500 500 500


Build without FMA:
cd ..
rm -r -fo build
mkdir build
cd build
cmake .. -DCPU_ONLY=ON -DUSE_FMA=OFF
cmake --build . --config Release
.\Release\mp1_cpu.exe 100 100 100
.\Release\mp1_cpu.exe 1000 1000 1000

