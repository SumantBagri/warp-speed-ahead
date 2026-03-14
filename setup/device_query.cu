// setup/device_query.cu
// ─────────────────────
// Query and print properties of all available CUDA devices.
// This is always the first thing to run — it gives you the key hardware
// numbers you'll reason about throughout this project:
//   - SM count, CUDA cores per SM, total CUDA cores
//   - Peak memory bandwidth (GB/s)
//   - Shared memory per SM / per block
//   - Register file size per SM
//   - Max threads per block, max warps per SM
//   - Compute capability (determines supported features)
//   - L2 cache size
//
// Reference: cudaDeviceProp struct
//   https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
//
// Build:  cmake --build build --target device_query
// Run:    ./build/setup/device_query

#include <cstdio>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"

// Ampere SM → CUDA core mapping (fill in other arches as you learn them)
static int cores_per_sm(int major, int minor) {
    // TODO: return core count based on major.minor
    // Hints:
    //   SM 8.6 (RTX 30xx / Ampere GA10x) = 128 cores/SM
    //   SM 8.0 (A100    / Ampere GA100)  =  64 cores/SM
    //   SM 7.5 (Turing)                  =  64 cores/SM
    //   SM 7.0 (Volta)                   =  64 cores/SM
    return -1;
}

int main() {
    // TODO: get device count with cudaGetDeviceCount
    // TODO: loop over devices, call cudaGetDeviceProperties for each
    // TODO: print the fields listed in the header comment above
    // TODO: compute peak bandwidth = 2 × (memoryBusWidth/8) × memoryClockRate
    //       (factor of 2 because DDR = Double Data Rate)
    return 0;
}
