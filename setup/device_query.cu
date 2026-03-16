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

#include <cuda_runtime.h>

#include <cstdio>

#include "../common/cuda_utils.h"

static int cores_per_sm(int major, int minor) {
    switch (major) {
        case 9:
            switch (minor) {
                case 0:
                    return 128;
            }
            break;
        case 8:
            switch (minor) {
                case 0:
                    return 64;
                case 6:
                    return 128;
            }
            break;
        case 7:
            switch (minor) {
                case 0:
                    return 64;
                case 5:
                    return 64;
            }
            break;
        case 6:
            switch (minor) {
                case 0:
                    return 64;
                case 1:
                    return 128;
            }
            break;
    }
    return -1;
}

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("Found %d CUDA device(s)\n\n", device_count);

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, i));

        int cores = cores_per_sm(p.major, p.minor);
        int total_cores = (cores > 0) ? cores * p.multiProcessorCount : -1;
        double peak_bw = 2.0 * (p.memoryBusWidth / 8.0) * p.memoryClockRate * 1e3 / 1e9;

        printf("========================================\n");
        printf("Device %d:                    %s\n", i, p.name);
        printf("========================================\n");
        printf("  Compute capability:        %d.%d\n", p.major, p.minor);
        printf("  SM count:                  %d\n", p.multiProcessorCount);
        printf("  CUDA cores per SM:         %d\n", cores);
        printf("  Total CUDA cores:          %d\n", total_cores);
        printf("  SM clock:                  %.2f GHz\n", p.clockRate / 1e6);
        printf("  Total global memory:       %.2f GB\n", p.totalGlobalMem / 1e9);
        printf("  Memory clock:              %.2f GHz\n", p.memoryClockRate / 1e6);
        printf("  Memory bus width:          %d bits\n", p.memoryBusWidth);
        printf("  Peak memory bandwidth:     %.2f GB/s\n", peak_bw);
        printf("  Shared memory per SM:      %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
        printf("  Shared memory per block:   %zu KB\n", p.sharedMemPerBlock / 1024);
        printf("  Registers per SM:          %d\n", p.regsPerMultiprocessor);
        printf("  Registers per block:       %d\n", p.regsPerBlock);
        printf("  L2 cache size:             %d KB\n", p.l2CacheSize / 1024);
        printf("  Max threads per block:     %d\n", p.maxThreadsPerBlock);
        printf("  Max warps per SM:          %d\n", p.maxThreadsPerMultiProcessor / 32);
        printf("  Warp size:                 %d\n", p.warpSize);
        printf("  Max grid dimensions:       [%d, %d, %d]\n", p.maxGridSize[0], p.maxGridSize[1],
               p.maxGridSize[2]);
        printf("  Max block dimensions:      [%d, %d, %d]\n", p.maxThreadsDim[0],
               p.maxThreadsDim[1], p.maxThreadsDim[2]);
        printf("  Unified memory support:    %s\n", p.unifiedAddressing ? "Yes" : "No");
        printf("  Concurrent kernels:        %s\n", p.concurrentKernels ? "Yes" : "No");
        printf("  Async engine count:        %d\n", p.asyncEngineCount);
        printf("\n");
    }

    return 0;
}
