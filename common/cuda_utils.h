#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ─── Error checking ───────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d  —  %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─── Timing via CUDA events ───────────────────────────────────────────────────

struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~CudaTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }

    void begin(cudaStream_t s = 0) { cudaEventRecord(start, s); }
    float end(cudaStream_t s = 0) {
        cudaEventRecord(stop, s);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// ─── Device info helpers ──────────────────────────────────────────────────────

inline void print_device_info(int device = 0) {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, device);
    printf("Device %d: %s\n", device, p.name);
    printf("  SM count       : %d\n", p.multiProcessorCount);
    printf("  CUDA cores/SM  : (see arch table)\n");
    printf("  SM clock (MHz) : %d\n", p.clockRate / 1000);
    printf("  Mem clock (MHz): %d\n", p.memoryClockRate / 1000);
    printf("  Mem bus width  : %d-bit\n", p.memoryBusWidth);
    printf("  Peak BW (GB/s) : %.1f\n",
           2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1e6);
    printf("  Global mem     : %.0f MB\n", p.totalGlobalMem / 1e6);
    printf("  Shared mem/SM  : %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
    printf("  Registers/SM   : %d\n", p.regsPerMultiprocessor);
    printf("  Warp size      : %d\n", p.warpSize);
    printf("  Max threads/blk: %d\n", p.maxThreadsPerBlock);
    printf("  Compute cap.   : %d.%d\n", p.major, p.minor);
    printf("  L2 cache (MB)  : %.1f\n", p.l2CacheSize / 1e6);
}

// ─── Simple correctness check ─────────────────────────────────────────────────

inline bool arrays_equal(const float* a, const float* b, int n, float tol = 1e-4f) {
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > tol) {
            fprintf(stderr, "Mismatch at index %d: %.6f vs %.6f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}
