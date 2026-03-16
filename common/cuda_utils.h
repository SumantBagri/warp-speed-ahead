#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

// ─── Error checking ───────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d  —  %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ─── Timing via CUDA events ───────────────────────────────────────────────────

struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void begin(cudaStream_t s = 0) { cudaEventRecord(start, s); }
    float end(cudaStream_t s = 0) {
        cudaEventRecord(stop, s);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

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
