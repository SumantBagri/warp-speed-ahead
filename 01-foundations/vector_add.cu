// 01-foundations/vector_add.cu
// ────────────────────────────
// The "hello world" of GPU programming: c[i] = a[i] + b[i]
//
// Concepts covered:
//   - __global__ kernel declaration
//   - Kernel launch syntax:  kernel<<<gridDim, blockDim>>>(args)
//   - Thread global index:   idx = blockIdx.x * blockDim.x + threadIdx.x
//   - Bounds check:          if (idx < n)
//   - cudaMalloc / cudaFree
//   - cudaMemcpy (HostToDevice, DeviceToHost)
//   - cudaDeviceSynchronize
//   - CUDA_CHECK error macro (see common/cuda_utils.h)
//   - CudaTimer for kernel timing (see common/cuda_utils.h)
//
// Key numbers to understand:
//   - blockSize = 256 is a common default (must be a multiple of 32)
//   - gridSize  = ceil(N / blockSize) — enough blocks to cover all elements
//   - Total threads launched = gridSize × blockSize  (may be > N, hence the guard)
//
// After the kernel runs, measure effective memory bandwidth:
//   bandwidth = (3 × N × 4 bytes) / kernel_time_seconds
//   (reads a, reads b, writes c → 3 arrays)
//   Compare against RTX 3070 peak: ~448 GB/s
//
// Build:  cmake --build build --target vector_add
// Run:    ./build/01-foundations/vector_add

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../common/cuda_utils.h"

// TODO: write the vector_add kernel
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    const int N = 1 << 24;  // 16M elements
    const size_t bytes = N * sizeof(float);

    // TODO: allocate host arrays h_a, h_b, h_c, h_ref
    std::vector<float> h_a(N, 0.0f);
    std::vector<float> h_b(N, 0.0f);
    std::vector<float> h_c(N, 0.0f);
    std::vector<float> h_ref(N, 0.0f);

    // TODO: initialize h_a and h_b; compute h_ref = h_a + h_b on CPU
    for (int i = 0; i < N; i++) {
        h_a[i] = std::pow(std::sin(i), 2);
        h_b[i] = std::pow(std::cos(i), 2);
        h_ref[i] = h_a[i] + h_b[i];
    }

    // TODO: allocate device arrays d_a, d_b, d_c with CUDA_CHECK
    float* d_a;
    float* d_b;
    float* d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // TODO: copy h_a → d_a and h_b → d_b
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // TODO: choose blockSize and compute gridSize
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // warmup
    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    // TODO: start CudaTimer, launch kernel, stop timer
    CudaTimer timer;
    timer.begin();
    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.end();  // synchronizes internally

    // TODO: copy d_c → h_c
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // TODO: verify with arrays_equal(), print result
    bool is_equal = arrays_equal(h_ref.data(), h_c.data(), N);
    printf("[%s] Vector Add - N=%d\n", is_equal ? "PASS" : "FAIL", N);

    // TODO: print kernel time and effective bandwidth
    float bandwidth = (3.0 * N * sizeof(float)) / (ms * 1e-3f * 1e9f);
    printf("Time: %.3f ms | Bandwidth: %.1f GB/s", ms, bandwidth);

    // TODO: free all device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
