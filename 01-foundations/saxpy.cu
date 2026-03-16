// 01-foundations/saxpy.cu
// ───────────────────────
// SAXPY: y[i] = a * x[i] + y[i]   (Single-precision A·X Plus Y)
// Classic BLAS level-1 operation.
//
// Concepts covered:
//   - Same full host/device lifecycle as vector_add (deliberate repetition)
//   - Scalar kernel argument: `a` is passed by value directly to the kernel
//     (no need to cudaMalloc a scalar — it fits in a register)
//   - In-place output: y is both read and written by the kernel
//
// After vector_add, you should need no hints here.
// The structure is identical; only the arithmetic changes.
//
// Build:  cmake --build build --target saxpy
// Run:    ./build/01-foundations/saxpy

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../common/cuda_utils.h"

// TODO: write the saxpy kernel
__global__ void saxpy(float a, const float* x, float* y, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        y[tid] = a * x[tid] + y[tid];
    }
}

int main() {
    const int N = 1 << 24;
    const size_t bytes = N * sizeof(float);
    const float A = 2.5f;

    // TODO: full host/device lifecycle (same pattern as vector_add)
    // Reference result: h_ref[i] = A * h_x[i] + h_y[i]

    std::vector<float> h_x(N, 0.0f);
    std::vector<float> h_y(N, 0.0f);
    std::vector<float> h_ref(N, 0.0f);

    for (int i = 0; i < N; i++) {
        h_x[i] = i;
        h_y[i] = i;
        h_ref[i] = std::fmaf(A, h_x[i], h_y[i]);
    }

    float* d_x;
    float* d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

    int blockDim = 256;
    int gridDim = (N + blockDim - 1) / blockDim;
    // warmup
    saxpy<<<gridDim, blockDim>>>(A, d_x, d_y, N);
    CUDA_CHECK(cudaGetLastError());

    // reset d_y before the timed run
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

    CudaTimer timer;
    timer.begin();
    saxpy<<<gridDim, blockDim>>>(A, d_x, d_y, N);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.end();

    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

    bool is_equal = arrays_equal(h_ref.data(), h_y.data(), N);
    printf("[%s] SAXPY - N=%d\n", is_equal ? "PASSED" : "FAILED", N);

    int num_mem_ops = 3;
    float bw = ((float)num_mem_ops * N * sizeof(float)) / (ms * 1e-3 * 1e9);
    printf("Time: %.3f ms | Bandwidth: %.1f GB/s", ms, bw);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return 0;
}
