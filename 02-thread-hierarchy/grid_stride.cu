// 02-thread-hierarchy/grid_stride.cu
// ─────────────────────────────────────
// The grid-stride loop pattern — preferred over one-thread-per-element for
// large arrays and better GPU utilisation.
//
// Concepts covered:
//   - Grid stride: total threads in grid = gridDim.x * blockDim.x
//   - Each thread processes multiple elements spaced `stride` apart
//   - Decouples problem size N from launch config (you can launch 1024 blocks
//     regardless of N, and threads loop to cover the rest)
//   - Better cache reuse and occupancy than a single-pass kernel on huge arrays
//   - Required pattern for __device__ functions that must work at any N
//
// Pattern:
//   int stride = gridDim.x * blockDim.x;
//   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride)
//       output[i] = f(input[i]);
//
// Exercise:
//   Re-implement vector_add using a grid-stride loop.
//   Launch with only 1024 blocks instead of (N + 255) / 256 blocks.
//   Verify correctness and compare kernel time vs the original.
//
// Build:  cmake --build build --target grid_stride
// Run:    ./build/02-thread-hierarchy/grid_stride

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../common/cuda_utils.h"

constexpr int GRID_DIM = 1024;
constexpr int BLOCK_DIM = 1024;

// TODO: implement vector_add_stride using a grid-stride loop
__global__ void vector_add_stride(const float* a, const float* b, float* c, int N) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1 << 24;
    const size_t bytes = N * sizeof(float);

    // TODO: standard host/device lifecycle
    // Launch with fixed <<<1024, 256>>> regardless of N
    // Verify and print timing

    std::vector<float> h_a(N, 0.0);
    std::vector<float> h_b(N, 0.0);
    std::vector<float> h_c(N, 0.0);
    std::vector<float> h_ref(N, 0.0);
    for (int i = 0; i < N; ++i) {
        h_a[i] = std::pow(std::sin(i), 2);
        h_b[i] = std::pow(std::cos(i), 2);
        h_ref[i] = h_a[i] + h_b[i];
    }

    float* d_a;
    float* d_b;
    float* d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // warmup
    vector_add_stride<<<GRID_DIM, BLOCK_DIM>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    CudaTimer timer;
    timer.begin();
    vector_add_stride<<<GRID_DIM, BLOCK_DIM>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.end();

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    bool is_equal = arrays_equal(h_ref.data(), h_c.data(), N);
    printf("[%s] Vector Strided Add - N=%d\n", is_equal ? "PASS" : "FAIL", N);

    float bandwidth = (3.0 * N * sizeof(float)) / (ms * 1e-3f * 1e9f);
    printf("Time: %.3f ms | Bandwidth: %.1f GB/s", ms, bandwidth);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
