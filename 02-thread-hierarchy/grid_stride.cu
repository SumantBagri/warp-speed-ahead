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

#include "../common/cuda_utils.h"

// TODO: implement vector_add_stride using a grid-stride loop

int main() {
    const int N = 1 << 24;
    const size_t bytes = N * sizeof(float);

    // TODO: standard host/device lifecycle
    // Launch with fixed <<<1024, 256>>> regardless of N
    // Verify and print timing

    return 0;
}
