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

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"

// TODO: write the vector_add kernel

int main() {
    const int N        = 1 << 24;   // 16M elements
    const size_t bytes = N * sizeof(float);

    // TODO: allocate host arrays h_a, h_b, h_c, h_ref
    // TODO: initialize h_a and h_b; compute h_ref = h_a + h_b on CPU

    // TODO: allocate device arrays d_a, d_b, d_c with CUDA_CHECK
    // TODO: copy h_a → d_a and h_b → d_b

    // TODO: choose blockSize and compute gridSize
    // TODO: start CudaTimer, launch kernel, stop timer

    // TODO: cudaGetLastError() + cudaDeviceSynchronize()
    // TODO: copy d_c → h_c
    // TODO: verify with arrays_equal(), print result
    // TODO: print kernel time and effective bandwidth
    // TODO: free all device and host memory

    return 0;
}
