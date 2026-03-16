// 03-memory-model/shared_histogram.cu
// ─────────────────────────────────────
// Count occurrences of values in [0, BINS) using shared memory per block,
// then merge per-block histograms into a global result.
//
// Concepts covered:
//   - __shared__ array declaration
//   - __syncthreads() — mandatory barrier before reading data written by other threads
//   - atomicAdd() in shared memory (fast) vs global memory (slow)
//   - Two-phase pattern: accumulate locally in shared, then flush to global
//
// Why shared memory here:
//   A naive histogram does one atomicAdd to global memory per element.
//   With many threads hitting the same bin, global atomics serialize badly.
//   Using shared memory per block: contention is limited to blockDim.x threads,
//   and atomicAdd on shared mem is ~10× faster than on global.
//
// Pattern:
//   Phase 1 — each block initialises its shared histogram to zero
//             (all threads cooperatively zero it, then __syncthreads)
//   Phase 2 — each thread atomically increments the shared bin for its element
//             (then __syncthreads)
//   Phase 3 — each thread contributes one bin from shared → global with atomicAdd
//
// Exercise:
//   Input: array of N random integers in [0, BINS)
//   Output: histogram[b] = count of elements equal to b
//   Verify: sum of histogram == N
//
// Build:  cmake --build build --target shared_histogram
// Run:    ./build/03-memory-model/shared_histogram

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "../common/cuda_utils.h"

#define BINS 256

// TODO: implement histogram kernel using shared memory
// Signature: __global__ void histogram(const int* data, int* hist, int n)

int main() {
    const int N = 1 << 22;

    // TODO: generate random input in [0, BINS), run kernel, verify sum == N

    return 0;
}
