// 03-memory-model/constant_filter.cu
// ─────────────────────────────────────
// Apply a convolution filter stored in constant memory.
//
// Concepts covered:
//   - __constant__ declaration (file-scope, device-visible, read-only)
//   - cudaMemcpyToSymbol — the correct way to populate constant memory
//   - When constant memory excels: all threads read the same address (broadcast)
//   - When it's bad: non-uniform access across a warp (serialised, not cached)
//
// Constant memory characteristics:
//   - 64 KB total, cached in a dedicated constant cache (per-SM)
//   - When all threads in a warp read the same address → single broadcast (fast)
//   - When threads read different addresses → requests are serialised (slow)
//   - Best use case: filter kernels, look-up tables, parameters read uniformly
//
// Exercise:
//   Implement a 1D convolution where the filter coefficients live in
//   __constant__ float filter[FILTER_SIZE].
//   Compare performance vs passing the filter as a regular pointer argument
//   (which goes through L1/L2 global cache, not the constant cache).
//
// Build:  cmake --build build --target constant_filter
// Run:    ./build/03-memory-model/constant_filter

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "../common/cuda_utils.h"

#define FILTER_SIZE 17

// TODO: declare the filter in constant memory
// Hint: __constant__ float filter[FILTER_SIZE];  (file scope, outside any function)

// TODO: implement the convolution kernel using the constant-memory filter
// TODO: implement a second version that takes filter as a const float* argument
// Compare their timings.

int main() {
    const int N = 1 << 22;

    // TODO: populate filter with cudaMemcpyToSymbol
    // TODO: run both versions, compare timing

    return 0;
}
