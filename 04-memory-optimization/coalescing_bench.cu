// 04-memory-optimization/coalescing_bench.cu
// ─────────────────────────────────────────────
// Benchmark coalesced vs strided global memory access patterns.
//
// Concepts covered:
//   - Memory coalescing: the GPU memory controller combines warp accesses into
//     the fewest possible 128-byte cache-line transactions
//   - Stride 1 (consecutive): entire warp fits in 1–4 transactions → fast
//   - Stride S: each thread accesses a different cache line → S transactions → slow
//   - The hardware metric "sectors per request" captures this:
//       1.0 = perfect coalescing, 32.0 = fully uncoalesced (stride 32)
//
// Access patterns to benchmark:
//   stride_1:  a[idx]          — coalesced, 1 transaction/warp
//   stride_2:  a[idx * 2]      — 2 transactions/warp
//   stride_4:  a[idx * 4]      — 4 transactions/warp
//   stride_32: a[idx * 32]     — 32 transactions/warp (worst case)
//   random:    a[perm[idx]]    — random permutation (also worst case, different pattern)
//
// Exercise:
//   Write a kernel template or separate kernels for each stride.
//   Each kernel just reads from input[idx * stride] and writes to output[idx].
//   Time all variants, print bandwidth (GB/s) for each.
//   The coalesced version should approach the RTX 3070 peak of ~448 GB/s.
//
// Build:  cmake --build build --target coalescing_bench
// Run:    ./build/04-memory-optimization/coalescing_bench

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "../common/cuda_utils.h"

// TODO: write kernels for each stride (or a templated kernel with stride as template param)
// TODO: write random-access kernel using a precomputed permutation array

int main() {
    const int N = 1 << 25;

    // TODO: benchmark all patterns, print stride vs bandwidth table

    return 0;
}
