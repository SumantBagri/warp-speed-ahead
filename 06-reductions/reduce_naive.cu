// 06-reductions/reduce_naive.cu
// ───────────────────────────────
// Parallel sum reduction — naive version with warp divergence.
//
// Concepts covered:
//   - Parallel reduction tree: O(log N) steps, each step halves active threads
//   - Shared memory accumulation within a block
//   - Multi-block reduction: each block produces one partial sum,
//     then a second kernel pass (or atomicAdd) combines them
//   - The divergence problem: if (tid % (2*s) == 0) → threads within a warp
//     diverge increasingly as s grows, serialising the warp
//
// Naive reduction pattern (divergent):
//   for s = 1, 2, 4, ..., blockDim.x/2:
//     if (tid % (2*s) == 0):   ← only 1/2, 1/4, ... of threads active
//       smem[tid] += smem[tid + s]
//     __syncthreads()
//
// This works but is inefficient:
//   - At step s=1:  half the warp is idle (divergence)
//   - At step s=16: 31 of 32 threads in a warp are idle
//   - Also: non-contiguous active threads → poor memory access pattern
//
// Exercise:
//   Implement and measure. Print time and effective bandwidth.
//   This is the baseline you'll beat in reduce_optimized.cu.
//
// Build:  cmake --build build --target reduce_naive
// Run:    ./build/06-reductions/reduce_naive

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"

// TODO: implement reduce_naive kernel (one block → one partial sum in result[blockIdx.x])
// TODO: implement a second-pass kernel or use atomicAdd to produce the final scalar
// Verify: GPU result == CPU sum (use double on CPU to avoid float precision issues)

int main() {
    const int N = 1 << 24;

    // TODO: run, verify, print time

    return 0;
}
