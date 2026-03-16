// 06-reductions/reduce_warp.cu
// ──────────────────────────────
// Warp-level reduction using shuffle intrinsics — eliminates __syncthreads
// for the final 5 reduction steps.
//
// Concepts covered:
//   - __shfl_down_sync: thread at lane L receives the value from lane L+delta
//     (lanes beyond 31 are clamped / inactive)
//   - The mask argument (0xffffffff) specifies which lanes participate
//   - Within a warp, shuffle is synchronous — no barrier needed
//   - Combining block-level shared memory reduction (for warps > 1)
//     with warp-level shuffle (for the final within-warp step)
//
// Warp reduce pattern (reduces 32 values to 1 in lane 0):
//   val += __shfl_down_sync(0xffffffff, val, 16);
//   val += __shfl_down_sync(0xffffffff, val, 8);
//   val += __shfl_down_sync(0xffffffff, val, 4);
//   val += __shfl_down_sync(0xffffffff, val, 2);
//   val += __shfl_down_sync(0xffffffff, val, 1);
//   // lane 0 now holds sum of all 32 lanes
//
// Full block reduce strategy:
//   1. Each thread accumulates a private sum from global memory (grid-stride)
//   2. Each warp reduces its 32 private sums → warp sum in lane 0
//   3. Warp sums are written to shared memory (one entry per warp)
//   4. First warp reads all warp sums and does a final shuffle reduce
//
// Build:  cmake --build build --target reduce_warp
// Run:    ./build/06-reductions/reduce_warp

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "../common/cuda_utils.h"

// TODO: implement warp_reduce_sum(__shfl_down_sync chain) as a __device__ function
// TODO: implement full block reduction using warp_reduce_sum + shared memory staging
// Compare timing against reduce_optimized

int main() {
    const int N = 1 << 24;

    // TODO: run, verify, print time

    return 0;
}
