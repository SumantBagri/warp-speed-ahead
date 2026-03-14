// 06-reductions/reduce_optimized.cu
// ────────────────────────────────────
// Parallel reduction — sequential addressing, no divergence.
//
// Concepts covered:
//   - Sequential addressing: iterate s from blockDim.x/2 down to 1
//     → active threads are always 0..s-1, i.e., the lower half of the block
//     → no warp divergence, no idle threads within an active warp
//   - First-add-during-load: each thread loads two elements and adds them
//     before entering the reduction tree, halving the number of iterations
//
// Sequential addressing pattern:
//   for s = blockDim.x/2; s > 0; s >>= 1:
//     if (tid < s):          ← first s threads are always a contiguous group
//       smem[tid] += smem[tid + s]
//     __syncthreads()
//
// First-add-during-load:
//   Instead of loading smem[tid] = data[gid] and smem[tid + blockDim.x] = data[gid + blockDim.x]
//   in separate steps, do: smem[tid] = data[gid] + data[gid + gridStride]
//   This halves the shared memory needed and doubles work per thread.
//
// Exercise:
//   Implement and compare timing against reduce_naive.
//   Expected improvement: 2–4× depending on N and block size.
//
// Build:  cmake --build build --target reduce_optimized
// Run:    ./build/06-reductions/reduce_optimized

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"

// TODO: implement reduce_sequential kernel (no divergence version)
// TODO: add first-add-during-load optimisation

int main() {
    const int N = 1 << 24;

    // TODO: run, verify, print time vs naive

    return 0;
}
