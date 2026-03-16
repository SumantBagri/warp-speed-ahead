// 07-warp-programming/warp_vote.cu
// ──────────────────────────────────
// Use warp vote intrinsics to make collective decisions without shared memory.
//
// Concepts covered:
//   - __all_sync(mask, pred)    — true iff ALL active lanes have pred != 0
//   - __any_sync(mask, pred)    — true iff ANY active lane has pred != 0
//   - __ballot_sync(mask, pred) — returns 32-bit bitmask of lanes where pred != 0
//   - __popc(x)                 — population count (number of set bits) — CPU intrinsic
//   - These are warp-synchronous: no __syncthreads needed
//
// Exercises:
//
// A — count elements above threshold:
//   Each thread checks if its value > threshold.
//   Use __ballot_sync to get a bitmask, then __popc to count.
//   Compare to atomicAdd-based counting (hint: vote is much faster).
//
// B — early exit optimisation:
//   In a loop over blocks of data, use __any_sync to check if any thread
//   in the warp still has work to do. If not, break early.
//   This avoids branches inside the loop body.
//
// C — warp-uniform branch:
//   Use __all_sync to take a fast path when all threads in a warp agree
//   on a condition (no divergence on that branch).
//
// Build:  cmake --build build --target warp_vote
// Run:    ./build/07-warp-programming/warp_vote

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "../common/cuda_utils.h"

// TODO: implement kernels for exercises A, B, C

int main() {
    // TODO: run each exercise, verify and print results

    return 0;
}
