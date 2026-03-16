// 07-warp-programming/shuffle_reduce.cu
// ─────────────────────────────────────────
// Explore all four warp shuffle intrinsics through focused micro-exercises.
//
// Concepts covered:
//   - __shfl_sync(mask, val, srcLane)       — broadcast: all lanes get srcLane's value
//   - __shfl_up_sync(mask, val, delta)      — lane L gets value from lane L-delta
//   - __shfl_down_sync(mask, val, delta)    — lane L gets value from lane L+delta
//   - __shfl_xor_sync(mask, val, laneMask)  — butterfly: lane L gets value from lane L^laneMask
//   - All shuffle ops are intra-warp and synchronous (no barrier needed)
//   - The mask argument must include all participating lanes (use 0xffffffff for full warp)
//
// Exercises (each in its own small kernel, launched with 1 block of 32 threads):
//
// A — broadcast:
//   Use __shfl_sync to broadcast lane 5's value to all lanes.
//   Every thread should print the same number.
//
// B — inclusive prefix sum using __shfl_up_sync:
//   for offset = 1, 2, 4, 8, 16:
//     if (laneId >= offset): val += __shfl_up_sync(0xffffffff, val, offset)
//   Lane L should hold sum of original values [0..L].
//
// C — segmented reduce using __shfl_xor_sync (butterfly):
//   for mask = 1, 2, 4, 8, 16:
//     val += __shfl_xor_sync(0xffffffff, val, mask)
//   All lanes should hold the warp sum after 5 steps.
//
// Build:  cmake --build build --target shuffle_reduce
// Run:    ./build/07-warp-programming/shuffle_reduce

#include <cuda_runtime.h>

#include <cstdio>

#include "../common/cuda_utils.h"

// TODO: implement kernel_A (broadcast), kernel_B (prefix sum), kernel_C (butterfly reduce)
// Each kernel: launch <<<1, 32>>> and have each thread printf its result

int main() {
    // TODO: launch each exercise kernel, verify results

    return 0;
}
