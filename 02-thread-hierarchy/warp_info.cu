// 02-thread-hierarchy/warp_info.cu
// ──────────────────────────────────
// Observe warp divergence by forcing threads to take different code paths
// based on their lane ID.
//
// Concepts covered:
//   - Warp divergence: when threads in the same warp branch differently,
//     the warp executes both paths serially (inactive threads are masked)
//   - __activemask(): returns a 32-bit mask of currently active lanes
//   - Clock counter: clock64() for measuring divergence cost
//   - Divergence between warps is free; divergence within a warp is costly
//
// Exercise A — observe the mask:
//   In a kernel, have each thread print its lane_id and __activemask()
//   inside an if (lane_id < 16) / else branch.
//   Expected: inside the if-branch, only lanes 0–15 are active → mask = 0x0000FFFF
//
// Exercise B — measure divergence cost:
//   Create two kernels:
//     1. divergent_kernel:  if (lane_id % 2 == 0) do_work_A(); else do_work_B();
//     2. uniform_kernel:    all threads do the same amount of work
//   Time both and compare.
//
// Build:  cmake --build build --target warp_info
// Run:    ./build/02-thread-hierarchy/warp_info

#include <cstdio>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"

// TODO: implement a kernel that prints lane_id and __activemask() in a branching block
// TODO: implement divergent_kernel and uniform_kernel for timing comparison

int main() {
    // Exercise A: launch with 1 block of 32 threads, print mask values
    // Exercise B: launch with large N, time divergent vs uniform

    return 0;
}
