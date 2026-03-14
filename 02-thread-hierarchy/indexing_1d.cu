// 02-thread-hierarchy/indexing_1d.cu
// ─────────────────────────────────────
// Visualise how threads are organised into blocks, warps, and a grid.
//
// Concepts covered:
//   - blockIdx.x, blockDim.x, threadIdx.x
//   - Global index formula: idx = blockIdx.x * blockDim.x + threadIdx.x
//   - Warp ID within a block: threadIdx.x / warpSize   (warpSize == 32)
//   - Lane ID within a warp:  threadIdx.x % warpSize   (equivalently: & 31)
//
// Exercise:
//   Fill the ThreadInfo struct for each thread, copy to host, and print.
//   Then re-launch with different configs — same 128 threads, different layouts:
//     <<<2, 64>>>   <<<4, 32>>>   <<<1, 128>>>   <<<8, 16>>>
//   Observe: global_idx is always the same; block_idx / warp_id change.
//
// Build:  cmake --build build --target indexing_1d
// Run:    ./build/02-thread-hierarchy/indexing_1d

#include <cstdio>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"

struct ThreadInfo {
    int global_idx;
    int block_idx;
    int thread_idx;
    int warp_id;
    int lane_id;
};

// TODO: implement the kernel — fill info[global_idx] for each thread

int main() {
    const int N = 128;

    // TODO: allocate d_info, launch with <<<2, 64>>>, copy back, print table
    // Columns: global_idx | block_idx | thread_idx | warp_id | lane_id

    return 0;
}
