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

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "../common/cuda_utils.h"

constexpr int BLOCK_SIZE = 128;

struct ThreadInfo {
    int global_idx;
    int block_idx;
    int thread_idx;
    int warp_id;
    int lane_id;
};

// TODO: implement the kernel — fill info[global_idx] for each thread
__global__ void get_thread_info(ThreadInfo* info) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    info[tid].global_idx = tid;
    info[tid].block_idx = blockIdx.x;
    info[tid].thread_idx = threadIdx.x;
    info[tid].warp_id = threadIdx.x / warpSize;
    info[tid].lane_id = threadIdx.x % warpSize;
}

int main() {
    const int N = 128;
    const size_t bytes = N * sizeof(ThreadInfo);

    // TODO: allocate d_info, launch with <<<2, 64>>>, copy back, print table
    // Columns: global_idx | block_idx | thread_idx | warp_id | lane_id
    ThreadInfo* d_info;
    CUDA_CHECK(cudaMalloc(&d_info, bytes));

    const int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    get_thread_info<<<gridSize, BLOCK_SIZE>>>(d_info);
    CUDA_CHECK(cudaGetLastError());

    std::vector<ThreadInfo> h_info(N);
    CUDA_CHECK(cudaMemcpy(h_info.data(), d_info, bytes, cudaMemcpyDeviceToHost));

    printf("%-10s | %-9s | %-10s | %-7s | %-7s\n", "global_idx", "block_idx", "thread_idx",
           "warp_id", "lane_id");
    for (auto thread_info : h_info) {
        printf("%-10d | %-9d | %-10d | %-7d | %-7d\n", thread_info.global_idx,
               thread_info.block_idx, thread_info.thread_idx, thread_info.warp_id,
               thread_info.lane_id);
    }

    CUDA_CHECK(cudaFree(d_info));

    return 0;
}
