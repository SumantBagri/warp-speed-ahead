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

#include <cuda_runtime.h>

#include <cstdio>

#include "../common/cuda_utils.h"

// TODO: implement a kernel that prints lane_id and __activemask() in a branching block
__global__ void print_lane_id() {
    int lane_id = threadIdx.x & (warpSize - 1);

    if (lane_id < 16) {
        printf("lane id: %d | active mask: 0x%08x\n", lane_id, __activemask());
    } else {
        printf("lane id: %d | active mask: 0x%08x\n", lane_id, __activemask());
    }
}

__global__ void divergent_kernel(float* out, long long* clk, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;
    int lane_id = threadIdx.x & (warpSize - 1);

    long long start = clock64();
    float val = (float)tid;
    if (lane_id % 2 == 0) {
        for (int i = 0; i < 1000; i++) val = val * 1.01f + 0.5f;
    } else {
        for (int i = 0; i < 1000; i++) val = val * 0.99f - 0.5f;
    }
    out[tid] = val;
    // lane 0 of block 0 records cycles; warp serialises both branches so
    // it waits out the odd-lane path too — giving the full divergent cost
    if (blockIdx.x == 0 && lane_id == 0)
        *clk = clock64() - start;
}

__global__ void uniform_kernel(float* out, long long* clk, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;
    int lane_id = threadIdx.x & (warpSize - 1);

    long long start = clock64();
    float val = (float)tid;
    for (int i = 0; i < 1000; i++) val = val * 1.01f + 0.5f;
    out[tid] = val;
    if (blockIdx.x == 0 && lane_id == 0)
        *clk = clock64() - start;
}

int main() {
    // Exercise A: launch with 1 block of 32 threads, print mask values
    // print_lane_id<<<1, 32>>>();
    // CUDA_CHECK(cudaDeviceSynchronize());

    // Exercise B: launch with large N, time divergent vs uniform
    const int N = 1 << 24;
    const int block = 256;
    const int grid = (N + block - 1) / block;

    float* d_out;
    long long *d_clk_div, *d_clk_uni;
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_clk_div, sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_clk_uni, sizeof(long long)));

    divergent_kernel<<<grid, block>>>(d_out, d_clk_div, N);
    CUDA_CHECK(cudaGetLastError());

    uniform_kernel<<<grid, block>>>(d_out, d_clk_uni, N);
    CUDA_CHECK(cudaGetLastError());

    long long clk_div, clk_uni;
    CUDA_CHECK(cudaMemcpy(&clk_div, d_clk_div, sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&clk_uni, d_clk_uni, sizeof(long long), cudaMemcpyDeviceToHost));

    printf("\ndivergent_kernel: %lld cycles\n", clk_div);
    printf("uniform_kernel:   %lld cycles\n", clk_uni);
    printf("slowdown:         %.2fx\n", (double)clk_div / clk_uni);

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_clk_div));
    CUDA_CHECK(cudaFree(d_clk_uni));

    return 0;
}
