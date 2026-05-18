// 03-memory-model/shared_histogram.cu
// ─────────────────────────────────────
// Count occurrences of values in [0, BINS) using shared memory per block,
// then merge per-block histograms into a global result.
//
// Concepts covered:
//   - __shared__ array declaration
//   - __syncthreads() — mandatory barrier before reading data written by other threads
//   - atomicAdd() in shared memory (fast) vs global memory (slow)
//   - Two-phase pattern: accumulate locally in shared, then flush to global
//
// Why shared memory here:
//   A naive histogram does one atomicAdd to global memory per element.
//   With many threads hitting the same bin, global atomics serialize badly.
//   Using shared memory per block: contention is limited to blockDim.x threads,
//   and atomicAdd on shared mem is ~10× faster than on global.
//
// Pattern:
//   Phase 1 — each block initialises its shared histogram to zero
//             (all threads cooperatively zero it, then __syncthreads)
//   Phase 2 — each thread atomically increments the shared bin for its element
//             (then __syncthreads)
//   Phase 3 — each thread contributes one bin from shared → global with atomicAdd
//
// Exercise:
//   Input: array of N random integers in [0, BINS)
//   Output: histogram[b] = count of elements equal to b
//   Verify: sum of histogram == N
//
// Build:  cmake --build build --target shared_histogram
// Run:    ./build/03-memory-model/shared_histogram

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "../common/cuda_utils.h"

#define BINS 256

// TODO: implement histogram kernel using shared memory
// Signature: __global__ void histogram(const int* data, int* hist, int n)
__global__ void histogram(const int* data, int* hist, int n) {
    __shared__ int smem[BINS];

    // Zero shared histogram — loop handles blockDim.x != BINS
    for (int b = threadIdx.x; b < BINS; b += blockDim.x)
        smem[b] = 0;
    __syncthreads();

    // Grid-stride loop: each thread processes multiple elements, spreading
    // atomic contention across more iterations and hiding global-load latency
    // by keeping more warps in flight.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        atomicAdd(&smem[data[i]], 1);
    __syncthreads();

    // Flush shared → global
    for (int b = threadIdx.x; b < BINS; b += blockDim.x)
        atomicAdd(&hist[b], smem[b]);
}

int main() {
    const int N = 1 << 22;
    const size_t data_bytes = N * sizeof(int);
    const size_t hist_bytes = BINS * sizeof(int);

    // Generate random input in [0, BINS)
    int* h_data = (int*)malloc(data_bytes);
    for (int i = 0; i < N; i++) h_data[i] = rand() % BINS;

    int* d_data;
    int* d_hist;
    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_hist, hist_bytes));
    CUDA_CHECK(cudaMemset(d_hist, 0, hist_bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice));

    const int BLOCK = 1024;
    const int GRID = ((N + BLOCK - 1) / BLOCK < 4096) ? (N + BLOCK - 1) / BLOCK : 4096;
    histogram<<<GRID, BLOCK>>>(d_data, d_hist, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_hist[BINS];
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, hist_bytes, cudaMemcpyDeviceToHost));

    // Verify sum == N
    long long sum = 0;
    for (int i = 0; i < BINS; i++) sum += h_hist[i];
    printf("sum = %lld, N = %d -> %s\n", sum, N, sum == N ? "PASS" : "FAIL");

    free(h_data);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_hist));
    return 0;
}
