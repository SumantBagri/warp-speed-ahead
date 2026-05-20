// 03-memory-model/constant_filter.cu
// ─────────────────────────────────────
// Apply a convolution filter stored in constant memory.
//
// Concepts covered:
//   - __constant__ declaration (file-scope, device-visible, read-only)
//   - cudaMemcpyToSymbol — the correct way to populate constant memory
//   - When constant memory excels: all threads read the same address (broadcast)
//   - When it's bad: non-uniform access across a warp (serialised, not cached)
//
// Constant memory characteristics:
//   - 64 KB total, cached in a dedicated constant cache (per-SM)
//   - When all threads in a warp read the same address → single broadcast (fast)
//   - When threads read different addresses → requests are serialised (slow)
//   - Best use case: filter kernels, look-up tables, parameters read uniformly
//
// Exercise:
//   Implement a 1D convolution where the filter coefficients live in
//   __constant__ float filter[FILTER_SIZE].
//   Compare performance vs passing the filter as a regular pointer argument
//   (which goes through L1/L2 global cache, not the constant cache).
//
// Build:  cmake --build build --target constant_filter
// Run:    ./build/03-memory-model/constant_filter

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "../common/cuda_utils.h"

#define FILTER_SIZE 17

// TODO: declare the filter in constant memory
// Hint: __constant__ float filter[FILTER_SIZE];  (file scope, outside any function)
__constant__ float d_filter[FILTER_SIZE];

// TODO: implement the convolution kernel using the constant-memory filter
// TODO: implement a second version that takes filter as a const float* argument
// Compare their timings.
__global__ void conv1d_smem(const float* in, float* out, int n) {
    extern __shared__ float smem[];

    int tile = blockDim.x + FILTER_SIZE - 1;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int block_start = blockIdx.x * blockDim.x;
    for (int s = threadIdx.x; s < tile; s += blockDim.x) {
        int g = block_start - FILTER_SIZE / 2 + s;
        smem[s] = (g >= 0 && g < n) ? in[g] : 0.0f;
    }
    __syncthreads();

    if (tid < n) {
        float acc = 0.0f;
        for (int i = 0; i < FILTER_SIZE; ++i) {
            acc += d_filter[i] * smem[threadIdx.x + i];
        }
        out[tid] = acc;
    }
}

__global__ void conv1d_constant(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;

    float acc = 0.0f;
    for (int i = 0; i < FILTER_SIZE; ++i) {
        int j = tid - FILTER_SIZE / 2 + i;
        acc += d_filter[i] * (j >= 0 && j < n ? in[j] : 0.0f);
    }
    out[tid] = acc;
}

__global__ void conv1d_pointer(const float* in, const float* filter, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n)
        return;

    float acc = 0.0f;
    for (int i = 0; i < FILTER_SIZE; ++i) {
        int j = tid - FILTER_SIZE / 2 + i;
        acc += filter[i] * (j >= 0 && j < n ? in[j] : 0.0f);
    }
    out[tid] = acc;
}

int main() {
    const int N = 1 << 22;
    const size_t bytes = N * sizeof(float);
    const size_t filter_bytes = FILTER_SIZE * sizeof(float);

    // TODO: populate filter with cudaMemcpyToSymbol
    float h_filter[FILTER_SIZE];
    for (int i = 0; i < FILTER_SIZE; ++i) h_filter[i] = 1.0f / FILTER_SIZE;

    float* h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) h_in[i] = (float)rand() / RAND_MAX;

    float* h_ref = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) {
        float acc = 0.0f;
        for (int r = 0; r < FILTER_SIZE; ++r) {
            int j = i - FILTER_SIZE / 2 + r;
            acc += h_filter[r] * (j >= 0 && j < N ? h_in[j] : 0.0f);
        }
        h_ref[i] = acc;
    }

    float* h_out = (float*)malloc(bytes);

    // allocate device buffers
    float *d_in, *d_out, *d_filt;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_filt, filter_bytes));

    // copy input host -> device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int BLOCK_SIZE = 256;
    int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // copy symbol for constant kernel
    CUDA_CHECK(cudaMemcpyToSymbol(d_filter, h_filter, filter_bytes));

    size_t smem_bytes = (BLOCK_SIZE + FILTER_SIZE - 1) * sizeof(float);

    // warmup smem kernel
    conv1d_smem<<<GRID_SIZE, BLOCK_SIZE, smem_bytes>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.begin();
    conv1d_smem<<<GRID_SIZE, BLOCK_SIZE, smem_bytes>>>(d_in, d_out, N);
    float end = timer.end();

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    bool is_equal = arrays_equal(h_out, h_ref, N);
    printf("[%s] - Conv1d with constant filter and shared memory: %.3f ms\n",
           is_equal ? "PASS" : "FAIL", end);

    // copy the filter
    CUDA_CHECK(cudaMemcpy(d_filt, h_filter, filter_bytes, cudaMemcpyHostToHost));

    // warmup constant kernel
    conv1d_constant<<<GRID_SIZE, BLOCK_SIZE>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.begin();
    conv1d_constant<<<GRID_SIZE, BLOCK_SIZE>>>(d_in, d_out, N);
    end = timer.end();

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    is_equal = arrays_equal(h_out, h_ref, N);
    printf("[%s] - Conv1d with constant filter: %.3f ms\n", is_equal ? "PASS" : "FAIL", end);

    // copy the filter
    CUDA_CHECK(cudaMemcpy(d_filt, h_filter, filter_bytes, cudaMemcpyHostToHost));

    // warmup pointer kernel
    conv1d_pointer<<<GRID_SIZE, BLOCK_SIZE>>>(d_in, d_filt, d_out, N);

    timer.begin();
    conv1d_pointer<<<GRID_SIZE, BLOCK_SIZE>>>(d_in, d_filt, d_out, N);
    end = timer.end();

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    is_equal = arrays_equal(h_out, h_ref, N);
    printf("[%s] - Conv1d with pointer filter: %.3f ms\n", is_equal ? "PASS" : "FAIL", end);

    free(h_in);
    free(h_ref);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_filt);

    return 0;
}
