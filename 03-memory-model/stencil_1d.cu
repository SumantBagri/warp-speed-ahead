// 03-memory-model/stencil_1d.cu
// ───────────────────────────────
// Apply a 1D stencil (weighted sum of neighbours) using shared memory tiling.
//
// Concepts covered:
//   - Halo / ghost cells: each tile needs RADIUS extra elements on each side
//   - Loading a tile with halos into shared memory cooperatively
//   - Boundary handling for the first and last tiles
//   - __syncthreads() placement: after loading, before computing
//
// Stencil definition:
//   out[i] = sum_{r=-RADIUS}^{RADIUS} coeff[r+RADIUS] * in[i+r]
//
// Without shared memory: each output element re-reads 2*RADIUS+1 global values.
//   With blockSize=256 and RADIUS=4: 256*9 = 2304 global reads per block.
// With shared memory: load blockSize + 2*RADIUS elements once per block.
//   256+8 = 264 global reads → most reads served from shared memory.
//
// Tile layout in shared memory (RADIUS=4, blockSize=8 shown):
//   [h h h h | 0 1 2 3 4 5 6 7 | h h h h]
//    left halo  block elements    right halo
//
// Exercise:
//   RADIUS = 4, coeff = {1,1,1,1,1,1,1,1,1} / 9  (box filter / moving average)
//   Verify against CPU reference.
//
// Build:  cmake --build build --target stencil_1d
// Run:    ./build/03-memory-model/stencil_1d

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "../common/cuda_utils.h"

#define RADIUS 4

// TODO: implement stencil kernel with shared memory tiling and halo loading
// Tip: shared tile size = blockDim.x + 2 * RADIUS
//      Each thread loads its own element; boundary threads also load halos.
__global__ void stencil_1d(const float* in, const float* coeff, float* out, int n) {
    extern __shared__ float smem[];

    int tile = blockDim.x + 2 * RADIUS;
    int block_start = blockIdx.x * blockDim.x;

    // load the tile in smem
    for (int s = threadIdx.x; s < tile; s += blockDim.x) {
        int g = block_start - RADIUS + s;
        smem[s] = (g >= 0 && g < n) ? in[g] : 0.0f;
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float acc = 0.0f;
        for (int r = 0; r < 2 * RADIUS + 1; ++r) {
            acc += coeff[r] * smem[threadIdx.x + r];
        }
        out[tid] = acc;
    }
}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    // Box-filter coefficients: uniform average over 2*RADIUS+1 = 9 elements
    const int NCOEFF = 2 * RADIUS + 1;
    float h_coeff[NCOEFF];
    for (int i = 0; i < NCOEFF; i++) h_coeff[i] = 1.0f / NCOEFF;

    // Host input: random floats in [0, 1)
    float* h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)rand() / RAND_MAX;

    // CPU reference: zero-pad out-of-bounds reads
    float* h_ref = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        float acc = 0.0f;
        for (int r = -RADIUS; r <= RADIUS; r++) {
            int j = i + r;
            acc += h_coeff[r + RADIUS] * (j >= 0 && j < N ? h_in[j] : 0.0f);
        }
        h_ref[i] = acc;
    }

    // Device buffers
    float *d_in, *d_out, *d_coeff;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_coeff, NCOEFF * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coeff, h_coeff, NCOEFF * sizeof(float), cudaMemcpyHostToDevice));

    const int BLOCK = 256;
    const int GRID = (N + BLOCK - 1) / BLOCK;
    const size_t smem_bytes = (BLOCK + 2 * RADIUS) * sizeof(float);
    stencil_1d<<<GRID, BLOCK, smem_bytes>>>(d_in, d_coeff, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_out = (float*)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Verify: max absolute error across all elements
    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float err = fabsf(h_out[i] - h_ref[i]);
        if (err > max_err)
            max_err = err;
    }
    printf("max_err = %.6e -> %s\n", max_err, max_err < 1e-5f ? "PASS" : "FAIL");

    free(h_in);
    free(h_ref);
    free(h_out);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_coeff));
    return 0;
}
