// 02-thread-hierarchy/indexing_2d.cu
// ─────────────────────────────────────
// Extend 1D indexing to 2D grids and blocks — essential for matrix kernels.
//
// Concepts covered:
//   - 2D block: dim3 blockDim(TILE, TILE)  →  threadIdx.x = col, threadIdx.y = row
//   - 2D grid:  dim3 gridDim(ceil(W/TILE), ceil(H/TILE))
//   - Global row/col:
//       row = blockIdx.y * blockDim.y + threadIdx.y
//       col = blockIdx.x * blockDim.x + threadIdx.x
//   - Row-major linear index:  idx = row * width + col
//   - dim3 type for multi-dimensional launch configs
//
// Exercise:
//   Write a kernel that fills a 2D matrix where each element stores its
//   global (row, col) as a packed int: value = row * 1000 + col
//   Print the top-left 8×8 corner to verify.
//
//   Then answer: for a 32×32 matrix with blockDim(16,16), how many blocks
//   are launched? How many threads are idle (out-of-bounds) in a non-square
//   matrix where width = 50, height = 50?
//
// Build:  cmake --build build --target indexing_2d
// Run:    ./build/02-thread-hierarchy/indexing_2d

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "../common/cuda_utils.h"

// TODO: implement a kernel that writes (row * 1000 + col) into each element
// Kernel signature: __global__ void fill_2d(int* mat, int width, int height)
__global__ void fill_2d(int* mat, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {
        mat[row * width + col] = row * 1000 + col;
    }
}

int main() {
    const int W = 32, H = 32;
    const size_t bytes = W * H * sizeof(int);

    // TODO: allocate, launch with a 2D dim3 config, copy back, print 8×8 corner
    std::vector<int> h_mat(W * H, 0);
    int* d_mat;
    CUDA_CHECK(cudaMalloc(&d_mat, bytes));

    const int TILE = 16;
    dim3 blockDim(TILE, TILE);
    dim3 gridDim((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);
    fill_2d<<<gridDim, blockDim>>>(d_mat, W, H);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_mat.data(), d_mat, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%d\t", h_mat[i * W + j]);
        }
        printf("\n");
    }

    CUDA_CHECK(cudaFree(d_mat));

    return 0;
}
