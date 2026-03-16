// 05-matmul/gemm_tiled.cu
// ─────────────────────────
// Tiled GEMM using shared memory — the most important CUDA optimisation pattern.
//
// Concepts covered:
//   - Tiling reduces global memory traffic by a factor of TILE
//   - Threads in a block cooperatively load a TILE×TILE submatrix of A and B
//     into shared memory, then compute their portion of the dot product
//   - Two __syncthreads() per tile iteration:
//       1st: after loading  — ensure all data is in smem before any thread reads it
//       2nd: after compute  — ensure all threads are done reading before the next tile overwrites
//       smem
//
// Tile iteration pseudocode:
//   for t in [0, ceil(K/TILE)):
//     smem_A[ty][tx] = A[row][t*TILE + tx]   // load tile of A (row from A, col tile from K)
//     smem_B[ty][tx] = B[t*TILE + ty][col]   // load tile of B (row tile from K, col from B)
//     __syncthreads()
//     for k in [0, TILE): sum += smem_A[ty][k] * smem_B[k][tx]
//     __syncthreads()
//
// Memory traffic comparison (M=N=K=1024, TILE=32):
//   Naive:  1024² threads × 1024 global loads each = 1B loads
//   Tiled:  1024²/32² tiles × 2×32² loads per tile = 2M loads  (~512× reduction)
//
// Build:  cmake --build build --target gemm_tiled
// Run:    ./build/05-matmul/gemm_tiled

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "../common/cuda_utils.h"

#define TILE 32

// TODO: implement gemm_tiled kernel with shared memory tiling
// Don't forget boundary checks for non-multiples-of-TILE matrix sizes

int main() {
    const int M = 1024, N = 1024, K = 1024;

    // TODO: same harness as gemm_naive — reuse code or factor it out
    // Compare TFLOPS: tiled should be 5–10× faster than naive

    return 0;
}
