// 04-memory-optimization/matrix_transpose.cu
// ─────────────────────────────────────────────
// Optimise a matrix transpose — the textbook coalescing example.
//
// Concepts covered:
//   - Why naive transpose is slow: one of read or write is always uncoalesced
//   - Using shared memory as a staging buffer to get coalesced I/O on both sides
//   - Shared memory bank conflicts in transpose and the +1 padding fix
//
// Three versions to implement:
//
// Version 1 — naive:
//   output[col][row] = input[row][col]
//   Read input[row][col]: threads in warp read consecutive cols → coalesced ✓
//   Write output[col][row]: threads in warp write to rows with stride N → NOT coalesced ✗
//
// Version 2 — shared memory tiled:
//   Load TILE×TILE submatrix of input into shared memory (coalesced read).
//   After __syncthreads, write transposed from shared to output (coalesced write).
//   shared[threadIdx.y][threadIdx.x] = input[row][col]   // coalesced read
//   output[out_row][out_col] = shared[threadIdx.x][threadIdx.y]  // coalesced write
//
// Version 3 — bank-conflict-free:
//   Version 2 has a 32-way bank conflict on the shared write:
//     threadIdx.y varies down the column → all hit bank (threadIdx.x % 32)
//   Fix: pad shared memory to TILE × (TILE + 1)
//   shared[TILE][TILE + 1]  — the +1 shifts each row by one bank
//
// Expected results on RTX 3070 (N=4096):
//   Naive:         ~100–150 GB/s (half the bandwidth wasted on uncoalesced writes)
//   Tiled:         ~300–380 GB/s
//   Tiled + pad:   ~380–420 GB/s
//
// Build:  cmake --build build --target matrix_transpose
// Run:    ./build/04-memory-optimization/matrix_transpose

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "../common/cuda_utils.h"

#define TILE 32

// TODO: implement transpose_naive
// TODO: implement transpose_tiled (shared memory, with bank conflicts)
// TODO: implement transpose_tiled_padded (shared memory, +1 padding)

int main() {
    const int N = 4096;  // square matrix N×N

    // TODO: allocate, run all three, verify each against CPU transpose, print timings

    return 0;
}
