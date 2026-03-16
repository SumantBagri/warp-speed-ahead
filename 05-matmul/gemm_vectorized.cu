// 05-matmul/gemm_vectorized.cu
// ──────────────────────────────
// Further optimise tiled GEMM with vectorised loads and register tiling.
//
// Concepts covered:
//   - float4 loads: read 4 floats in a single 128-bit instruction
//       float4 v = *reinterpret_cast<const float4*>(&ptr[i]);
//     Halves the number of load instructions, better utilises memory bus
//   - Register tiling (thread coarsening): each thread computes a WI×WJ
//     submatrix instead of a single element
//     - More arithmetic per thread → better compute/memory ratio
//     - Reduces __syncthreads() overhead relative to work done
//   - Double buffering: prefetch the next tile into a second shared buffer
//     while computing the current tile (hides global memory latency)
//
// Strategy (WI=WJ=4 register tile, TILE=64 or 128):
//   Each thread owns a WI×WJ register accumulator array.
//   Outer loop: load TILE columns of A and TILE rows of B into smem.
//   Inner loop: accumulate all WI×WJ products from smem into registers.
//   One __syncthreads() per tile instead of per element.
//
// Note: this is the direction professional GEMM libraries (CUTLASS, cuBLAS)
//       take before adding tensor cores. Don't worry about matching cuBLAS —
//       focus on understanding each technique.
//
// Build:  cmake --build build --target gemm_vectorized
// Run:    ./build/05-matmul/gemm_vectorized

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "../common/cuda_utils.h"

#define TILE 64
#define WI 4  // register tile height per thread
#define WJ 4  // register tile width per thread

// TODO: implement gemm with float4 loads and register tiling
// This is the hardest kernel so far — sketch the indexing on paper first

int main() {
    const int M = 1024, N = 1024, K = 1024;

    // TODO: same harness, compare TFLOPS across naive / tiled / vectorized

    return 0;
}
