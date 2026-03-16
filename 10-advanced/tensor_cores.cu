// 10-advanced/tensor_cores.cu
// ─────────────────────────────
// GEMM using Tensor Cores via the WMMA (Warp Matrix Multiply Accumulate) API.
//
// Concepts:
//   - Tensor Cores operate on 16×16×16 tiles: D = A × B + C
//   - Inputs must be FP16 (half), accumulator can be FP32
//   - The entire warp (32 threads) collaborates on one MMA operation
//   - WMMA fragments are opaque — layout is hardware-managed
//   - Performance: RTX 3070 has ~80 TFLOPS FP16 tensor (vs ~20 TFLOPS FP32 CUDA)
//
// Key API (#include <mma.h>, namespace nvcuda::wmma):
//   fragment<matrix_a, 16, 16, 16, half, row_major>    a_frag;
//   fragment<matrix_b, 16, 16, 16, half, col_major>    b_frag;
//   fragment<accumulator, 16, 16, 16, float>           c_frag;
//   fill_fragment(c_frag, 0.0f);
//   load_matrix_sync(a_frag, ptr, leading_dim);
//   mma_sync(c_frag, a_frag, b_frag, c_frag);
//   store_matrix_sync(ptr, c_frag, leading_dim, mem_row_major);
//
// Note: each warp handles one 16×16 output tile.
//       Block layout: tile the output matrix with one warp per 16×16 output tile.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdio>
#include <cstdlib>

#include "../common/cuda_utils.h"

using namespace nvcuda::wmma;

// TODO: implement tensor core GEMM
// TODO: compare TFLOPS to your gemm_tiled (FP32 CUDA cores) and to cuBLAS

int main() {
    const int M = 1024, N = 1024, K = 1024;
    // TODO

    return 0;
}
