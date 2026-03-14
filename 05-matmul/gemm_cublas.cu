// 05-matmul/gemm_cublas.cu
// ──────────────────────────
// Use cuBLAS SGEMM as the performance ceiling and correctness reference.
//
// Concepts covered:
//   - cuBLAS handle lifecycle: cublasCreate / cublasDestroy
//   - cublasSgemm signature and column-major convention
//   - Using a library kernel as a benchmark target
//
// cuBLAS uses column-major storage by default (Fortran convention).
// For row-major C arrays, you can use the identity:
//   C = A × B  (row-major)  ↔  Cᵀ = Bᵀ × Aᵀ  (column-major)
// So call cublasSgemm with A and B swapped and transposition flags adjusted.
//
// cublasSgemm signature:
//   cublasSgemm(handle, transB, transA, N, M, K,
//               &alpha, d_B, N, d_A, K, &beta, d_C, N)
//   (swapped A/B to handle row-major layout)
//
// Build:  cmake --build build --target gemm_cublas
// Run:    ./build/05-matmul/gemm_cublas

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../common/cuda_utils.h"

// No kernel to write here — the exercise is understanding the cuBLAS API
// and using it to establish a performance upper bound.

int main() {
    const int M = 1024, N = 1024, K = 1024;

    // TODO: cublasCreate handle
    // TODO: allocate and fill matrices
    // TODO: call cublasSgemm (pay attention to the row-major swap)
    // TODO: time it, print TFLOPS, compare to your gemm_tiled / gemm_vectorized
    // TODO: cublasDestroy handle

    return 0;
}
