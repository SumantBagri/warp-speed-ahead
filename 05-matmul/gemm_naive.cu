// 05-matmul/gemm_naive.cu
// ─────────────────────────
// Naive GEMM: C = A × B   (M×K) × (K×N) → (M×N)
// One thread computes one output element by iterating over the K dimension.
//
// Concepts covered:
//   - 2D thread/block mapping onto output matrix
//   - Inner-loop over the K (contraction) dimension
//   - Why this is slow: A is read with stride K (coalesced per row),
//     B is read with stride N per thread → column-major access = uncoalesced
//   - Each element of A and B is loaded from global memory K times per output block
//
// Arithmetic intensity:
//   FLOPs:  2 × M × N × K  (one multiply + one add per inner step)
//   Bytes:  (M×K + K×N + M×N) × 4  (read A, read B, write C)
//   For M=N=K=1024: AI ≈ 2×1024³ / (3×1024²×4) ≈ 170 FLOP/byte
//   → compute-bound on paper, but naive version is memory-bound due to
//     repeated global loads (no data reuse)
//
// Exercise:
//   Implement and run for M=N=K=1024.
//   Compute and print: kernel time, TFLOPS achieved.
//   cuBLAS on RTX 3070 achieves ~20 TFLOPS for FP32.
//   You should see ~0.5–2 TFLOPS here — the gap motivates all future optimisations.
//
// Build:  cmake --build build --target gemm_naive
// Run:    ./build/05-matmul/gemm_naive

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"

// TODO: implement gemm_naive kernel
// Each thread: row = ..., col = ...
//              for k in [0, K): sum += A[row*K + k] * B[k*N + col]
//              C[row*N + col] = sum

int main() {
    const int M = 1024, N = 1024, K = 1024;

    // TODO: allocate, fill with random data, launch, verify against CPU, print TFLOPS
    // TFLOPS = (2.0 * M * N * K) / (time_ms * 1e-3) / 1e12

    return 0;
}
