// 01-foundations/saxpy.cu
// ───────────────────────
// SAXPY: y[i] = a * x[i] + y[i]   (Single-precision A·X Plus Y)
// Classic BLAS level-1 operation.
//
// Concepts covered:
//   - Same full host/device lifecycle as vector_add (deliberate repetition)
//   - Scalar kernel argument: `a` is passed by value directly to the kernel
//     (no need to cudaMalloc a scalar — it fits in a register)
//   - In-place output: y is both read and written by the kernel
//
// After vector_add, you should need no hints here.
// The structure is identical; only the arithmetic changes.
//
// Build:  cmake --build build --target saxpy
// Run:    ./build/01-foundations/saxpy

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "../common/cuda_utils.h"

// TODO: write the saxpy kernel
// Signature: __global__ void saxpy(float a, const float* x, float* y, int n)

int main() {
    const int N = 1 << 24;
    const size_t bytes = N * sizeof(float);
    const float A = 2.5f;

    // TODO: full host/device lifecycle (same pattern as vector_add)
    // Reference result: h_ref[i] = A * h_x[i] + h_y[i]

    return 0;
}
