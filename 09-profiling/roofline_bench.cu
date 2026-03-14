// 09-profiling/roofline_bench.cu
// ────────────────────────────────
// Build a roofline chart by measuring arithmetic intensity and performance
// for kernels across the memory-bound / compute-bound spectrum.
//
// Concepts covered:
//   - Roofline model: Performance = min(Peak_FLOPS, Peak_BW × AI)
//   - Arithmetic Intensity (AI) = FLOPs / bytes of DRAM traffic
//   - Ridge point: AI where compute roof = memory roof
//       RTX 3070: 20.3 TFLOPS / 448 GB/s ≈ 45 FLOP/byte
//   - Kernels below the ridge are memory-bound; above are compute-bound
//   - How to measure DRAM traffic: Nsight Compute l1tex metrics or manual calculation
//
// Kernels to benchmark (covering different AI values):
//   vector_copy:   out[i] = in[i]               AI ≈ 0.08 (2 arrays, 0 FLOP)
//   vector_add:    c = a + b                     AI ≈ 0.08 (3 arrays, 1 FLOP/elem)
//   stencil:       9-point 1D stencil            AI ≈ 0.25 (depends on cache reuse)
//   dot_product:   reduce(a[i] * b[i])           AI ≈ 0.08 (memory-bound)
//   poly_eval:     y = a0 + a1*x + a2*x² + ...  AI increases with poly degree
//   gemm_tiled:    matrix multiply (1024³)       AI ≈ 45+ with tiling (ridge point)
//
// For each kernel, collect:
//   - Kernel time (ms) from CudaTimer
//   - Manually computed FLOPs and bytes
//   - Achieved GFLOPS = FLOPs / time_s / 1e9
//   - Achieved BW = bytes / time_s / 1e9
//   - AI = FLOPs / bytes
//
// Print a table: kernel | AI (FLOP/byte) | GFLOPS | GB/s | bound?
//
// Build:  cmake --build build --target roofline_bench
// Run:    ./build/09-profiling/roofline_bench

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include "../common/cuda_utils.h"

// TODO: implement each kernel listed above (or reuse from earlier modules)
// TODO: collect timing data and print the roofline table

int main() {
    const int N = 1 << 24;

    // TODO: benchmark all kernels, print results

    return 0;
}
