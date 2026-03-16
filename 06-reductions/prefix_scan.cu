// 06-reductions/prefix_scan.cu
// ──────────────────────────────
// Parallel prefix sum (scan): out[i] = in[0] + in[1] + ... + in[i]
//
// Concepts covered:
//   - Inclusive vs exclusive scan
//   - Hillis-Steele (work-inefficient) scan: O(N log N) ops, O(log N) steps
//   - Blelloch (work-efficient) scan: O(N) ops, O(log N) steps — up-sweep + down-sweep
//   - CUB DeviceScan as the production-quality reference
//
// Hillis-Steele (easier to implement, less efficient):
//   for d = 1, 2, 4, ..., N/2:
//     if tid >= d: smem[tid] += smem[tid - d]
//     __syncthreads()
//   Total adds: N/2 + N/2 + ... = O(N log N) — more work than sequential scan
//
// Blelloch (work-efficient, used in practice):
//   Up-sweep (reduction tree): build partial sums bottom-up
//     for d = 1, 2, 4, ..., N/2:  smem[tid * 2d - 1] += smem[tid * 2d - 1 - d]
//   Set smem[N-1] = 0  (identity for addition)
//   Down-sweep: distribute partial sums top-down
//     for d = N/2, ..., 2, 1:  swap and add
//   Result: exclusive prefix sum
//
// For this exercise, implement Hillis-Steele for a single block (up to 1024 elements).
// Then compare to thrust::exclusive_scan or cub::DeviceScan for large arrays.
//
// Build:  cmake --build build --target prefix_scan
// Run:    ./build/06-reductions/prefix_scan

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "../common/cuda_utils.h"

// TODO: implement hillis_steele_scan kernel (single block, N ≤ 1024)
// TODO: verify against CPU prefix sum

int main() {
    const int N = 1024;  // start small: single block

    // TODO: run scan, verify, print result for first 16 elements

    return 0;
}
