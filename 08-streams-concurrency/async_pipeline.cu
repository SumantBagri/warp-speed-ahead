// 08-streams-concurrency/async_pipeline.cu
// ───────────────────────────────────────────
// Overlap memory transfers with kernel execution using multiple streams.
//
// Concepts covered:
//   - cudaMallocHost (pinned memory): required for async transfers
//     Pageable memory cannot be DMA'd — the driver must bounce it through a staging buffer,
//     which blocks the CPU until the copy finishes. Pinned memory allows true async DMA.
//   - cudaMemcpyAsync: returns immediately, copy happens in background
//   - Double-buffering pipeline: while GPU processes chunk i, copy chunk i+1
//   - Async engine count: RTX 3070 has 2 copy engines (H2D and D2H can run simultaneously)
//
// Pipeline pattern (N_STREAMS = 2, double buffer):
//
//   Stream 0: [H2D chunk 0] [kernel chunk 0] [D2H chunk 0]
//   Stream 1:               [H2D chunk 1]    [kernel chunk 1] [D2H chunk 1]
//              ──────────────────────────────────────────────────────────→ time
//
//   vs sequential:
//   Default:  [H2D chunk 0] [kernel chunk 0] [D2H chunk 0]
//             [H2D chunk 1] [kernel chunk 1] [D2H chunk 1]
//
// Exercise:
//   Process a large array in CHUNKS equal pieces.
//   Compare: (1) sequential stream, (2) 2-stream double buffer, (3) 4-stream pipeline.
//   Measure total time for each. The pipeline should be meaningfully faster.
//
// Build:  cmake --build build --target async_pipeline
// Run:    ./build/08-streams-concurrency/async_pipeline

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"

#define N_STREAMS 2
#define CHUNKS    8

// TODO: implement the compute kernel (e.g., multiply each element by 3)
// TODO: implement sequential version (baseline)
// TODO: implement pipelined version (N_STREAMS, double buffering)
// Note: use cudaMallocHost for host memory (pinned, required for async)

int main() {
    const int N        = 1 << 24;
    const size_t bytes = N * sizeof(float);

    // TODO: run both versions, compare timing

    return 0;
}
