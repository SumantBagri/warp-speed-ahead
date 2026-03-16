// 08-streams-concurrency/streams_basic.cu
// ──────────────────────────────────────────
// CUDA streams: issue independent operations that can overlap on the GPU.
//
// Concepts covered:
//   - Default stream (stream 0): all operations serialised
//   - Named streams: cudaStreamCreate / cudaStreamDestroy
//   - Operations in different streams may overlap (HW-permitting)
//   - cudaStreamSynchronize(stream) — wait for one stream
//   - cudaDeviceSynchronize()       — wait for all streams
//   - cudaEvent + cudaEventRecord across streams for cross-stream sync
//
// Exercise A — serial vs parallel streams:
//   Launch 4 independent kernels (e.g., vector_scale on 4 separate arrays).
//   Version 1: all in the default stream → serialised, time = 4 × T
//   Version 2: each in its own stream    → parallel, time ≈ T (if GPU has capacity)
//   Compare total wall-clock times using cudaEvents around the full sequence.
//
// Exercise B — stream ordering:
//   Demonstrate that operations within a single stream always execute in order,
//   while operations in different streams may interleave.
//   Use NVTX ranges (see 09-profiling) to visualise in Nsight Systems.
//
// Requirement: operations must be truly independent to overlap.
//   Data dependencies across streams require explicit event-based sync.
//
// Build:  cmake --build build --target streams_basic
// Run:    ./build/08-streams-concurrency/streams_basic

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "../common/cuda_utils.h"

// TODO: implement a simple kernel to time (e.g., vector scale: out[i] = in[i] * 2)
// TODO: Exercise A — serial (default stream) vs 4 named streams, compare times
// TODO: Exercise B — verify in-stream ordering

int main() {
    // TODO

    return 0;
}
