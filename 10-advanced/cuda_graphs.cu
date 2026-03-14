// 10-advanced/cuda_graphs.cu
// ────────────────────────────
// Capture a sequence of operations into a CUDA Graph and replay with minimal overhead.
//
// Concepts:
//   - Problem: each cudaLaunchKernel call has ~5–10 µs CPU overhead.
//     For thousands of small kernels (e.g., a neural net inference loop), this adds up.
//   - CUDA Graphs capture the entire sequence into a DAG once, then replay in one call.
//   - Three phases: capture → instantiate → launch (repeat)
//   - Graph updates: modify node parameters (e.g., pointers, constants) without re-instantiation
//
// Key API:
//   cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
//   // ... issue kernels and memcpys to stream ...
//   cudaStreamEndCapture(stream, &graph);
//   cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
//   cudaGraphLaunch(instance, stream);          // replays the whole graph
//   cudaGraphExecDestroy(instance);
//   cudaGraphDestroy(graph);
//
// Exercise:
//   Create a 10-kernel pipeline (e.g., 10× vector scale in sequence).
//   Measure total time for 1000 iterations:
//     Version A: explicit launch loop (1000 × 10 kernel launches)
//     Version B: capture into graph, launch graph 1000 times
//   At ~5 µs/launch: Version A ≈ 50 ms overhead; Version B ≈ 1 ms overhead.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"

// TODO: implement a simple kernel (vector scale)
// TODO: Version A — explicit loop
// TODO: Version B — graph capture and replay
// TODO: compare timing

int main() {
    // TODO

    return 0;
}
