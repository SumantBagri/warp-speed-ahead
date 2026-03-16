// 08-streams-concurrency/event_timing.cu
// ─────────────────────────────────────────
// CUDA events for precise kernel timing and cross-stream synchronisation.
//
// Concepts covered:
//   - cudaEventCreate / cudaEventDestroy
//   - cudaEventRecord(event, stream) — insert a timestamp into a stream
//   - cudaEventSynchronize(event)    — CPU blocks until event is reached
//   - cudaEventElapsedTime(&ms, start, stop) — GPU-side elapsed time in ms
//   - cudaStreamWaitEvent(stream, event) — stream waits for event from another stream
//     (cross-stream dependency without blocking the CPU)
//
// Exercise A — kernel timing:
//   Time 5 back-to-back kernel launches with a single start/stop event pair.
//   Compare against timing each kernel individually (overhead of extra events).
//
// Exercise B — cross-stream dependency:
//   Stream 1: kernel A → records event E
//   Stream 2: cudaStreamWaitEvent(stream2, E) → kernel B (depends on A's output)
//   Without cudaStreamWaitEvent, stream 2 might start kernel B before A finishes.
//   Verify the dependency is enforced.
//
// Exercise C — measure transfer + compute overlap:
//   Use events to measure:
//     - Transfer time (H2D only)
//     - Kernel time (compute only)
//     - Total pipeline time (with overlap)
//   Show that total < transfer + compute when using async pipeline.
//
// Build:  cmake --build build --target event_timing
// Run:    ./build/08-streams-concurrency/event_timing

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "../common/cuda_utils.h"

// TODO: implement exercises A, B, C

int main() {
    // TODO

    return 0;
}
