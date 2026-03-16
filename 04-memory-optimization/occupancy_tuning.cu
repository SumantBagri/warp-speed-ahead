// 04-memory-optimization/occupancy_tuning.cu
// ─────────────────────────────────────────────
// Explore how block size, register usage, and shared memory affect occupancy.
//
// Concepts covered:
//   - Occupancy = active warps per SM / maximum warps per SM
//     (RTX 3070 Ampere: max 48 warps per SM = 1536 threads)
//   - The three limiters: registers, shared memory, block size
//   - cudaOccupancyMaxActiveBlocksPerMultiprocessor — query achieved occupancy
//   - cudaOccupancyMaxPotentialBlockSize — suggest optimal blockSize
//   - __launch_bounds__(maxThreads, minBlocks) — hint to the compiler
//   - High occupancy hides memory latency but is not always the goal
//     (a register-heavy kernel at 50% occupancy can outperform a simple kernel at 100%)
//
// RTX 3070 SM resources:
//   Max warps:       48  (1536 threads)
//   Max blocks:      16
//   Registers:       65536 (per SM), max 255 per thread
//   Shared memory:   up to 100 KB per block (configurable)
//
// Exercise A — vary block size:
//   Write a simple bandwidth-bound kernel (e.g., scaled copy: out[i] = in[i] * 2).
//   Run it with blockSize = 32, 64, 128, 256, 512, 1024.
//   For each, query theoretical occupancy and measure actual throughput.
//   Question: is peak throughput always at maximum occupancy?
//
// Exercise B — register pressure:
//   Write a compute-heavy kernel that uses many local variables (registers).
//   Use `nvcc --ptxas-options=-v` to see register count.
//   Observe how register count caps the number of active warps per SM.
//
// Exercise C — shared memory:
//   Vary the amount of dynamic shared memory requested per block.
//   Use cudaOccupancyMaxActiveBlocksPerMultiprocessor to see occupancy drop
//   as shared mem per block increases.
//
// Build:  cmake --build build --target occupancy_tuning
// Run:    ./build/04-memory-optimization/occupancy_tuning

#include <cuda_runtime.h>

#include <cstdio>

#include "../common/cuda_utils.h"

// TODO: implement the bandwidth-bound copy kernel for Exercise A
// TODO: query and print occupancy for each block size using:
//   cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, kernel, blockSize, sharedBytes)
//   float occupancy = (float)(activeBlocks * blockSize) / prop.maxThreadsPerMultiProcessor;

int main() {
    const int N = 1 << 25;

    // TODO: Exercise A — sweep block sizes, print: blockSize | occupancy% | GB/s
    // TODO: Exercise B — register-heavy kernel, observe occupancy drop
    // TODO: Exercise C — vary sharedBytes, observe occupancy drop

    return 0;
}
