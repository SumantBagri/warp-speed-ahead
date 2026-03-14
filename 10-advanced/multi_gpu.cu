// 10-advanced/multi_gpu.cu
// ──────────────────────────
// Multi-GPU programming: peer access, P2P copies, and a simple all-reduce.
//
// Concepts:
//   - cudaSetDevice(n) — switch the active device for subsequent API calls
//   - cudaDeviceCanAccessPeer(a, b) — check if GPU a can access GPU b's memory
//   - cudaDeviceEnablePeerAccess(peer, 0) — enable P2P access
//   - cudaMemcpyPeer(dst, dstDev, src, srcDev, size) — direct GPU-to-GPU copy
//   - Manual all-reduce: each GPU computes a partial sum, results are gathered
//     and summed, then broadcast back
//   - For production: use NCCL (NVIDIA Collective Communications Library)
//
// Note: this machine has 1 GPU (RTX 3070), so P2P exercises use cudaMemcpy
//       between the same device to simulate the API pattern. The concepts
//       transfer directly to multi-GPU systems.
//
// Exercise A — peer access query:
//   Print the peer access matrix for all available devices.
//
// Exercise B — simulated all-reduce:
//   Split an array into N_GPU chunks, "process" each on a separate stream
//   (simulate multi-GPU with streams), manually reduce partial sums.
//
// Exercise C (if multi-GPU available):
//   Actual P2P: copy data from device 0 to device 1, process, copy back.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"

// TODO: Exercise A — peer access matrix
// TODO: Exercise B — simulated all-reduce with streams
// TODO: Exercise C — actual P2P (if hardware available)

int main() {
    // TODO

    return 0;
}
