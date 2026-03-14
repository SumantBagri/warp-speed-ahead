// 09-profiling/nvtx_annotated.cu
// ─────────────────────────────────
// Instrument code with NVTX markers so Nsight Systems shows labelled ranges.
//
// Concepts covered:
//   - nvtxRangePushA / nvtxRangePop — named, nestable time ranges
//   - nvtxMarkA                     — single-point event (not a range)
//   - nvtxRangeStartA / nvtxRangeEnd — non-nested ranges (return a handle)
//   - Colour-coded ranges for readability in the Nsight Systems timeline
//   - Using NVTX to correlate CPU work with GPU kernels
//
// Header: #include <nvtx3/nvToolsExt.h>
// Link:   -lnvToolsExt  (handled in CMakeLists.txt via CUDA::nvToolsExt)
//
// Example API:
//   nvtxRangePushA("label");   // opens a range
//   ... code ...
//   nvtxRangePop();            // closes the most recently opened range
//
//   // With colour:
//   nvtxEventAttributes_t attr = {};
//   attr.version   = NVTX_VERSION;
//   attr.size      = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
//   attr.colorType = NVTX_COLOR_ARGB;
//   attr.color     = 0xFF00FF00;   // green
//   attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
//   attr.message.ascii = "green_range";
//   nvtxRangePushEx(&attr);
//
// Exercise:
//   Take your gemm_tiled or reduce_warp implementation and annotate:
//     - Data preparation (host alloc + init)
//     - H2D transfer
//     - Kernel execution
//     - D2H transfer
//     - Verification
//   Then run under Nsight Systems and inspect the annotated timeline:
//     nsys profile --stats=true ./build/09-profiling/nvtx_annotated
//
// Build:  cmake --build build --target nvtx_annotated
// Run:    nsys profile ./build/09-profiling/nvtx_annotated

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include "../common/cuda_utils.h"

// TODO: copy your gemm_tiled or reduce_warp kernel here
// TODO: wrap each phase with nvtxRangePushA / nvtxRangePop

int main() {
    // TODO: annotated version of a previous exercise

    return 0;
}
