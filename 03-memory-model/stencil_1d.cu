// 03-memory-model/stencil_1d.cu
// ───────────────────────────────
// Apply a 1D stencil (weighted sum of neighbours) using shared memory tiling.
//
// Concepts covered:
//   - Halo / ghost cells: each tile needs RADIUS extra elements on each side
//   - Loading a tile with halos into shared memory cooperatively
//   - Boundary handling for the first and last tiles
//   - __syncthreads() placement: after loading, before computing
//
// Stencil definition:
//   out[i] = sum_{r=-RADIUS}^{RADIUS} coeff[r+RADIUS] * in[i+r]
//
// Without shared memory: each output element re-reads 2*RADIUS+1 global values.
//   With blockSize=256 and RADIUS=4: 256*9 = 2304 global reads per block.
// With shared memory: load blockSize + 2*RADIUS elements once per block.
//   256+8 = 264 global reads → most reads served from shared memory.
//
// Tile layout in shared memory (RADIUS=4, blockSize=8 shown):
//   [h h h h | 0 1 2 3 4 5 6 7 | h h h h]
//    left halo  block elements    right halo
//
// Exercise:
//   RADIUS = 4, coeff = {1,1,1,1,1,1,1,1,1} / 9  (box filter / moving average)
//   Verify against CPU reference.
//
// Build:  cmake --build build --target stencil_1d
// Run:    ./build/03-memory-model/stencil_1d

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "../common/cuda_utils.h"

#define RADIUS 4

// TODO: implement stencil kernel with shared memory tiling and halo loading
// Tip: shared tile size = blockDim.x + 2 * RADIUS
//      Each thread loads its own element; boundary threads also load halos.

int main() {
    const int N = 1 << 20;

    // TODO: init input, launch kernel, compare to CPU reference

    return 0;
}
