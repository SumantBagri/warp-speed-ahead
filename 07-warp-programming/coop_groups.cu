// 07-warp-programming/coop_groups.cu
// ─────────────────────────────────────
// Cooperative Groups — flexible, composable synchronisation abstractions.
//
// Concepts covered:
//   - cg::this_thread_block()       — the classic __syncthreads() scope
//   - cg::tiled_partition<N>(block) — sub-block tile (N must be power of 2, ≤ 32)
//   - tile.sync()                   — synchronise within the tile only
//   - cg::reduce(tile, val, op)     — built-in reduce over a tile
//   - Why CG over raw intrinsics: composable, portable, self-documenting
//
// Header: #include <cooperative_groups.h>
//         namespace cg = cooperative_groups;
//
// Exercises:
//
// A — tile reduce:
//   Create a tile<16> (half-warp) and use cg::reduce to sum 16 values.
//   Compare to a manual __shfl_down_sync chain.
//
// B — sub-warp scan:
//   Partition a block into tile<8> groups.
//   Each tile computes an inclusive prefix sum of its 8 elements.
//   tile.shfl_up(val, offset) works like __shfl_up_sync but scoped to the tile.
//
// C — block-level sync vs tile-level sync:
//   Show that tile.sync() only synchronises the tile (32 or fewer threads),
//   which has lower overhead than block.sync() (__syncthreads).
//   Design a kernel where you can use tile.sync() instead of block.sync().
//
// Build:  cmake --build build --target coop_groups
// Run:    ./build/07-warp-programming/coop_groups

#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "../common/cuda_utils.h"

namespace cg = cooperative_groups;

// TODO: implement kernels for exercises A, B, C

int main() {
    // TODO: run each exercise

    return 0;
}
