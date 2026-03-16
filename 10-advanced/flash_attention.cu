// 10-advanced/flash_attention.cu
// ────────────────────────────────
// Flash Attention: IO-aware attention that tiles computation to stay in SRAM.
//
// This is a capstone exercise combining: tiling, shared memory, online softmax,
// register reuse, and arithmetic intensity reasoning.
//
// Background:
//   Standard attention:  S = Q × Kᵀ, P = softmax(S), O = P × V
//   Problem: S is N×N — for N=4096, that's 64M floats = 256 MB just for S.
//   Standard attention reads/writes N² values to HBM → memory-bound.
//
//   Flash Attention avoids materialising S by fusing the three ops into one kernel
//   and computing softmax incrementally using the "online softmax" trick.
//
// Online softmax (key insight):
//   softmax requires the global max to normalise — but you don't need to see
//   all values first. You can maintain a running (max, sum) pair and rescale
//   the accumulator each time you see a new tile that changes the running max.
//
//   m_new = max(m_old, tile_max)
//   acc   = acc * exp(m_old - m_new) + exp(tile_scores - m_new) × V_tile
//   l_new = l_old * exp(m_old - m_new) + sum(exp(tile_scores - m_new))
//   Output = acc / l_new
//
// Tiling strategy:
//   For each query tile (Br rows of Q):
//     For each key/value tile (Bc cols of K, rows of V):
//       Load Q tile, K tile, V tile into SRAM
//       Compute S_tile = Q_tile × K_tileᵀ  (Br × Bc)
//       Update running (m, l, acc) using online softmax
//     Write output tile to HBM
//
// Memory complexity: O(N) HBM reads (not O(N²))
//
// References:
//   Dao et al. (2022) — FlashAttention: Fast and Memory-Efficient Exact Attention
//   Dao (2023)        — FlashAttention-2
//
// Exercise:
//   Implement a simplified single-head Flash Attention for short sequences (N ≤ 1024).
//   Verify against standard (naive) attention.
//   Then profile both with Nsight Compute and compare HBM traffic.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "../common/cuda_utils.h"

#define HEAD_DIM 64  // d_k — dimension of Q, K, V vectors

// TODO: implement naive_attention (materialise full N×N score matrix)
// TODO: implement flash_attention (tiled, online softmax, no N×N matrix)
// TODO: verify both produce identical output
// TODO: compare kernel time and (via Nsight) HBM bytes read/written

int main() {
    const int N = 1024;  // sequence length
    const int D = HEAD_DIM;

    // TODO

    return 0;
}
