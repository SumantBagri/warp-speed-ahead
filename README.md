# warp-speed-ahead

A structured CUDA C++ learning project covering GPU programming from fundamentals to advanced optimization techniques — built for interview preparation for roles like **CUDA Kernel Developer** and **GPU Performance Engineer**.

**Hardware target:** NVIDIA RTX 3070 (Ampere, SM 8.6) | **CUDA:** 12.3

---

## Repository Structure

```
warp-speed-ahead/
├── README.md                    ← this file (roadmap + concept reference)
├── CMakeLists.txt               ← top-level build
├── setup/                       ← environment check, device query
├── 01-foundations/              ← Day 1: GPU model, launches, vector ops
├── 02-thread-hierarchy/         ← Day 2: grids, blocks, warps, indexing
├── 03-memory-model/             ← Day 3: global, shared, constant, local
├── 04-memory-optimization/      ← Day 4: coalescing, bank conflicts, cache
├── 05-matmul/                   ← Day 5-6: naive → tiled → vectorized GEMM
├── 06-reductions/               ← Day 6: reduction trees, warp intrinsics
├── 07-warp-programming/         ← Day 7: shuffle, vote, cooperative groups
├── 08-streams-concurrency/      ← Day 7+: streams, async transfers, overlap
├── 09-profiling/                ← ongoing: Nsight, roofline, NVTX
└── 10-advanced/                 ← async: tensor cores, graphs, multi-GPU, ...
```

---

## Roadmap

### Week 1 — Foundations to Intermediate

| Day | Module | Topics | Examples |
|-----|--------|---------|---------|
| 1 | [Foundations](#module-1-foundations) | GPU arch, CUDA model, launches | Device query, vector add, error handling |
| 2 | [Thread Hierarchy](#module-2-thread-hierarchy) | Grids, blocks, warps, indexing | 1D/2D indexing, grid-stride loops, warp ID |
| 3 | [Memory Model](#module-3-memory-model) | Memory hierarchy, shared/constant | Histogram, stencil, constant cache |
| 4 | [Memory Optimization](#module-4-memory-optimization) | Coalescing, bank conflicts | Coalesced vs strided access, matrix transpose |
| 5 | [Matrix Multiplication I](#module-5-matrix-multiplication) | Naive GEMM, tiled GEMM | Naive GEMM, shared-memory tiling |
| 6 | [Reductions](#module-6-reductions--scan) | Reduction trees, prefix scan | Naive → warp-shuffle reduction, thrust scan |
| 7 | [Warp Programming + Streams](#module-7-warp-level-programming) | Shuffle, vote, streams, overlap | Warp reduce, ballot, pipeline overlap |

### Async / Advanced (Post Week 1)

| Module | Topics |
|--------|--------|
| [Matrix Multiplication II](#module-5-matrix-multiplication) | Vectorized loads, register tiling, cuBLAS comparison |
| [Profiling Deep Dive](#module-9-profiling) | Nsight Compute, roofline model, NVTX instrumentation |
| [Tensor Cores](#module-10-advanced-topics) | WMMA API, mma.sync PTX, matrix accumulation |
| [CUDA Graphs](#module-10-advanced-topics) | Graph capture, instantiation, update |
| [Persistent Kernels](#module-10-advanced-topics) | Work queues, dynamic parallelism |
| [Multi-GPU](#module-10-advanced-topics) | Peer access, NVLink, NCCL all-reduce |
| [Flash Attention](#module-10-advanced-topics) | Online softmax, SRAM tiling, kernel fusion |
| [Custom Allocators](#module-10-advanced-topics) | Memory pools, caching allocator patterns |

---

## Concept Reference

All key concepts are documented here as a live reference that grows alongside the code examples.

---

### Module 1: Foundations

#### GPU Architecture Overview

A modern NVIDIA GPU (Ampere/Hopper) is organized as follows:

```
GPU
├── N × Streaming Multiprocessors (SMs)          ← RTX 3070: 46 SMs
│   ├── M × CUDA Cores (FP32 ALUs)               ← 128 per SM on Ampere
│   ├── Tensor Cores (matrix math units)
│   ├── L1 Cache / Shared Memory (unified pool)  ← 128 KB per SM on Ampere
│   ├── Register File                            ← 65536 × 32-bit regs per SM
│   └── Warp Schedulers (4 per SM on Ampere)
└── L2 Cache (shared across all SMs)             ← 4 MB on RTX 3070
    └── DRAM (device memory / global memory)     ← 8 GB GDDR6
```

**Key insight:** The CPU (host) and GPU (device) are separate processors with separate memory spaces. You must explicitly copy data between them — or use Unified Memory.

#### CUDA Programming Model

```
Host (CPU) code                   Device (GPU) code
─────────────────                 ─────────────────
Normal C++ functions              __global__ kernels (called from host)
                                  __device__ functions (called from device)
                                  __host__ __device__ (callable from both)

Execution flow:
1. Allocate device memory          cudaMalloc(&d_ptr, size)
2. Copy data host → device         cudaMemcpy(d, h, size, H2D)
3. Launch kernel                   kernel<<<grid, block>>>(args)
4. Synchronize                     cudaDeviceSynchronize()
5. Copy results device → host      cudaMemcpy(h, d, size, D2H)
6. Free device memory              cudaFree(d_ptr)
```

#### Kernel Launch Configuration

```cpp
// kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args);

// 1D example: process N elements
int blockSize = 256;
int gridSize  = (N + blockSize - 1) / blockSize;   // ceiling division
kernel<<<gridSize, blockSize>>>(d_data, N);

// Thread's global index:
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) { /* bounds check always required */ }
```

#### Error Handling Pattern

```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

CUDA_CHECK(cudaMalloc(&d_ptr, size));
```

**Examples in this module:** `setup/device_query.cu`, `01-foundations/vector_add.cu`, `01-foundations/saxpy.cu`

---

### Module 2: Thread Hierarchy

#### The Three-Level Hierarchy

```
Grid  (gridDim.x, gridDim.y, gridDim.z)
└── Block  (blockDim.x, blockDim.y, blockDim.z)   ← max 1024 threads/block
    └── Warp (32 threads, hardware scheduling unit)
        └── Thread (threadIdx.x, .y, .z)
```

**Rules:**
- Max threads per block: **1024**
- Max block dimensions: 1024 × 1024 × 64
- Max grid dimensions: 2³¹-1 × 65535 × 65535
- Warp size: always **32** threads (SIMT execution)

#### Global Index Formulas

```cpp
// 1D grid of 1D blocks
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D grid of 2D blocks (e.g., matrix operations)
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * width + col;   // row-major linear index

// Warp ID and lane ID (useful for warp intrinsics)
int warpId  = threadIdx.x / 32;
int laneId  = threadIdx.x % 32;   // or: threadIdx.x & 31
```

#### Grid-Stride Loop Pattern

Preferred over one-thread-per-element for large arrays — handles any N, better for occupancy:

```cpp
__global__ void kernel(float* data, int N) {
    int stride = gridDim.x * blockDim.x;          // total thread count
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride) {
        data[i] = process(data[i]);
    }
}
// Launch with fewer blocks than N requires — GPU reuses threads
kernel<<<1024, 256>>>(d_data, N);
```

#### Warps and SIMT Execution

- A **warp** is 32 threads executing the same instruction simultaneously (SIMT)
- All 32 threads in a warp share a program counter
- **Warp divergence:** when threads in the same warp take different branches, both paths execute serially — active threads are masked off on each pass
- Avoid divergence within warps; divergence *between* warps is fine

```cpp
// BAD: threads in same warp diverge
if (threadIdx.x % 2 == 0) { path_A(); } else { path_B(); }

// BETTER: structure so first 16 threads take A, next 16 take B
// (still diverges within warp, but unavoidable here)

// GOOD: divergence only between warps
if (threadIdx.x / 32 == 0) { path_A(); } else { path_B(); }
```

**Examples:** `02-thread-hierarchy/indexing_1d.cu`, `indexing_2d.cu`, `grid_stride.cu`, `warp_info.cu`

---

### Module 3: Memory Model

#### Memory Hierarchy (Speed → Capacity)

| Memory | Location | Scope | Lifetime | Latency | Size |
|--------|----------|-------|----------|---------|------|
| **Registers** | On-chip (per-thread) | Thread | Thread | ~1 cycle | 255 regs/thread |
| **Shared Memory** | On-chip (per-SM) | Block | Block | ~4-8 cycles | up to 48 KB* |
| **L1 Cache** | On-chip (per-SM) | SM | Kernel | ~4-8 cycles | shared pool with smem |
| **L2 Cache** | On-chip (whole GPU) | All SMs | Application | ~30-40 cycles | 4 MB (RTX 3070) |
| **Global Memory** | Off-chip (DRAM) | Host+Device | Application | ~200-400 cycles | 8 GB |
| **Constant Memory** | Off-chip (cached) | Device | Application | ~4 cycles (hit) | 64 KB |
| **Texture Memory** | Off-chip (cached) | Device | Application | ~4 cycles (hit) | subset of global |
| **Local Memory** | Off-chip (DRAM) | Thread | Thread | ~200-400 cycles | per-thread spill |

*On Ampere, the 128 KB SM on-chip pool is partitioned between L1 and shared memory — configurable up to 100 KB shared.

#### Shared Memory

The most important optimization primitive. Shared memory is:
- Programmer-managed L1 cache
- Shared among all threads in a block
- ~100× faster than global memory

```cpp
__global__ void kernel() {
    // Static allocation (size known at compile time)
    __shared__ float smem[256];

    // Dynamic allocation (size passed at launch: kernel<<<g, b, smemBytes>>>)
    extern __shared__ float smem_dyn[];

    // All threads write, then synchronize before any thread reads
    smem[threadIdx.x] = global_data[...];
    __syncthreads();   // ← REQUIRED barrier before reading neighbors

    float neighbor = smem[threadIdx.x + 1];
}
```

#### Shared Memory Bank Conflicts

Shared memory is divided into **32 banks** (one per warp lane). When multiple threads in a warp access the same bank simultaneously → **bank conflict** → serialized access.

```
32 banks:  [0][1][2]...[31][0][1][2]...[31]...
Address:    0  1  2   31  32  33  34   63
           (bank = address % 32  for 4-byte words)
```

```cpp
// NO conflict: each thread accesses a different bank (stride 1)
float val = smem[threadIdx.x];              // bank = laneId % 32 — all unique

// 2-WAY conflict: stride 2, threads 0 and 16 hit bank 0
float val = smem[threadIdx.x * 2];

// 32-WAY conflict: stride 32, all threads hit bank 0 (worst case)
float val = smem[threadIdx.x * 32];

// NO conflict (broadcast): all threads read the SAME address → broadcast
float val = smem[0];                        // hardware broadcasts, no conflict
```

#### Constant Memory

64 KB, cached, read-only. Optimal when **all threads read the same address** (broadcast). Bad for non-uniform access.

```cpp
__constant__ float filter[64];    // global scope

// Host side: use cudaMemcpyToSymbol
cudaMemcpyToSymbol(filter, h_filter, 64 * sizeof(float));
```

**Examples:** `03-memory-model/shared_histogram.cu`, `stencil_1d.cu`, `constant_filter.cu`

---

### Module 4: Memory Optimization

#### Memory Coalescing

The GPU memory system (128-byte cache lines) is most efficient when threads in a warp access **consecutive memory addresses** → all accesses combine into 1-4 transactions.

```
Warp: thread 0..31
addresses: 0,1,2,3,...,31    → 1 transaction   (coalesced ✓)
addresses: 0,2,4,6,...,62    → 2 transactions  (stride 2)
addresses: 0,32,64,...,992   → 32 transactions (stride 32, worst case ✗)
```

**Row-major vs column-major access:**

```cpp
// GOOD: threads read consecutive columns (row-major, coalesced)
float val = matrix[row * width + col];   // thread N reads col N → coalesced

// BAD: threads read consecutive rows (column-major, strided)
float val = matrix[col * height + row];  // thread N reads row N → NOT coalesced
```

**Matrix Transpose — the classic coalescing example:**

```cpp
// Naive: coalesced read, non-coalesced write (or vice versa) → bad
output[col * height + row] = input[row * width + col];

// Optimized: use shared memory as staging buffer
// Read coalesced into shared → write coalesced from shared
__shared__ float tile[TILE][TILE + 1];  // +1 to avoid bank conflicts
tile[ty][tx] = input[row * N + col];    // coalesced read
__syncthreads();
output[out_row * M + out_col] = tile[tx][ty];  // coalesced write (transposed idx)
```

#### Occupancy

**Occupancy** = active warps per SM / maximum warps per SM. Higher occupancy hides memory latency by allowing the warp scheduler to switch to other warps while one waits for data.

Limiting factors (all compete for SM resources):
1. **Register usage** — each thread uses N registers; max 65536 per SM
2. **Shared memory** — each block uses M bytes; max 128 KB per SM
3. **Block size** — max 1024 threads/block; max 16 blocks/SM (Ampere)

```cpp
// Query occupancy at runtime
int blockSize = 256;
int minGridSize, optBlockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optBlockSize, myKernel, 0, 0);

// Check achieved occupancy
int activeBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, myKernel, blockSize, 0);
float occupancy = (float)(activeBlocks * blockSize) / props.maxThreadsPerMultiProcessor;
```

**Compile-time occupancy hints:**
```cpp
// Limit register usage (may cause spilling to local memory)
__launch_bounds__(256, 2)   // maxThreadsPerBlock=256, minBlocksPerSM=2
__global__ void myKernel(...) { ... }
```

**Examples:** `04-memory-optimization/coalescing_benchmark.cu`, `matrix_transpose.cu`, `occupancy_tuning.cu`

---

### Module 5: Matrix Multiplication

GEMM (General Matrix-Matrix Multiplication) is the canonical GPU kernel — every optimization technique applies here.

#### Naive GEMM

```cpp
__global__ void gemm_naive(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];  // K global mem reads per output
    }
    C[row * N + col] = sum;
}
// Problem: each element of A/B is loaded from global memory O(N) times
// Bandwidth bound, not compute bound
```

#### Tiled GEMM (Shared Memory)

Key insight: threads in a block collectively compute a TILE×TILE output submatrix. Load tiles of A and B into shared memory → each element loaded once per tile instead of K times.

```
For M=N=K=1024, TILE=32:
  Naive:  1024² outputs × 1024 loads = 1B global memory reads
  Tiled:  1024² outputs × 32 loads  = 32M shared memory reads
                                       + 1024²/32² × 2×32² = 2M global reads
  → ~32× reduction in global memory traffic
```

```cpp
#define TILE 32
__global__ void gemm_tiled(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Load tile of A and B collaboratively
        As[threadIdx.y][threadIdx.x] = (row < M && t*TILE+threadIdx.x < K)
            ? A[row * K + t*TILE + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (t*TILE+threadIdx.y < K && col < N)
            ? B[(t*TILE+threadIdx.y) * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; k++) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}
```

#### Further GEMM Optimizations (Advanced)

| Technique | What it does | Speedup |
|-----------|-------------|---------|
| **Register tiling** | Each thread computes a WI×WJ submatrix using registers | 2-4× |
| **Vectorized loads** | Use `float4` to load 4 elements per instruction | 1.3-2× |
| **Double buffering** | Prefetch next tile while computing current | hides latency |
| **Warp tiling** | Structure tile assignment for warp-level locality | SM efficiency |
| **Tensor Cores** | Use WMMA/MMA instructions (16×16×16 ops) | 4-8× over FP32 |

**Examples:** `05-matmul/gemm_naive.cu`, `gemm_tiled.cu`, `gemm_vectorized.cu`, `gemm_cublas.cu`

---

### Module 6: Reductions & Scan

#### Parallel Reduction

Reduce N elements to 1 (sum, max, min, etc.) using a binary tree pattern.

**Naive (divergent, non-coalesced):**
```cpp
__global__ void reduce_naive(float* data, float* result, int N) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    smem[tid] = (gid < N) ? data[gid] : 0.0f;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0)          // BAD: warp divergence
            smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) result[blockIdx.x] = smem[0];
}
```

**Optimized (sequential addressing, no divergence):**
```cpp
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid + s];   // only first s threads active
    __syncthreads();
}
```

**Warp-level reduction (fastest — eliminates sync for last warp):**
```cpp
// Within a warp, __syncwarp or shuffle-based — no __syncthreads needed
if (tid < 32) {
    // Warp shuffle reduce
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
}
```

#### Prefix Scan (Prefix Sum)

Compute `out[i] = in[0] + in[1] + ... + in[i]` for all i in parallel.

- **Naive scan:** O(N log N) work (Hillis-Steele)
- **Work-efficient scan:** O(N) work, O(log N) steps (Blelloch up-sweep/down-sweep)
- **Real usage:** use Thrust `thrust::exclusive_scan` or CUB `DeviceScan`

**Examples:** `06-reductions/reduce_naive.cu`, `reduce_optimized.cu`, `reduce_warp.cu`, `prefix_scan.cu`

---

### Module 7: Warp-Level Programming

#### Warp Shuffle Intrinsics

Threads within a warp can directly exchange register values — no shared memory needed.

```cpp
// __shfl_sync(mask, var, srcLane)     — broadcast from srcLane
// __shfl_up_sync(mask, var, delta)    — shift up (higher lanes ← lower)
// __shfl_down_sync(mask, var, delta)  — shift down (lower lanes ← higher)
// __shfl_xor_sync(mask, var, laneMask)— butterfly pattern

int laneId = threadIdx.x & 31;
float val = ...; // each thread has a value

// Broadcast lane 0's value to all lanes
float broadcast = __shfl_sync(0xffffffff, val, 0);

// Prefix sum within warp (inclusive)
for (int offset = 1; offset < 32; offset <<= 1) {
    float n = __shfl_up_sync(0xffffffff, val, offset);
    if (laneId >= offset) val += n;
}
```

#### Warp Vote Intrinsics

```cpp
// All threads in warp evaluate predicate, results combined
unsigned mask = __activemask();

__all_sync(mask, pred)      // true if ALL active threads have pred != 0
__any_sync(mask, pred)      // true if ANY active thread has pred != 0
__ballot_sync(mask, pred)   // 32-bit mask of which lanes have pred != 0

// Example: find threads with value > threshold
unsigned active = __ballot_sync(0xffffffff, val > threshold);
int count = __popc(active);   // popcount = number of active threads
```

#### Cooperative Groups (CUDA 9+)

More flexible synchronization beyond `__syncthreads()`:

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel() {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    cg::thread_block_tile<16> half_warp = cg::tiled_partition<16>(block);

    warp.sync();              // synchronize just the warp
    float val = cg::reduce(warp, myVal, cg::plus<float>());  // warp reduce
}
```

**Examples:** `07-warp-programming/shuffle_reduce.cu`, `warp_vote.cu`, `coop_groups.cu`

---

### Module 8: Streams & Concurrency

#### CUDA Streams

A **stream** is a sequence of operations (kernels, memcpy) that execute in order on the GPU. Operations in *different* streams can overlap.

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// These may run concurrently on GPU
kernel_A<<<grid, block, 0, stream1>>>(d_a);
kernel_B<<<grid, block, 0, stream2>>>(d_b);

// Synchronize specific stream
cudaStreamSynchronize(stream1);

// Or synchronize all streams
cudaDeviceSynchronize();

cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

#### Overlapping Compute and Memory Transfer

**Requires pinned (page-locked) host memory** for async transfers:

```cpp
float* h_pinned;
cudaMallocHost(&h_pinned, size);    // pinned allocation

// Pipeline: while GPU processes chunk i, CPU prepares/receives chunk i±1
for (int i = 0; i < chunks; i++) {
    int stream = i % N_STREAMS;
    cudaMemcpyAsync(d_data + offset, h_pinned + offset, chunkSize, H2D, streams[stream]);
    kernel<<<grid, block, 0, streams[stream]>>>(d_data + offset, ...);
    cudaMemcpyAsync(h_out + offset, d_out + offset, chunkSize, D2H, streams[stream]);
}
```

#### CUDA Events (Timing & Synchronization)

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>(d_data);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);   // milliseconds
printf("Kernel time: %.3f ms\n", ms);
```

**Examples:** `08-streams-concurrency/streams_basic.cu`, `async_pipeline.cu`, `event_timing.cu`

---

### Module 9: Profiling

#### Roofline Model

The roofline model determines whether a kernel is **compute-bound** or **memory-bandwidth-bound**:

```
Performance = min(Peak FLOPS, Peak BW × Arithmetic Intensity)

Arithmetic Intensity (AI) = FLOPs / Bytes of memory traffic

RTX 3070 specs:
  Peak FP32:  20.3 TFLOPS
  Peak BW:    448 GB/s
  Ridge point: 20.3e12 / 448e9 ≈ 45 FLOP/byte

If AI < 45: memory-bound (focus on reducing memory traffic)
If AI > 45: compute-bound (focus on reducing FLOPs or using Tensor Cores)
```

**Kernel AI examples:**
- Vector add: ~1 FLOP/12 bytes → AI ≈ 0.08 (heavily memory-bound)
- Naive GEMM (N=1024): ~2N³ FLOPs / 3N² × 4 bytes → AI ≈ 170 (compute-bound if tiled)
- Stencil: ~10 FLOPs / 40 bytes → AI ≈ 0.25 (memory-bound)

#### Nsight Compute Workflow

```bash
# Profile a kernel, collect all metrics
ncu --set full -o profile_output ./my_kernel

# Focus on memory metrics
ncu --section MemoryWorkloadAnalysis ./my_kernel

# Focus on compute metrics
ncu --section ComputeWorkloadAnalysis ./my_kernel

# View in GUI
ncu-ui profile_output.ncu-rep
```

**Key metrics to watch:**
- `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` — global load bytes
- `smsp__sass_thread_inst_executed_op_fadd_pred_on` — FP32 adds
- `sm__warps_active.avg.pct_of_peak_sustained_active` — warp occupancy
- `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld` — sectors/request (1 = perfect coalescing)

#### NVTX Annotations

Mark ranges in your code so they appear labeled in Nsight:

```cpp
#include <nvtx3/nvToolsExt.h>

nvtxRangePush("data_preparation");
prepare_data(h_data, N);
nvtxRangePop();

nvtxRangePushA("forward_pass");
forward<<<grid, block>>>(d_data);
cudaDeviceSynchronize();
nvtxRangePop();
```

**Examples:** `09-profiling/nvtx_annotated.cu`, `roofline_benchmark.cu`

---

### Module 10: Advanced Topics

#### Tensor Cores (WMMA API)

Tensor Cores perform 16×16×16 matrix multiply-accumulate in a single instruction. Available in Volta+ (RTX 3070 = Ampere = 3rd gen tensor cores).

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void tensor_core_gemm(...) {
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    fill_fragment(c_frag, 0.0f);
    load_matrix_sync(a_frag, a_ptr, K);        // load from global/shared
    load_matrix_sync(b_frag, b_ptr, N);
    mma_sync(c_frag, a_frag, b_frag, c_frag);  // 16×16×16 MMA
    store_matrix_sync(c_ptr, c_frag, N, mem_row_major);
}
// Note: requires cooperative_groups or special warp arrangement
```

#### CUDA Graphs

Capture a sequence of operations into a graph, then replay with minimal CPU overhead (avoids kernel launch latency):

```cpp
cudaGraph_t graph;
cudaGraphExec_t instance;

// Capture
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernel_A<<<grid, block, 0, stream>>>(args);
kernel_B<<<grid, block, 0, stream>>>(args);
cudaStreamEndCapture(stream, &graph);

// Instantiate and launch
cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
cudaGraphLaunch(instance, stream);    // near-zero CPU overhead per launch

// Update parameters without re-instantiating
cudaGraphExecKernelNodeSetParams(instance, node, &newParams);
cudaGraphLaunch(instance, stream);
```

#### Multi-GPU Programming

```cpp
int nDevices;
cudaGetDeviceCount(&nDevices);

// Enable peer access between GPUs
cudaSetDevice(0);
cudaDeviceEnablePeerAccess(1, 0);   // GPU 0 can access GPU 1's memory

// Direct P2P copy
cudaMemcpyPeer(d_dst_gpu1, 1, d_src_gpu0, 0, size);

// NCCL all-reduce (distributed compute)
ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, stream);
```

#### Flash Attention (Case Study)

Flash Attention is a landmark GPU kernel that rewrites the attention mechanism to be IO-aware:

- Standard attention: O(N²) HBM reads/writes (materialized attention matrix)
- Flash Attention: O(N) HBM traffic via **online softmax** + SRAM tiling
- Key techniques: kernel fusion, online softmax with rescaling, blocked computation

This example combines: shared memory tiling, warp-level ops, register reuse, and careful arithmetic intensity maximization.

---

## Build System

```bash
# Build all modules
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86   # SM 8.6 = RTX 3070
make -j$(nproc)

# Build single module
cd 01-foundations && make
```

### CMake configuration
```cmake
cmake_minimum_required(VERSION 3.20)
project(warp-speed-ahead CUDA CXX)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xcompiler -Wall")
# -lineinfo: enables source-level profiling in Nsight
```

---

## Interview Topic Checklist

### Fundamentals
- [ ] Explain the SIMT execution model and how it differs from SIMD
- [ ] What is a warp? What happens during warp divergence?
- [ ] Describe the GPU memory hierarchy and access latencies
- [ ] How does memory coalescing work? What accesses are coalesced?
- [ ] What is occupancy, and what are the three factors that limit it?
- [ ] Explain the difference between `__syncthreads()` and `__syncwarp()`

### Memory
- [ ] When would you use shared memory vs registers vs global memory?
- [ ] What are shared memory bank conflicts and how do you avoid them?
- [ ] What is the difference between pinned and pageable host memory?
- [ ] How does the L1/L2 cache hierarchy work on Ampere?
- [ ] When is Unified Memory beneficial vs harmful?

### Optimization
- [ ] Walk through optimizing a matrix transpose kernel step by step
- [ ] How do you tile a GEMM kernel? What is the arithmetic intensity improvement?
- [ ] How does the roofline model guide optimization decisions?
- [ ] What is warp shuffle and when is it better than shared memory?
- [ ] How do CUDA streams enable computation/transfer overlap?

### Advanced
- [ ] How do Tensor Cores work? What precision modes are available?
- [ ] What problem do CUDA Graphs solve?
- [ ] How does Flash Attention reduce memory bandwidth usage?
- [ ] What is the difference between `cudaMalloc` and a pooled allocator?
- [ ] How would you design a multi-GPU all-reduce operation?

### Profiling & Debugging
- [ ] What Nsight Compute metrics do you check first for a slow kernel?
- [ ] How do you identify if a kernel is memory-bound vs compute-bound?
- [ ] What causes register spilling and how do you detect it?
- [ ] How do you use NVTX to annotate code for profiling?

---

## References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA Samples (NVIDIA)](https://github.com/NVIDIA/cuda-samples)
- [CUB Library](https://nvlabs.github.io/cub/)
- [cutlass (NVIDIA GEMM templates)](https://github.com/NVIDIA/cutlass)
- Volkov, V. — *Better Performance at Lower Occupancy* (GTC 2010)
- Harris, M. — *Optimizing Parallel Reduction in CUDA* (NVIDIA)
- Dao et al. — *FlashAttention: Fast and Memory-Efficient Exact Attention* (2022)
