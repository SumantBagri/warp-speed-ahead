# GPU Architecture: From Basics to Advanced

A comprehensive guide to GPU internals — from the first transistor abstraction to Hopper's Transformer Engine.

---

## Table of Contents

1. [Why GPUs Exist](#1-why-gpus-exist)
2. [GPU vs CPU — The Core Difference](#2-gpu-vs-cpu--the-core-difference)
3. [GPU Hardware Hierarchy](#3-gpu-hardware-hierarchy)
4. [Streaming Multiprocessors (SMs)](#4-streaming-multiprocessors-sms)
5. [CUDA Thread Hierarchy](#5-cuda-thread-hierarchy)
6. [Warp Execution Model](#6-warp-execution-model)
7. [Memory Hierarchy](#7-memory-hierarchy)
8. [Memory Access Patterns](#8-memory-access-patterns)
9. [Occupancy and Latency Hiding](#9-occupancy-and-latency-hiding)
10. [Modern GPU Features](#10-modern-gpu-features)
11. [Architecture Generations](#11-architecture-generations)
12. [Key Takeaways](#12-key-takeaways)

---

## 1. Why GPUs Exist

GPUs were originally designed to accelerate real-time rasterization for graphics — transforming 3D geometry into 2D pixels involves applying the same simple math to millions of independent vertices and fragments simultaneously. That workload shaped GPU design: **maximize arithmetic throughput on data-parallel tasks** rather than minimize latency on sequential, branchy workloads.

This same property makes GPUs exceptional for scientific computing, machine learning, and any domain where the same operation is applied across a large dataset.

---

## 2. GPU vs CPU — The Core Difference

The most important architectural difference is how transistor budget is spent:

![CPU vs GPU die area allocation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-devotes-more-transistors-to-data-processing.png)

*Source: NVIDIA CUDA C++ Programming Guide*

| Feature | CPU | GPU |
|---|---|---|
| Core count | 8–128 | 1,000s–10,000s |
| Core design | Complex, out-of-order, deep pipeline | Simple, in-order, shallow pipeline |
| Cache | Large (MBs per core) | Small (KBs per SM) |
| Latency tolerance | Low (branch prediction, speculative exec) | High (hides via massive thread switching) |
| Optimization target | Single-thread throughput | Aggregate data-parallel throughput |

**CPUs minimize latency.** They use branch prediction, speculative execution, out-of-order execution, and large caches so any single thread runs as fast as possible.

**GPUs hide latency.** Instead of eliminating stalls, they switch to another group of threads the instant one group stalls. This works because they have thousands of threads in flight simultaneously.

---

## 3. GPU Hardware Hierarchy

A modern NVIDIA GPU (e.g., A100, H100) is organized as a strict hierarchy:

```
GPU
└── GPC (Graphics Processing Cluster) × N
    └── TPC (Texture Processing Cluster) × N
        └── SM (Streaming Multiprocessor) × N
            ├── Warp Schedulers × 4
            ├── CUDA Cores (INT32 / FP32 / FP64)
            ├── Tensor Cores
            ├── LD/ST Units
            ├── Special Function Units (SFU)
            ├── Register File (256 KB)
            └── Shared Memory / L1 Cache (up to 228 KB)
```

![NVIDIA A100 GPU Architecture](https://developer.nvidia.com/blog/wp-content/uploads/2020/06/a100-full-gpu-block-diagram-e1591322807599.png)

*Source: NVIDIA A100 Architecture Whitepaper — GA100 full die with 7 GPCs*

The A100 GA100 die has:
- **7 GPCs**, each with 7–8 SMs
- **108 SMs** total
- **6912 CUDA cores** (FP32)
- **432 Tensor Cores** (3rd gen)
- **80 GB HBM2e** memory

---

## 4. Streaming Multiprocessors (SMs)

The SM is the fundamental execution unit of a GPU. Every thread that runs on a GPU runs on an SM. Understanding SMs is the most important architectural concept for writing fast CUDA code.

### Ampere SM Internal Diagram

![NVIDIA Ampere SM Diagram](https://developer.nvidia.com/blog/wp-content/uploads/2020/09/nvidia-ampere-architecture-a100-sm.png)

*Source: NVIDIA Ampere Architecture Whitepaper*

### SM Components

| Component | A100 (per SM) | Purpose |
|---|---|---|
| FP32 CUDA Cores | 64 | Single-precision floating point |
| FP64 CUDA Cores | 32 | Double-precision floating point |
| INT32 Cores | 64 | Integer operations |
| 3rd-gen Tensor Cores | 4 | Matrix multiply-accumulate (MMA) |
| LD/ST Units | 32 | Load/store to global/shared memory |
| SFUs | 16 | sin, cos, sqrt, reciprocal |
| Warp Schedulers | 4 | Issue instructions each cycle |
| Register File | 256 KB | 65,536 × 32-bit registers per SM |
| Shared Mem / L1 | Up to 228 KB | Programmable fast on-chip memory |

### Execution Pipeline

Each SM has **4 warp schedulers**. Every clock cycle, each scheduler selects one eligible warp and issues up to 2 independent instructions from it (dual-issue). This means an SM can issue **8 instructions per cycle** total.

The scheduler selects from **resident warps** — warps that are allocated to the SM and not stalled. A warp stalls when:
- It is waiting on a memory load (L1/L2/DRAM)
- It is waiting on a preceding instruction in the same warp to complete
- It is waiting on a synchronization barrier

When a warp stalls, the scheduler immediately switches to another eligible warp at **zero cost** — no context-switch overhead. This zero-cost switching is the GPU's primary latency-hiding mechanism.

---

## 5. CUDA Thread Hierarchy

CUDA maps a programming model (threads → blocks → grids) onto the hardware (threads → warps → SMs → GPU).

![CUDA Grid, Block, Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-thread-blocks.png)

*Source: NVIDIA CUDA C++ Programming Guide*

### Programming Model

| CUDA Concept | Hardware Mapping |
|---|---|
| Thread | Single lane of a warp |
| Warp (32 threads) | Hardware execution unit on SM |
| Block | Set of warps assigned to one SM |
| Grid | All blocks dispatched to all SMs |

### Thread Indexing

Within a kernel, each thread identifies itself with:

```cuda
// 1D example
int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D example (for matrix operations)
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

### Block Scheduling

The SM can hold multiple blocks simultaneously, limited by:
- **Register usage**: Total registers = (threads/block) × (registers/thread)
- **Shared memory usage**: Total shared mem = blocks × (shared mem/block)
- **Max threads per SM**: Hardware limit (e.g., 2048 on A100)
- **Max blocks per SM**: Hardware limit (e.g., 32 on A100)

Blocks are dispatched to SMs as SMs become free. Once a block is assigned, it stays on that SM until all its threads complete.

---

## 6. Warp Execution Model

### SIMT Execution

GPUs execute code in **SIMT** (Single Instruction, Multiple Threads) fashion. All 32 threads in a warp execute the **same instruction at the same time**, each on its own data.

This is different from SIMD (Single Instruction, Multiple Data on a CPU vector unit) because:
- Each thread has its own program counter and register state
- Threads can individually stall, exit, or take different code paths
- The hardware manages divergence automatically

### Warp Divergence

When threads in a warp take different branches of an `if/else`, the warp must serialize both paths:

```
if (threadIdx.x < 16) {
    // Path A — executed first, threads 16-31 masked off
} else {
    // Path B — executed second, threads 0-15 masked off
}
```

![Warp divergence serializes branches](https://developer.nvidia.com/blog/wp-content/uploads/2018/09/cuda-pro-tip-write-flexible-kernels-with-warp-size-agnostic-code-2.png)

*Diverging threads within a warp execute serially rather than in parallel.*

**Impact**: A 50/50 split costs roughly 2× the time of a non-divergent warp. Minimize by:
- Structuring data access so all 32 threads take the same path
- Using `__ballot_sync()` and warp intrinsics for warp-level logic
- Ensuring branch conditions are aligned to warp boundaries (multiples of 32)

### Warp Scheduling Timeline

```
Cycle 1:  Warp A issues LOAD (→ will stall next cycle)
Cycle 2:  Warp B issues FADD (A is waiting on memory)
Cycle 3:  Warp C issues FMUL
Cycle 4:  Warp D issues LOAD
Cycle 5:  Warp B issues FMUL (back to B, was ready)
...
Cycle N:  Warp A issues FADD (memory returned, A is now eligible)
```

The scheduler hides the memory latency (~200–800 cycles for global DRAM) by keeping other warps in flight.

---

## 7. Memory Hierarchy

GPU memory is a strict hierarchy of speed vs. capacity:

![GPU Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/memory-hierarchy.png)

*Source: NVIDIA CUDA C++ Programming Guide*

### Full Hierarchy Table

| Memory Type | Location | Scope | Latency | Bandwidth | Size |
|---|---|---|---|---|---|
| **Registers** | On-chip (SM) | Thread | 1 cycle | ~20 TB/s | 256 KB/SM |
| **Shared Memory** | On-chip (SM) | Block | ~5 cycles | ~10 TB/s | Up to 228 KB/SM |
| **L1 Cache** | On-chip (SM) | SM | ~30 cycles | — | Part of shared mem pool |
| **L2 Cache** | On-chip (GPU) | GPU | ~200 cycles | 4 TB/s (A100) | 40 MB (A100) |
| **Global Memory (HBM)** | Off-chip | All threads | ~500–800 cycles | 2 TB/s (A100) | 40–80 GB |
| **Constant Memory** | Off-chip, cached | All threads (read-only) | 1 cycle if cached | — | 64 KB |
| **Texture Memory** | Off-chip, cached | All threads (read-only) | Variable | — | Up to global mem |

### Registers

- Fastest storage on the GPU
- Allocated statically at compile time per thread
- **Register spilling**: if a thread needs more registers than available, the compiler spills them to global memory — catastrophic for performance
- Check with: `nvcc --ptxas-options=-v` or Nsight Compute

### Shared Memory

Shared memory is manually managed, programmer-controlled cache shared among all threads in a block. It is the most impactful optimization lever in CUDA.

```cuda
__global__ void matmulKernel(float* A, float* B, float* C, int N) {
    __shared__ float tileA[TILE][TILE];   // lives in shared mem
    __shared__ float tileB[TILE][TILE];

    // Load tiles collaboratively, then compute from fast shared mem
    tileA[ty][tx] = A[row * N + (phase * TILE + tx)];
    tileB[ty][tx] = B[(phase * TILE + ty) * N + col];
    __syncthreads();
    // ... accumulate dot product from tiles
}
```

**Shared memory is ~100× faster than global memory and ~10 TB/s of aggregate bandwidth per SM.**

### L2 Cache (A100: 40 MB)

- Shared across all SMs
- Stores recently accessed global memory lines
- On A100, the large L2 can cache entire model weights for inference
- Residency hints: `cudaAccessPolicyWindow` can pin hot data in L2

### HBM2e / GDDR6X (Global Memory)

- Main GPU memory, off-chip
- High bandwidth but high latency (~500–800 cycles)
- A100 SXM: 80 GB HBM2e @ 2 TB/s
- H100 SXM: 80 GB HBM3 @ 3.35 TB/s

---

## 8. Memory Access Patterns

### Coalesced Memory Access

The memory controller fetches global memory in **128-byte cache lines**. If 32 threads in a warp access 32 consecutive 4-byte floats (128 bytes total), that is **one cache line** — fully coalesced.

**Coalesced (1 transaction):**
```cuda
// Thread i accesses element i — contiguous
float val = data[threadIdx.x + blockIdx.x * blockDim.x];
```

**Uncoalesced (up to 32 transactions):**
```cuda
// Thread i accesses element i * stride — scattered
float val = data[threadIdx.x * stride];
```

![Memory coalescing diagram](https://developer.nvidia.com/blog/wp-content/uploads/2018/09/Access_Patterrn_2.png)

*Aligned, consecutive thread accesses collapse into a single memory transaction.*

**Impact of stride:** On A100, strided access with stride=32 can reduce effective bandwidth by **32×** compared to coalesced access.

### Shared Memory Bank Conflicts

Shared memory is divided into **32 banks** (one per warp lane). Each bank is 4 bytes wide. Simultaneous accesses to the same bank by multiple threads in a warp are **serialized**.

```
Banks:  0   1   2   3   4  ...  31
Addr:   0   4   8  12  16  ... 124   (bytes, first row)
       128 132 136 ...              (second row)
```

**No conflict** — threads access different banks:
```cuda
shared[threadIdx.x]          // thread 0→bank 0, thread 1→bank 1, ...
```

**2-way bank conflict** — half the threads share a bank:
```cuda
shared[threadIdx.x * 2]      // thread 0 and thread 16 both hit bank 0
```

**32-way conflict (worst case)**:
```cuda
shared[0]                    // all 32 threads hit bank 0 — full serialization
```

**Exception**: a broadcast — when all threads read the **same address** — is one transaction with no penalty.

### Matrix Transpose Example

A naive transpose with global memory has either uncoalesced reads or uncoalesced writes. The fix uses shared memory as a staging buffer:

```cuda
__global__ void transpose(float* out, const float* in, int N) {
    __shared__ float tile[TILE][TILE + 1];   // +1 avoids bank conflicts

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    tile[threadIdx.y][threadIdx.x] = in[y * N + x];   // coalesced read
    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    out[y * N + x] = tile[threadIdx.x][threadIdx.y];  // coalesced write
}
```

The `+1` padding in the shared memory declaration staggers the bank mapping, eliminating column-access conflicts on the transposed load.

---

## 9. Occupancy and Latency Hiding

### Theoretical Occupancy

**Occupancy** = (active warps per SM) / (maximum warps per SM).

A100 supports 64 warps per SM maximum (2048 threads / 32). If a kernel uses 32 warps per SM, occupancy is 50%.

Three resources limit occupancy (whichever is most constraining wins):

```
Occupancy limiters:

1. Registers per thread:
   - Budget: 65,536 registers / SM
   - 64 threads/block × 32 registers/thread = 2048 threads (32 warps) → 50% occ.

2. Shared memory per block:
   - Budget: 228 KB / SM
   - 48 KB/block → at most 4 blocks/SM

3. Max blocks per SM: 32 (hardware limit on A100)
```

Use NVIDIA's [Occupancy Calculator](https://developer.nvidia.com/gpu-occupancy-calculator) or Nsight Compute to find bottlenecks.

### The Latency Hiding Equation

To fully hide global memory latency of ~800 cycles, you need enough warps in flight:

```
Warps needed = Memory latency (cycles) / Instruction throughput (cycles/warp)
             ≈ 800 / 4  (A100: 4 warp schedulers, each issues 1 warp/cycle)
             ≈ 200 warps needed at peak
```

A100 max is 64 warps/SM, so even at 100% occupancy you may not fully hide DRAM latency — which is why arithmetic intensity (flops per byte) matters: more compute per byte means fewer stalls.

---

## 10. Modern GPU Features

### Tensor Cores

Tensor Cores are specialized matrix units introduced in Volta (2017) that perform **4×4 matrix multiply-accumulate (MMA)** operations in a single clock cycle.

![Tensor Core Matrix Multiply](https://developer.nvidia.com/blog/wp-content/uploads/2020/06/tensor-core-diagram.png)

*Source: NVIDIA Developer Blog — 4×4 MMA in one clock cycle*

| Generation | Architecture | Mixed Precision | Peak (per SM) |
|---|---|---|---|
| 1st gen | Volta | FP16 × FP16 + FP32 | 125 TFLOPS (A100 total) |
| 2nd gen | Turing | FP16, INT8, INT4 | — |
| 3rd gen | Ampere | FP16, BF16, TF32, FP64 | 312 TFLOPS (A100, TF32) |
| 4th gen | Hopper | FP8, FP16, BF16, TF32 | 989 TFLOPS (H100, FP8) |
| 5th gen | Blackwell | FP4, FP6, FP8 | 3958 TFLOPS (B100, FP4) |

Access in CUDA via:
- **WMMA API** (`nvcuda::wmma`) — warp-level MMA
- **PTX MMA instructions** — low-level
- **cuBLAS/cuDNN** — library level (recommended for production)

### NVLink and NVSwitch

For multi-GPU systems, NVLink provides GPU-to-GPU interconnects that are far faster than PCIe:

| Interconnect | Bandwidth (bidirectional) |
|---|---|
| PCIe 4.0 ×16 | 64 GB/s |
| NVLink 3.0 (A100) | 600 GB/s (12 links) |
| NVLink 4.0 (H100) | 900 GB/s (18 links) |

NVSwitch (in DGX systems) provides full all-to-all NVLink connectivity between 8 GPUs with no bandwidth degradation.

### Thread Block Clusters (Hopper)

Hopper introduces a new level between blocks and grids: **thread block clusters**. Threads in different blocks within a cluster can use **distributed shared memory** — accessing each other's shared memory via SM-to-SM interconnects at ~6 TB/s.

```cuda
__cluster_dims__(2, 1, 1)  // 2-block cluster
__global__ void kernel() {
    // Access neighbor block's shared memory
    float* remote = cluster.map_shared_rank(ptr, 1);
}
```

### Hardware Raytracing (RT Cores)

Present in Turing and later (consumer GPUs), RT Cores accelerate **bounding volume hierarchy (BVH) traversal** and **ray-triangle intersection** tests in hardware — independent from CUDA cores and Tensor Cores.

---

## 11. Architecture Generations

| Architecture | Year | Process | Key Innovation |
|---|---|---|---|
| Fermi (GF100) | 2010 | 40nm | First CUDA-capable GPU with ECC, L1/L2 cache |
| Kepler (GK110) | 2012 | 28nm | Dynamic Parallelism, Hyper-Q (32 CUDA streams) |
| Maxwell (GM200) | 2014 | 28nm | Unified shared mem / L1, better occupancy scheduling |
| Pascal (GP100) | 2016 | 16nm | NVLink 1.0, HBM2, FP16 support |
| Volta (GV100) | 2017 | 12nm | **1st-gen Tensor Cores**, independent thread scheduling |
| Turing (TU102) | 2018 | 12nm | RT Cores, 2nd-gen Tensor Cores, INT8/INT4 |
| Ampere (GA100) | 2020 | 7nm | 3rd-gen Tensor Cores (TF32/BF16/FP64), 2× NVLink |
| Hopper (GH100) | 2022 | 4nm | 4th-gen Tensor Cores (FP8), Thread Block Clusters, NVLink 4.0 |
| Ada Lovelace (AD102) | 2022 | 5nm | 4th-gen Tensor Cores (consumer), RT Cores 3rd gen |
| Blackwell (GB100) | 2024 | 4nm | 5th-gen Tensor Cores (FP4), NVLink 5.0, 2× HBM3e |

### Pascal → Volta: Independent Thread Scheduling

Before Volta, all threads in a warp were implicitly synchronized — the hardware guaranteed they were at the same program counter. Volta introduced **independent thread scheduling**: each thread has its own PC and can truly diverge, with warp reconvergence managed by the compiler.

This enables lock-free data structures and warp-cooperative algorithms that were impossible before.

### Ampere: A100 vs A10G

The A100 (GA100) and A10G (GA102) are both Ampere but target different use cases:

| | A100 | A10G |
|---|---|---|
| FP64 CUDA Cores | 3456 | 0 |
| FP32 TFLOPS | 19.5 | 31.2 |
| FP64 TFLOPS | 9.7 | — |
| Memory | 80 GB HBM2e | 24 GB GDDR6 |
| TDP | 400 W | 150 W |
| Target | HPC / training | Inference / rendering |

---

## 12. Key Takeaways

**For writing fast CUDA kernels**, in order of importance:

1. **Maximize arithmetic intensity** — more FLOPs per byte means fewer DRAM stalls dominate
2. **Coalesce global memory accesses** — misaligned/strided access can cost 32× in bandwidth
3. **Use shared memory as L1** — manually stage tiles to avoid redundant global reads
4. **Avoid warp divergence** — branch on warp-aligned conditions or use warp intrinsics
5. **Minimize register pressure** — register spilling silently kills performance
6. **Tune occupancy** — enough warps in flight to hide latency, but not so many you thrash caches
7. **Use Tensor Cores** — for any matrix workload, they offer 4–16× the throughput of CUDA cores

**The GPU's contract with you**: it will hide all latency if you give it enough independent work. Latency hiding via massive parallelism is the fundamental mechanism behind everything else.

---

## References

- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA A100 Architecture Whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf)
- [NVIDIA H100 Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)
- [NVIDIA Volta Architecture Whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [GPU Occupancy Calculator](https://developer.nvidia.com/gpu-occupancy-calculator)
- [CUDA Parallel Reduction (Harris, 2007)](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
