// 10-advanced/persistent_kernel.cu
// ───────────────────────────────────
// A persistent kernel occupies the GPU continuously and pulls work from a queue,
// eliminating kernel launch overhead for fine-grained dynamic workloads.
//
// Concepts:
//   - Persistent kernel: launched once with enough blocks to saturate the GPU,
//     loops internally pulling tasks from a shared work queue
//   - Work stealing via atomicAdd on a global counter (task_id++)
//   - Termination condition checked each iteration
//   - Useful when: task granularity is small, task count is dynamic, or
//     you need to avoid the CPU-side launch overhead for thousands of small tasks
//
// Pattern:
//   __global__ void persistent_kernel(Task* queue, int* counter, int total_tasks) {
//       while (true) {
//           int task_id = atomicAdd(counter, 1);
//           if (task_id >= total_tasks) break;
//           process(queue[task_id]);
//       }
//   }
//   // Launch: persistent_kernel<<<num_SMs, blockSize>>>(...)
//   // num_SMs = prop.multiProcessorCount (one block per SM)
//
// Exercise:
//   Implement a persistent kernel that processes a variable-length list of
//   independent tasks (e.g., per-task vector scaling with different lengths).
//   Compare to a standard one-kernel-per-task approach.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"

// TODO: define a Task struct
// TODO: implement persistent_kernel with atomicAdd work stealing
// TODO: compare to standard launch approach

int main() {
    // TODO

    return 0;
}
