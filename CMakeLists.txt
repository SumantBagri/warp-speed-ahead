cmake_minimum_required(VERSION 3.20)
project(warp-speed-ahead CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

# RTX 3070 = Ampere = SM 8.6
# To auto-detect: set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -lineinfo")

# Shared header path
include_directories(${CMAKE_SOURCE_DIR}/common)

add_subdirectory(setup)
add_subdirectory(01-foundations)
add_subdirectory(02-thread-hierarchy)
add_subdirectory(03-memory-model)
add_subdirectory(04-memory-optimization)
add_subdirectory(05-matmul)
add_subdirectory(06-reductions)
add_subdirectory(07-warp-programming)
add_subdirectory(08-streams-concurrency)
add_subdirectory(09-profiling)
add_subdirectory(10-advanced)
