add_executable(nvtx_annotated    nvtx_annotated.cu)
add_executable(roofline_bench    roofline_bench.cu)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(nvtx_annotated CUDA::nvToolsExt)
target_link_libraries(roofline_bench CUDA::nvToolsExt)
