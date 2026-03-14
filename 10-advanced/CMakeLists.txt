add_executable(tensor_cores       tensor_cores.cu)
add_executable(cuda_graphs        cuda_graphs.cu)
add_executable(persistent_kernel  persistent_kernel.cu)
add_executable(multi_gpu          multi_gpu.cu)
add_executable(flash_attention    flash_attention.cu)

target_link_libraries(tensor_cores cublas)
