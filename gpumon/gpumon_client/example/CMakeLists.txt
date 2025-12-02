cmake_minimum_required(VERSION 3.30)

enable_language(CUDA)

add_executable(gpumon_block_example block_style_example.cu)
target_link_libraries(gpumon_block_example PRIVATE gpumon::client)
set_target_properties(gpumon_block_example PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "native"
)