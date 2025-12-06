cmake_minimum_required(VERSION 3.30)

# Project definition is good practice
project(GpuMonitor LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

if(NOT TARGET CUDA::nvml)
    message(STATUS "CUDA::nvml target not found automatically. Searching manually...")
    find_library(NVML_LIB
            NAMES nvml        # Windows uses 'nvml', Linux uses 'nvidia-ml'
            HINTS ${CUDAToolkit_LIBRARY_DIR}
    )
    find_path(NVML_INC
            NAMES nvml.h
            HINTS ${CUDAToolkit_INCLUDE_DIRS}
    )

    if(NVML_LIB AND NVML_INC)
        add_library(CUDA::nvml UNKNOWN IMPORTED)
        set_target_properties(CUDA::nvml PROPERTIES
                IMPORTED_LOCATION "${NVML_LIB}"
                INTERFACE_INCLUDE_DIRECTORIES "${NVML_INC}"
        )
    else()
        message(FATAL_ERROR "Could not find NVML library. Please check your CUDA installation.")
    endif()
endif()

# Define Executables
add_executable(gpumon_block_example block_style_example.cu)
add_executable(system_monitor system_monitor.cu)

# Enable NVML in your code
target_compile_definitions(system_monitor PRIVATE GPUMON_ENABLE_NVML)

# 3. Link Libraries
# Link against the target 'CUDA::nvml' instead of raw library names
target_link_libraries(gpumon_block_example PRIVATE gpumon::gpumon)

target_link_libraries(system_monitor PRIVATE
        gpumon::gpumon
        CUDA::cudart
        CUDA::nvml
)

# Properties
set_target_properties(gpumon_block_example PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "native"
)
set_target_properties(system_monitor PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "native"
)

# Platform Specifics
if(WIN32)
    # Windows-specific settings
    target_compile_definitions(system_monitor PRIVATE _CRT_SECURE_NO_WARNINGS)
    # Link system libs to the target directly
    target_link_libraries(system_monitor PRIVATE winhttp ws2_32)
elseif(UNIX)
    # Linux-specific settings
    target_link_libraries(system_monitor PRIVATE pthread curl)
endif()