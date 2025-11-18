cmake_minimum_required(VERSION 3.30)
project(crawler LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

add_executable(crawler app_main.cu MetricsSender.cpp GpuMonitor.cpp Config.cpp
        NvidiaSmiHelper.cpp)

find_library(NVML_LIBRARY
        NAMES nvml
        HINTS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64"
)

# NVML headers are in the CUDA include directory
set(CUDA_INCLUDE_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include")
include_directories("${CUDA_INCLUDE_DIR}")

if (NOT NVML_LIBRARY)
    message(FATAL_ERROR "NVML library not found. Please check the path in your CUDA Toolkit directory.")
else()
    message(STATUS "Found NVML Library: ${NVML_LIBRARY}")
endif()


set_target_properties(crawler PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(crawler PRIVATE ${NVML_LIBRARY} winhttp)