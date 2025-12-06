cmake_minimum_required(VERSION 3.30)
project(. LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

add_executable(crawler app_main.cu MetricsSender.cpp GpuMonitor.cpp Config.cpp
        NvidiaSmiHelper.cpp ClientLogReader.cpp JsonUtils.cpp)

set_target_properties(crawler PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(crawler PRIVATE winhttp)