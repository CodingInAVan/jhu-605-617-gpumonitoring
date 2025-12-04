cmake_minimum_required(VERSION 3.18)

# Project name and version
project(gpu_monitoring VERSION 1.0.0 LANGUAGES CUDA CXX)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# CUDA and NVML configuration
find_package(CUDAToolkit REQUIRED)

# Find NVML library
find_library(NVML_LIBRARY
    NAMES nvml
    HINTS
        "${CUDAToolkit_LIBRARY_DIR}"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64"
        "$ENV{CUDA_PATH}/lib/x64"
    DOC "NVML library path"
)

if(NOT NVML_LIBRARY)
    message(FATAL_ERROR "NVML library not found. Please check your CUDA Toolkit installation.")
else()
    message(STATUS "Found NVML Library: ${NVML_LIBRARY}")
endif()

# NVML include directory
set(NVML_INCLUDE_DIR
    "${CUDAToolkit_INCLUDE_DIRS}"
    CACHE PATH "NVML include directory"
)
message(STATUS "NVML Include Directory: ${NVML_INCLUDE_DIR}")

# Export variables for subprojects
set(NVML_LIBRARY ${NVML_LIBRARY} CACHE INTERNAL "NVML library")
set(NVML_INCLUDE_DIR ${NVML_INCLUDE_DIR} CACHE INTERNAL "NVML include directory")

# Platform-specific settings
if(WIN32)
    # Windows-specific settings
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    set(PLATFORM_LIBS winhttp ws2_32)
elseif(UNIX)
    # Linux-specific settings
    set(PLATFORM_LIBS pthread curl)
endif()

# Export platform libs for subprojects
set(PLATFORM_LIBS ${PLATFORM_LIBS} CACHE INTERNAL "Platform-specific libraries")

# Build options
option(BUILD_CRAWLER "Build the GPU monitoring crawler application" ON)
option(BUILD_CLIENTLIB "Build the client library" ON)
option(BUILD_TESTS "Build tests" OFF)

# Add subdirectories
if(BUILD_CLIENTLIB)
    add_subdirectory(gpumon_client)
endif()

if(BUILD_CRAWLER)
    add_subdirectory(crawler)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "=== GPU Monitoring Project Configuration ===")
message(STATUS "Build Type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard:        ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard:       ${CMAKE_CUDA_STANDARD}")
message(STATUS "NVML Library:        ${NVML_LIBRARY}")
message(STATUS "Build Crawler:       ${BUILD_CRAWLER}")
message(STATUS "Build Client Lib:    ${BUILD_CLIENTLIB}")
message(STATUS "Build Tests:         ${BUILD_TESTS}")
message(STATUS "==========================================")
message(STATUS "")
