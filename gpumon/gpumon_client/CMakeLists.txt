cmake_minimum_required(VERSION 3.18)
project(gpumon_client
    VERSION 0.1.0
    LANGUAGES CXX CUDA
    DESCRIPTION "Header-only GPU monitoring client library"
)

# Header-only library
add_library(gpumon INTERFACE)

target_include_directories(gpumon INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(gpumon INTERFACE cxx_std_17)
add_library(gpumon::gpumon ALIAS gpumon)

# Optional: Build example
option(BUILD_GPUMON_EXAMPLE "Build gpumon example application" ON)
if(BUILD_GPUMON_EXAMPLE AND CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    add_subdirectory(example)
endif()

# Installation support
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install header files
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install targets
install(TARGETS gpumon
    EXPORT gpumon_clientTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets
install(EXPORT gpumon_clientTargets
    FILE gpumon_clientTargets.cmake
    NAMESPACE gpumon::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpumon_client
)

# Create package config file
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpumon_clientConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/gpumon_clientConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpumon_client
)

# Create package version file
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/gpumon_clientConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install package config files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/gpumon_clientConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/gpumon_clientConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpumon_client
)
