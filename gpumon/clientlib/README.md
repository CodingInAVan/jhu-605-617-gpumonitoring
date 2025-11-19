# GPUmon Client Library v0.1

A header-only C++ library for logging GPU kernel activity in CUDA applications.

## Overview

The GPUmon client library enables CUDA applications to generate detailed GPU activity logs in NDJSON format. These logs are consumed by the GPUmon crawler/agent and sent to the backend for analysis.

## Features

- **Header-only**: Single header file, no compilation required
- **NDJSON logging**: One event per line, easy to parse
- **Kernel monitoring**: Automatic timing and configuration logging
- **Thread-safe**: Safe to use in multi-threaded applications
- **Minimal overhead**: Low-impact instrumentation
- **Cross-platform**: Windows and Linux support

## Quick Start

### 1. Include the header

```cpp
#include <gpumon/gpumon.hpp>
```

### 2. Initialize GPUmon

```cpp
int main() {
    gpumon::InitOptions opts;
    opts.appName = "my_cuda_app";
    opts.logFilePath = "gpumon.log";
    
    if (!gpumon::init(opts)) {
        std::cerr << "Failed to initialize gpumon" << std::endl;
        return 1;
    }
    
    // Your application code here...
    
    gpumon::shutdown();
    return 0;
}
```

### 3. Wrap kernel launches

Replace:
```cpp
myKernel<<<grid, block, sharedMem, stream>>>(args...);
```

With:
```cpp
GPUMON_LAUNCH(myKernel, grid, block, sharedMem, stream, args...);
```

## API Reference

### Initialization

```cpp
namespace gpumon {

struct InitOptions {
    std::string appName;      // Application identifier
    std::string logFilePath;  // Path to NDJSON log file
};

bool init(const InitOptions& opts);
void shutdown();

}
```

**init()** - Initializes the logging system
- Opens the log file in append mode
- Writes an initialization event
- Returns `true` on success, `false` on failure

**shutdown()** - Closes the logging system
- Writes a shutdown event
- Closes the log file
- Safe to call multiple times

### Region Marking

```cpp
void beginRegion(const std::string& name);
void endRegion();
```

Mark logical regions in your code (e.g., training epochs, processing phases).

Example:
```cpp
gpumon::beginRegion("epoch_1");
// ... multiple kernel launches ...
gpumon::endRegion();
```

### Kernel Launch Macro

```cpp
GPUMON_LAUNCH(kernel, grid, block, sharedMem, stream, ...)
```

**Parameters:**
- `kernel` - Kernel function name
- `grid` - Grid dimensions (dim3)
- `block` - Block dimensions (dim3)
- `sharedMem` - Shared memory bytes (size_t)
- `stream` - CUDA stream (cudaStream_t, use 0 for default)
- `...` - Kernel arguments

**Behavior:**
1. Captures start timestamp
2. Launches the kernel
3. Calls `cudaGetLastError()`
4. Calls `cudaDeviceSynchronize()` (blocking)
5. Captures end timestamp
6. Logs the event in NDJSON format

**Note:** In v0.1, the macro performs synchronous execution for simplicity.

## NDJSON Log Format

Each log entry is a single JSON object per line.

### Initialization Event

```json
{
  "type": "init",
  "pid": 1234,
  "app": "my_cuda_app",
  "ts_ns": 1731958400123456
}
```

### Kernel Event

```json
{
  "type": "kernel",
  "pid": 1234,
  "app": "my_cuda_app",
  "kernel": "vectorAdd",
  "ts_start_ns": 1731958400123456,
  "ts_end_ns": 1731958400126789,
  "grid": [128, 1, 1],
  "block": [256, 1, 1],
  "shared_mem_bytes": 0,
  "cuda_error": "cudaSuccess"
}
```

**Fields:**
- `type` - Event type ("kernel")
- `pid` - Process ID
- `app` - Application name from InitOptions
- `kernel` - Kernel function name
- `ts_start_ns` - Start timestamp (nanoseconds, monotonic clock)
- `ts_end_ns` - End timestamp (nanoseconds, monotonic clock)
- `grid` - Grid dimensions [x, y, z]
- `block` - Block dimensions [x, y, z]
- `shared_mem_bytes` - Shared memory allocation
- `cuda_error` - CUDA error string (if any)

### Region Events

```json
{"type":"region_begin","pid":1234,"app":"my_cuda_app","name":"epoch_1","ts_ns":1731958400123456}
{"type":"region_end","pid":1234,"app":"my_cuda_app","ts_ns":1731958400126789}
```

### Shutdown Event

```json
{
  "type": "shutdown",
  "pid": 1234,
  "app": "my_cuda_app",
  "ts_ns": 1731958400123456
}
```

## Complete Example

```cpp
#include <gpumon/gpumon.hpp>
#include <iostream>

__global__
void vectorAdd(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // Initialize GPUmon
    gpumon::InitOptions opts;
    opts.appName = "vector_add_demo";
    opts.logFilePath = "gpumon.log";
    
    if (!gpumon::init(opts)) {
        std::cerr << "Failed to initialize gpumon" << std::endl;
        return 1;
    }
    
    // Allocate memory
    const int n = 1024;
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(int));
    cudaMalloc(&d_b, n * sizeof(int));
    cudaMalloc(&d_c, n * sizeof(int));
    
    // Launch kernel with monitoring
    dim3 grid(4);
    dim3 block(256);
    GPUMON_LAUNCH(vectorAdd, grid, block, 0, 0, d_a, d_b, d_c, n);
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    gpumon::shutdown();
    return 0;
}
```

## CMake Integration

### Method 1: FetchContent (Recommended)

The easiest way to integrate GPUmon into your project:

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_cuda_app LANGUAGES CXX CUDA)

include(FetchContent)

# Fetch gpumon_client from Git
FetchContent_Declare(
    gpumon_client
    GIT_REPOSITORY https://github.com/your-org/gpumon.git
    GIT_TAG        v0.1.0  # or main, or a specific commit
    SOURCE_SUBDIR  clientlib
)

FetchContent_MakeAvailable(gpumon_client)

# Your CUDA application
add_executable(my_app main.cu)
target_link_libraries(my_app PRIVATE gpumon::client)
set_target_properties(my_app PROPERTIES
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
)
```

### Method 2: Git Submodule + add_subdirectory

```bash
# Add as submodule
git submodule add https://github.com/your-org/gpumon.git external/gpumon
git submodule update --init --recursive
```

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_cuda_app LANGUAGES CXX CUDA)

# Add gpumon_client
add_subdirectory(external/gpumon/clientlib)

# Your CUDA application
add_executable(my_app main.cu)
target_link_libraries(my_app PRIVATE gpumon::client)
set_target_properties(my_app PROPERTIES
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
)
```

### Method 3: System Installation

```bash
# Install gpumon_client
cd gpumon/clientlib
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
cmake --build .
sudo cmake --install .
```

Then in your project:

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_cuda_app LANGUAGES CXX CUDA)

# Find installed package
find_package(gpumon_client 0.1 REQUIRED)

# Your CUDA application
add_executable(my_app main.cu)
target_link_libraries(my_app PRIVATE gpumon::client)
set_target_properties(my_app PROPERTIES
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
)
```

## Building the Example

### Linux / macOS

```bash
mkdir build && cd build
cmake ..
cmake --build .

# Run the example
./clientlib/example/gpumon_example

# View the logs
cat gpumon.log
```

### Windows

#### Using Visual Studio:

```cmd
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release

REM Run the example
clientlib\example\Release\gpumon_example.exe

REM View the logs
type gpumon.log
```

#### Using Ninja:

```cmd
mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
cmake --build .

REM Run the example
clientlib\example\gpumon_example.exe

REM View the logs
type gpumon.log
```

## Thread Safety

All public API functions are thread-safe:
- Log writes are protected by a mutex
- Safe to call from multiple threads
- Safe to use with multi-stream CUDA applications

## Performance Considerations

**v0.1 Limitations:**
- `GPUMON_LAUNCH` performs synchronous execution (`cudaDeviceSynchronize()`)
- This blocks the CPU until kernel completion
- Suitable for debugging and profiling, but adds overhead

**Future versions** will support:
- Asynchronous timing with CUDA events
- Stream callbacks for non-blocking operation
- Optional sync vs. async modes

## Roadmap

**v0.2:**
- Asynchronous kernel timing with CUDA events
- Python bindings
- Configurable sync/async mode

**v0.3:**
- CUPTI-based automatic instrumentation
- Memory transfer logging
- GPU memory usage tracking

**v1.0:**
- Vulkan and Metal support
- Network transport (optional direct-to-backend)
- Advanced filtering and sampling

## Troubleshooting

**Problem:** Log file is not created

**Solution:** Check that the directory exists and is writable. Use an absolute path.

---

**Problem:** Compilation errors about `dim3`

**Solution:** Ensure you're compiling with NVCC and have included `<cuda_runtime.h>`.

---

**Problem:** Undefined reference errors

**Solution:** Make sure your file has `.cu` extension or use:
```cmake
set_target_properties(target PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```

## License

Part of the GPUmon project.
