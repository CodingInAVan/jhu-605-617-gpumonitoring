# GPUmon Client Library (multi-header)

Header-only C++ instrumentation library that logs GPU activity and scoped phases in NDJSON, designed for multiple GPU backends. Currently ships with a CUDA backend; other backends can be added behind the same core API.

## What changed

The client moved from a single-header to a multi-header layout to support multiple platforms and cleaner separation of concerns:
- `gpumon/gpumon.hpp` — umbrella include that auto-selects a backend
- `gpumon/core/common.hpp` — core types/utilities (logging, JSON helpers)
- `gpumon/core/monitor.hpp` — public API (init/shutdown, ScopedMonitor, macros)
- `gpumon/backends/cuda.hpp` — CUDA-specific helpers and launch macros

You should include only `gpumon/gpumon.hpp` in application code.

## Features

- Header-only, zero link-time cost
- NDJSON logging, one event per line
- Scoped monitoring with optional periodic GPU memory sampling
- CUDA kernel auto-timing macros (sync for now)
- Thread-safe logging
- Cross-platform (Windows/Linux) for CUDA; backend abstraction for future platforms

## Backends and auto-detection

- CUDA is picked automatically if compiling with NVCC (`__CUDACC__`).
- To force-select, define one of:
  - `GPUMON_BACKEND_CUDA`
  - `GPUMON_BACKEND_OPENCL` (not yet implemented; will error if selected)

## Quick Start

### 1) Include

```cpp
#include <gpumon/gpumon.hpp>
```

### 2) Initialize and Shutdown

```cpp
int main() {
    gpumon::InitOptions opts;
    opts.appName = "my_cuda_app";
    opts.logFilePath = "gpumon.log";    // leave empty to use GPUMON_LOG_DIR
    opts.sampleIntervalMs = 0;           // >0 enables periodic memory sampling in scopes

    gpumon::init(opts);

    // ... your code ...

    gpumon::shutdown();
}
```

Environment support:
- `GPUMON_LOG_DIR` — if `logFilePath` is empty, the log path becomes `GPUMON_LOG_DIR/gpumon_<app>_<pid>.log`.

### 3) Monitor work

- Block-style scope with automatic begin/sample/end events:

```cpp
GPUMON_SCOPE("training-epoch") {
    // launch kernels, do work...
}
```

- RAII object (equivalent):

```cpp
{
    gpumon::ScopedMonitor m{"stage-1"};
    // work...
}
```

- Functional helper:

```cpp
gpumon::monitor("data-load", [&]{
    // work...
});
```

- CUDA kernel macro with auto-timing (synchronous for now):

```cpp
GPUMON_LAUNCH(MyKernel, grid, block, sharedMemBytes, stream, arg1, arg2);
// Also available: GPUMON_LAUNCH_TAGGED("tag", MyKernel, ...)
```

## Public API (summary)

```cpp
namespace gpumon {
  struct InitOptions {
    std::string appName;
    std::string logFilePath;   // empty -> use GPUMON_LOG_DIR
    uint32_t    sampleIntervalMs = 0; // background memory sampling period for scopes (0 = off)
  };

  bool init(const InitOptions&);
  void shutdown();

  class ScopedMonitor { /* RAII scope begin/sample/end */ };
  void monitor(const std::string& name, const std::function<void()>& fn, const std::string& tag = "");
}

// Macros (core):
//   GPUMON_SCOPE(name)
//   GPUMON_SCOPE_TAGGED(name, tag)

// Macros (CUDA backend):
//   GPUMON_LAUNCH(kernel, grid, block, sharedMem, stream, ...)
//   GPUMON_LAUNCH_TAGGED(tag, kernel, grid, block, sharedMem, stream, ...)
```

Notes:
- `GPUMON_SCOPE` optionally samples GPU memory periodically if `sampleIntervalMs > 0` in `InitOptions`.
- Scope end will call the backend `synchronize()` to ensure timing covers in-flight GPU work.

## NDJSON events

One JSON object per line (NDJSON). Key event types:

- Initialization

```json
{
  "type": "init",
  "pid": 1234,
  "app": "my_cuda_app",
  "logPath": "gpumon.log",
  "ts_ns": 1731958400123456
}
```

- Scope lifecycle (begin, sample, end). Examples (two lines shown as text to illustrate NDJSON):

```text
{"type":"scope_begin","pid":1234,"app":"my_cuda_app","name":"epoch_1","ts_ns":1731958400123456,"memory":[{"device":0,"used_mib":1024,"free_mib":8192,"total_mib":9216}]}
{"type":"scope_sample","pid":1234,"app":"my_cuda_app","name":"epoch_1","ts_ns":1731958401123456,"memory":[{"device":0,"used_mib":1100,"free_mib":8116,"total_mib":9216}]}
```

- Scope end includes full timing and final memory snapshot:

```json
{
  "type": "scope_end",
  "pid": 1234,
  "app": "my_cuda_app",
  "name": "epoch_1",
  "ts_start_ns": 1731958400123456,
  "ts_end_ns":   1731958403123456,
  "duration_ns": 3000000,
  "memory": [{"device":0, "used_mib":1110, "free_mib":8106, "total_mib":9216}]
}
```

- CUDA kernel event (from `GPUMON_LAUNCH`):

```json
{
  "type": "kernel",
  "pid": 1234,
  "app": "my_cuda_app",
  "kernel": "vectorAdd",
  "ts_start_ns": 1731958400123456,
  "ts_end_ns":   1731958400126789,
  "duration_ns": 3333,
  "grid": [128, 1, 1],
  "block": [256, 1, 1],
  "shared_mem_bytes": 0,
  "cuda_error": "cudaSuccess"
}
```

- Shutdown

```json
{
  "type": "shutdown",
  "pid": 1234,
  "app": "my_cuda_app",
  "ts_ns": 1731958404123456
}
```

## Data flow and schema (client → crawler → backend)

GPUmon client does not send data to your backend directly. Its sole responsibility is to write NDJSON log lines to a file.

- Client output: NDJSON log file written by the target process.
  - Location is either the explicit InitOptions::logFilePath you provide, or
  - If logFilePath is empty and GPUMON_LOG_DIR is set, the file is written as GPUMON_LOG_DIR/gpumon_<app>_<pid>.log.
- Crawler: A separate gpumon/crawler component is responsible for:
  - Discovering client log files (watching GPUMON_LOG_DIR or configured paths)
  - Validating each NDJSON line against the schema
  - Forwarding well-formed events to your backend/metrics pipeline
- Schema location: gpumon_client/schema contains both human-readable docs and a machine-readable JSON Schema.
  - gpumon_client/schema/README.md — event shapes, examples, notes
  - gpumon_client/schema/ndjson.schema.json — JSON Schema (draft‑07) using oneOf on the "type" field

This separation lets you deploy the lightweight header-only client anywhere, while evolving ingestion/transport inside the crawler without touching application code.

## End-to-end example

```cpp
#include <gpumon/gpumon.hpp>
#include <cuda_runtime.h>

__global__
void vectorAdd(int* a, int* b, int* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  gpumon::InitOptions opts;
  opts.appName = "vector_add_demo";
  opts.logFilePath = "gpumon.log";
  opts.sampleIntervalMs = 5; // try periodic sampling inside scopes

  gpumon::init(opts);

  dim3 grid(4), block(256);
  int *a, *b, *c; // assume allocated/initialized
  // ... allocate and copy ...

  // Monitor a phase
  GPUMON_SCOPE("phase-1") {
    vectorAdd<<<grid, block>>>(a, b, c, 1024);
    cudaDeviceSynchronize();
  }

  // Kernel timing (sync)
  GPUMON_LAUNCH(vectorAdd, grid, block, 0, 0, a, b, c, 1024);

  gpumon::shutdown();
}
```

## CMake integration

The library is header-only. Target name is `gpumon::gpumon`.


```cmake
add_subdirectory(path/to/gpumon/gpumon_client)

add_executable(my_app main.cu)
target_link_libraries(my_app PRIVATE gpumon::gpumon)
set_target_properties(my_app PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_STANDARD 17)
```


Install from this directory:

```bash
cmake -S . -B build
cmake --build build
cmake --install build --prefix <dest>
```

## Building the bundled example

From `gpumon_client` directory:

```bash
cmake -S . -B build
cmake --build build
# Executable name may be gpumon_block_example
``;

On Windows with VS Generator:

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

## Performance notes

- `GPUMON_LAUNCH` performs a `cudaDeviceSynchronize()` to measure kernel duration; this is simpler but adds overhead.
- Scoped monitoring calls backend `synchronize()` only when the scope ends to capture total work in the scope.

Roadmap items include async timing via CUDA events and non-blocking instrumentation.

## Troubleshooting

- No log file is produced
  - Ensure the directory exists and the process can write there
  - Try setting `GPUMON_LOG_DIR` or pass an absolute `logFilePath`

- CUDA compile errors about `dim3` or runtime headers
  - Make sure the translation unit is compiled with NVCC and includes `<cuda_runtime.h>`

- Link/undefined references
  - This is header-only; usually indicates the file isn’t compiled as CUDA (`.cu`) when needed. Enable `CUDA_SEPARABLE_COMPILATION` or place CUDA code in `.cu` sources
