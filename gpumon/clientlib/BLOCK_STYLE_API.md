# GPUmon Block-Style API Guide

This guide demonstrates the different ways to use GPUmon's monitoring API, including block-style syntax similar to Scala.

## Quick Comparison

| Style | Syntax | Use Case |
|-------|--------|----------|
| **GPUMON_SCOPE** | `GPUMON_SCOPE("label") { ... }` | Block of code with multiple operations |
| **ScopedMonitor** | `{ ScopedMonitor m("label"); ... }` | Explicit RAII control |
| **monitor()** | `monitor("label", []{ ... });` | Functional/lambda style |
| **GPUMON_LAUNCH** | `GPUMON_LAUNCH(kernel, ...)` | Single kernel with auto-timing |
| **GPUMON_LAUNCH_LABELED** | `GPUMON_LAUNCH_LABELED("label", kernel, ...)` | Single kernel with custom label |

---

## 1. GPUMON_SCOPE - Block Style 

This is the **block syntax** and the most ergonomic for monitoring code blocks.

### Syntax

```cpp
GPUMON_SCOPE("label") {
    // Your code here
    myKernel<<<grid, block>>>(...);
    cudaDeviceSynchronize();
    
    anotherKernel<<<grid, block>>>(...);
    cudaDeviceSynchronize();
}
// Automatically logs when exiting the block
```

### How It Works

- Uses C++17's `if` statement with initializer
- Creates an RAII object that logs on construction and destruction
- Automatically captures timing and logs `scope_begin` and `scope_end` events

### Example

```cpp
GPUMON_SCOPE("training-epoch-1") {
    forwardPass<<<grid, block>>>(...);
    cudaDeviceSynchronize();
    
    backwardPass<<<grid, block>>>(...);
    cudaDeviceSynchronize();
    
    updateWeights<<<grid, block>>>(...);
    cudaDeviceSynchronize();
}
```

### Log Output

```json
{"type":"scope_begin","pid":1234,"app":"my_app","label":"training-epoch-1","ts_ns":1731958400123456}
{"type":"scope_end","pid":1234,"app":"my_app","label":"training-epoch-1","ts_start_ns":1731958400123456,"ts_end_ns":1731958400456789,"duration_ns":333333}
```

---

## 2. ScopedMonitor - RAII Object

Explicit RAII-based monitoring using a class instance.

### Syntax

```cpp
{
    gpumon::ScopedMonitor monitor("label");
    
    // Your code here
    myKernel<<<grid, block>>>(...);
    cudaDeviceSynchronize();
}
// Monitor destructor logs automatically
```

### When to Use

- When you need explicit control over the monitor object
- When you want to store the monitor as a member variable
- When GPUMON_SCOPE syntax doesn't work in your context

### Example

```cpp
void processData(int* data) {
    gpumon::ScopedMonitor monitor("process-data");
    
    preprocess<<<grid, block>>>(data);
    cudaDeviceSynchronize();
    
    compute<<<grid, block>>>(data);
    cudaDeviceSynchronize();
    
    postprocess<<<grid, block>>>(data);
    cudaDeviceSynchronize();
    
    // Automatically logs when function returns
}
```

---

## 3. monitor() - Lambda/Functional Style

Wrap code in a lambda for functional-style monitoring.

### Syntax

```cpp
gpumon::monitor("label", [&]() {
    // Your code here
    myKernel<<<grid, block>>>(...);
    cudaDeviceSynchronize();
});
```

### When to Use

- When you prefer functional programming style
- When you want to pass monitoring logic around
- When integrating with callback-based APIs

### Example

```cpp
gpumon::monitor("inference-batch", [&]() {
    runInference<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
});
```

---

## 4. GPUMON_LAUNCH - Single Kernel with Auto-Timing

Wraps a single kernel launch with automatic timing and synchronization.

### Syntax

```cpp
GPUMON_LAUNCH(kernelName, grid, block, sharedMem, stream, arg1, arg2, ...);
```

### How It Works

- Captures start timestamp
- Launches the kernel
- Calls `cudaGetLastError()` and `cudaDeviceSynchronize()`
- Captures end timestamp
- Logs kernel event with timing

### Example

```cpp
GPUMON_LAUNCH(vectorAdd, dim3(4), dim3(256), 0, 0, d_a, d_b, d_c, n);
```

### Log Output

```json
{
  "type": "kernel",
  "pid": 1234,
  "app": "my_app",
  "kernel": "vectorAdd",
  "ts_start_ns": 1731958400123456,
  "ts_end_ns": 1731958400126789,
  "duration_ns": 3333,
  "grid": [4, 1, 1],
  "block": [256, 1, 1],
  "shared_mem_bytes": 0,
  "cuda_error": "cudaSuccess"
}
```

---

## 5. GPUMON_LAUNCH_LABELED - Single Kernel with Custom Label

Same as GPUMON_LAUNCH but with an additional label field.

### Syntax

```cpp
GPUMON_LAUNCH_LABELED("label", kernelName, grid, block, sharedMem, stream, arg1, arg2, ...);
```

### When to Use

- When you want to tag a kernel with additional metadata
- When you have multiple launches of the same kernel and want to distinguish them
- When you want to correlate kernels with application-level operations

### Example

```cpp
for (int epoch = 0; epoch < 100; epoch++) {
    std::string label = "epoch-" + std::to_string(epoch);
    GPUMON_LAUNCH_LABELED(label, trainKernel, grid, block, 0, 0, data, weights);
}
```

### Log Output

```json
{
  "type": "kernel",
  "pid": 1234,
  "app": "my_app",
  "kernel": "trainKernel",
  "label": "epoch-42",
  "ts_start_ns": 1731958400123456,
  "ts_end_ns": 1731958400126789,
  "duration_ns": 3333,
  "grid": [128, 1, 1],
  "block": [256, 1, 1],
  "shared_mem_bytes": 0,
  "cuda_error": "cudaSuccess"
}
```

---

## Advanced Usage

### Nested Scopes

You can nest monitoring scopes to create hierarchical traces:

```cpp
GPUMON_SCOPE("training") {
    
    GPUMON_SCOPE("forward-pass") {
        layer1<<<grid, block>>>(...);
        cudaDeviceSynchronize();
        
        layer2<<<grid, block>>>(...);
        cudaDeviceSynchronize();
    }
    
    GPUMON_SCOPE("backward-pass") {
        backprop2<<<grid, block>>>(...);
        cudaDeviceSynchronize();
        
        backprop1<<<grid, block>>>(...);
        cudaDeviceSynchronize();
    }
    
    GPUMON_SCOPE("update") {
        optimizer<<<grid, block>>>(...);
        cudaDeviceSynchronize();
    }
}
```

### Conditional Monitoring

```cpp
if (enableProfiling) {
    GPUMON_SCOPE("expensive-operation") {
        complexKernel<<<grid, block>>>(...);
        cudaDeviceSynchronize();
    }
} else {
    // Run without monitoring overhead (though overhead is minimal)
    complexKernel<<<grid, block>>>(...);
    cudaDeviceSynchronize();
}
```

### Mixing Styles

You can mix different monitoring styles in the same application:

```cpp
GPUMON_SCOPE("main-computation") {
    // Use GPUMON_LAUNCH for individual kernels
    GPUMON_LAUNCH(preprocess, grid, block, 0, 0, data);
    
    // Use lambda style for specific sections
    gpumon::monitor("critical-section", [&]() {
        processKernel<<<grid, block>>>(data);
        cudaDeviceSynchronize();
    });
    
    // Use labeled launch for tagged operations
    GPUMON_LAUNCH_LABELED("final-step", postprocess, grid, block, 0, 0, data);
}
```

---

## Comparison with Scala

In Scala, you might write:

```scala
gpumon.monitor(label = "computation") {
  kernel1()
  kernel2()
}
```

In C++ with GPUmon, the equivalent is:

```cpp
GPUMON_SCOPE("computation") {
    kernel1<<<grid, block>>>(...);
    cudaDeviceSynchronize();
    
    kernel2<<<grid, block>>>(...);
    cudaDeviceSynchronize();
}
```

The key difference is:
- **C++** doesn't support custom block syntax, so we use a macro with RAII
- The visual appearance is very similar
- The behavior is identical: automatic start/end logging

---

## Best Practices

### 1. Use GPUMON_SCOPE for blocks of operations

```cpp
✅ Good
GPUMON_SCOPE("data-processing") {
    kernel1<<<grid, block>>>(...);
    cudaDeviceSynchronize();
    kernel2<<<grid, block>>>(...);
    cudaDeviceSynchronize();
}

❌ Avoid
gpumon::beginRegion("data-processing");
kernel1<<<grid, block>>>(...);
cudaDeviceSynchronize();
kernel2<<<grid, block>>>(...);
cudaDeviceSynchronize();
gpumon::endRegion(); // Easy to forget!
```

### 2. Use GPUMON_LAUNCH for single kernels

```cpp
✅ Good
GPUMON_LAUNCH(simpleKernel, grid, block, 0, 0, data);

❌ Overkill
GPUMON_SCOPE("simple-kernel") {
    simpleKernel<<<grid, block>>>(data);
    cudaDeviceSynchronize();
}
```

### 3. Use meaningful labels

```cpp
✅ Good
GPUMON_SCOPE("training-epoch-5-forward-pass")

❌ Not helpful
GPUMON_SCOPE("temp")
GPUMON_SCOPE("test123")
```

### 4. Nest scopes for hierarchy

```cpp
✅ Good - Creates clear hierarchy
GPUMON_SCOPE("epoch-1") {
    GPUMON_SCOPE("forward") { ... }
    GPUMON_SCOPE("backward") { ... }
}

❌ Flat - Loses structure
GPUMON_SCOPE("epoch-1-forward") { ... }
GPUMON_SCOPE("epoch-1-backward") { ... }
```

---

## Performance Considerations

### Overhead

- **GPUMON_SCOPE**: ~1-2μs per scope (timestamp + file I/O)
- **GPUMON_LAUNCH**: Includes `cudaDeviceSynchronize()` - blocks CPU
- **File I/O**: Buffered writes, flushed per event

### Optimization Tips

1. **Use scopes for coarse-grained operations** (milliseconds+)
2. **Avoid monitoring micro-kernels** (< 10μs) unless necessary
3. **File I/O is synchronized** - use SSD for better performance
4. **Consider sampling** in production (monitor 1 in N iterations)

---

## Complete Example

```cpp
#include <gpumon/gpumon.hpp>

__global__ void process(int* data, int n) { /* ... */ }

int main() {
    gpumon::InitOptions opts;
    opts.appName = "ml_training";
    opts.logFilePath = "training.log";
    gpumon::init(opts);
    
    int* d_data;
    cudaMalloc(&d_data, 1024 * sizeof(int));
    
    // Block-style monitoring (Scala-like)
    GPUMON_SCOPE("training-loop") {
        for (int epoch = 0; epoch < 10; epoch++) {
            std::string label = "epoch-" + std::to_string(epoch);
            
            GPUMON_SCOPE(label) {
                // Individual kernel monitoring
                GPUMON_LAUNCH(process, dim3(4), dim3(256), 0, 0, d_data, 1024);
            }
        }
    }
    
    cudaFree(d_data);
    gpumon::shutdown();
    return 0;
}
```

---

## Log Analysis

After running your application, you'll have an NDJSON log file:

```bash
# View all scopes
grep "scope_begin\|scope_end" training.log | jq .

# Calculate scope durations
grep "scope_end" training.log | jq '.duration_ns / 1000000' # Convert to ms

# Find slow kernels
grep "kernel" training.log | jq 'select(.duration_ns > 1000000)'
```

---

## Migration Guide

### From beginRegion/endRegion

**Before:**
```cpp
gpumon::beginRegion("compute");
// ... code ...
gpumon::endRegion();
```

**After:**
```cpp
GPUMON_SCOPE("compute") {
    // ... code ...
}
```

### From Manual Timing

**Before:**
```cpp
auto start = std::chrono::steady_clock::now();
myKernel<<<grid, block>>>(...);
cudaDeviceSynchronize();
auto end = std::chrono::steady_clock::now();
// ... manual logging ...
```

**After:**
```cpp
GPUMON_LAUNCH(myKernel, grid, block, 0, 0, ...);
```

---

## Troubleshooting

**Q: GPUMON_SCOPE doesn't compile**

A: Ensure you're using C++17 or later:
```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
```

**Q: Nested scopes don't appear in logs**

A: Check that `gpumon::init()` was called successfully and the log file is writable.

**Q: Want to disable monitoring in production**

A: Define a no-op macro in your production builds:
```cpp
#ifdef PRODUCTION
#undef GPUMON_SCOPE
#define GPUMON_SCOPE(label) if (true)
#endif
```
