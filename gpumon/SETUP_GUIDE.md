# GPUmon Setup Guide

This guide explains how to integrate the GPUmon client library into your CUDA application and configure the crawler to collect enriched metrics.

## Architecture Overview

```
┌─────────────────────────────────────────┐
│  Your CUDA Application                  │
│  (with gpumon.hpp included)             │
│                                          │
│  - Logs kernel launches, regions, scopes│
│  - Writes NDJSON to log files           │
└────────────┬────────────────────────────┘
             │ Writes to
             │ $GPUMON_LOG_DIR/gpumon_{appName}.log
             ▼
┌─────────────────────────────────────────┐
│  Log Directory                           │
│  (configured in crawler config)         │
│                                          │
│  - gpumon_training_app.log         │
│  - gpumon_inference_app.log        │
└────────────┬────────────────────────────┘
             │ Reads from
             │ (scans for gpumon_*.log)
             ▼
┌─────────────────────────────────────────┐
│  GPUmon Crawler                         │
│  (NVML + log reader)                    │
│                                          │
│  - Collects GPU metrics via NVML       │
│  - Reads clientlib logs                 │
│  - Correlates by PID + timestamp        │
│  - Sends enriched metrics to backend    │
└─────────────────────────────────────────┘
```

## Step 1: Set Up Log Directory

Choose a directory where your application will write logs and the crawler will read them.

**Important**: You must explicitly configure this directory. There is no automatic discovery.

### Linux/macOS:
```bash
# Create log directory
mkdir -p /var/log/gpumon

# Set environment variable for your application
export GPUMON_LOG_DIR=/var/log/gpumon
```

### Windows (PowerShell):
```powershell
# Create log directory
New-Item -ItemType Directory -Path C:\Logs\gpumon -Force

# Set environment variable for your application
$env:GPUMON_LOG_DIR="C:\Logs\gpumon"
```

### Windows (cmd):
```cmd
set GPUMON_LOG_DIR=C:\Logs\gpumon
mkdir "%GPUMON_LOG_DIR%"
```

## Step 2: Integrate Client Library in Your Application

### Copy the Header

Copy `clientlib/include/gpumon/gpumon.hpp` to your project's include directory.

### Initialize GPUmon

```cpp
#include <gpumon/gpumon.hpp>

int main() {
    // Initialize with app name only - log path auto-determined from GPUMON_LOG_DIR
    gpumon::InitOptions opts;
    opts.appName = "my_training_app";
    opts.logFilePath = "";  // Empty = use GPUMON_LOG_DIR/{appName}.log

    if (!gpumon::init(opts)) {
        std::cerr << "Failed to initialize gpumon" << std::endl;
        return 1;
    }

    // Your CUDA application code...

    gpumon::shutdown();
    return 0;
}
```

**Note:** If `logFilePath` is empty:
- If `GPUMON_LOG_DIR` is set: writes to `$GPUMON_LOG_DIR/gpumon_{appName}_{pid}.log`
- If `GPUMON_LOG_DIR` is not set: operates in silent mode (no logging)

### Instrument Your Code

#### Basic Kernel Monitoring:
```cpp
// Replace this:
myKernel<<<grid, block, 0, 0>>>(args...);

// With this:
GPUMON_SCOPE(myKernel, grid, block, 0, 0, args...);
```

#### With Custom Tags for Categorization:
```cpp
GPUMON_LAUNCH_TAGGED("training", myKernel, grid, block, 0, 0, args...);
```

#### Monitoring Code Regions:
```cpp
gpumon::beginRegion("epoch_1", "training");
// ... multiple kernel launches ...
gpumon::endRegion("training");
```

#### Scoped Monitoring (RAII):
```cpp
GPUMON_SCOPE("forward_pass") {
    GPUMON_LAUNCH(matmul, grid, block, 0, 0, A, B, C);
    GPUMON_LAUNCH(activation, grid, block, 0, 0, C);
}
```

#### With Tags:
```cpp
GPUMON_SCOPE_TAGGED("training_loop", "production") {
    gpumon::beginRegion("data_loading", "io_intensive");
    // ... kernels ...
    gpumon::endRegion("io_intensive");
}
```

### Example Application:

```cpp
#include <gpumon/gpumon.hpp>
#include <cuda_runtime.h>
#include <iostream>

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    // Initialize GPUmon (uses GPUMON_LOG_DIR)
    gpumon::InitOptions opts;
    opts.appName = "vector_add";
    opts.logFilePath = "";  // Auto-determined

    if (!gpumon::init(opts)) {
        std::cerr << "Failed to initialize gpumon\n";
        return 1;
    }

    const int N = 1024 * 1024;
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    dim3 grid(4096);
    dim3 block(256);

    // Monitor with tag for categorization
    GPUMON_SCOPE_TAGGED("computation", "benchmark") {
        GPUMON_LAUNCH_TAGGED("compute", vectorAdd, grid, block, 0, 0, d_a, d_b, d_c, N);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    gpumon::shutdown();
    return 0;
}
```

## Step 3: Build and Run Your Application

### CMake Integration:

```cmake
# Include gpumon header
include_directories(path/to/clientlib/include)

# Your CUDA target
add_executable(my_app main.cu)
set_target_properties(my_app PROPERTIES CUDA_STANDARD 17)
```

### Run Your Application:

```bash
# Set log directory
export GPUMON_LOG_DIR=/tmp/gpumon_logs

# Run your app
./my_app
```

This will create a log file like: `/tmp/gpumon_logs/vector_add_12345.log`

## Step 4: Configure and Run the Crawler

The crawler must be explicitly configured to know where to find log files.

### Option 1: Interactive Setup (First Run)

```bash
./crawler
```

You'll be prompted to configure:
1. API key
2. Backend URL
3. **Log directory** (e.g., `/var/log/gpumon` or `C:\Logs\gpumon`)

This saves configuration to:
- Linux/macOS: `~/.gpu-crawler/config.json`
- Windows: `%APPDATA%\gpu-crawler\config.json`

### Option 2: Environment Variables

```bash
export GPUMON_LOG_DIR=/var/log/gpumon
export GPU_BACKEND_URL=http://localhost:8080
export GPU_API_KEY=your_api_key_here

./crawler
```

### Option 3: Command Line Arguments

```bash
./crawler --log-dir=/var/log/gpumon --backend-url=http://localhost:8080 --api-key=your_key
```

### Option 4: Edit Config File Directly

Edit `~/.gpu-crawler/config.json` (or `%APPDATA%\gpu-crawler\config.json` on Windows):

```json
{
  "backendUrl": "http://localhost:8080",
  "apiKey": "your_api_key",
  "logDirectory": "/var/log/gpumon"
}
```

### Example Output:

```
Scanning for gpumon logs in: /var/log/gpumon
Will monitor 2 clientlib log file(s) for enrichment:
  - /var/log/gpumon/gpumon_vector_add_12345.log
  - /var/log/gpumon/gpumon_training_app_67890.log
Found 1 NVIDIA device(s).
Initialized log reader for: /var/log/gpumon/gpumon_vector_add_12345.log
[ClientLog] Read 15 new events. Total cached: 15
...
```

## Step 5: Verify Enriched Metrics

The crawler will send enriched process metrics to your backend with fields like:

```json
{
  "timestamp": "2025-01-18T10:30:45Z",
  "hostname": "gpu-server-01",
  "gpuId": "GPU-xxx",
  "metricType": "process",
  "pid": 12345,
  "processName": "my_app",
  "processUsedMemoryMiB": 2048,

  // Enriched fields from clientlib:
  "appName": "vector_add",
  "kernelName": "vectorAdd",
  "scopeName": "computation",
  "tag": "benchmark"
}
```

## Environment Variables Reference

### Shared (Both Client Library and Crawler):

| Variable | Description | Example |
|----------|-------------|---------|
| `GPUMON_LOG_DIR` | Directory for log files | `/tmp/gpumon_logs` |

### Client Library Only:

| Variable | Description | Default |
|----------|-------------|---------|
| (none currently) | | |

### Crawler Only:

| Variable | Description | Default |
|----------|-------------|---------|
| `GPUMON_CLIENT_LOGS` | Explicit log file paths (fallback) | (empty) |
| `GPU_BACKEND_URL` | Backend server URL | `http://localhost:8080` |
| `GPU_API_KEY` | API key for backend | (required) |
| `METRICS_INTERVAL_MS` | How often to send metrics | `5000` |
| `METRICS_SAMPLE_INTERVAL_MS` | How often to sample GPU | `500` |

## Troubleshooting

### Problem: Clientlib log file not created

**Check:**
1. `GPUMON_LOG_DIR` is set and directory exists
2. Application has write permissions to the directory
3. Check if `gpumon::init()` returned `true`

**Debug:**
```cpp
if (!gpumon::init(opts)) {
    std::cerr << "GPUmon init failed. Check GPUMON_LOG_DIR: "
              << std::getenv("GPUMON_LOG_DIR") << std::endl;
}
```

### Problem: Crawler not finding log files

**Check:**
1. `GPUMON_LOG_DIR` points to the same directory as your application
2. Log files exist with `.log` extension
3. Crawler has read permissions

**Debug:**
```bash
# Verify log directory
echo $GPUMON_LOG_DIR
ls -la $GPUMON_LOG_DIR

# Run crawler with verbose output
./crawler
```

### Problem: No enrichment in metrics

**Check:**
1. Log files are being written (check file size)
2. PID in log matches running process
3. Timestamps are recent

**Debug:**
Look for crawler output:
```
Initialized log reader for: /tmp/gpumon_logs/my_app_1234.log
Process on GPU 0: PID=1234, Name=my_app, App=my_app, Kernel=vectorAdd, Tag=benchmark
```

## Advanced Configuration

### Multiple Applications on Same Machine

Each application automatically gets its own log file:
```
/tmp/gpumon_logs/
  ├── training_app_1234.log
  ├── inference_app_5678.log
  └── preprocess_app_9012.log
```

The crawler will monitor all of them simultaneously.

### Custom Log Paths (Override)

If you need specific log paths instead of `GPUMON_LOG_DIR`:

**Application:**
```cpp
opts.logFilePath = "/custom/path/my_app.log";
```

**Crawler:**
```bash
export GPUMON_CLIENT_LOGS="/custom/path/my_app.log,/other/path/app2.log"
./crawler
```

### Log Rotation

For long-running applications, consider log rotation:

```bash
# In your application wrapper script
export GPUMON_LOG_DIR=/tmp/gpumon_logs

# Rotate logs daily
logrotate /etc/logrotate.d/gpumon
```

## Best Practices

1. **Use a dedicated log directory:** Keep GPUmon logs separate from application logs
2. **Use tags for categories:** Tag kernels with "training", "inference", "preprocessing", etc.
3. **Monitor multiple apps:** Run one crawler per machine to monitor all applications
4. **Clean up old logs:** Set up log rotation for long-running systems
5. **Use meaningful app names:** Make it easy to identify applications in metrics
6. **Scope large regions:** Use `GPUMON_SCOPE` for logical boundaries (epochs, batches, etc.)

## Summary

1. **Set `GPUMON_LOG_DIR`** environment variable
2. **Include `gpumon.hpp`** in your CUDA application
3. **Initialize with empty `logFilePath`** to use auto-discovery
4. **Instrument kernels** with `GPUMON_LAUNCH` or `GPUMON_LAUNCH_TAGGED`
5. **Run crawler** with same `GPUMON_LOG_DIR`
6. **Enjoy enriched metrics** in your backend!
