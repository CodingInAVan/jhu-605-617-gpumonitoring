#ifndef GPUMON_HPP
#define GPUMON_HPP

// 1. Backend Auto-Detection
#if !defined(GPUMON_BACKEND_CUDA) && !defined(GPUMON_BACKEND_OPENCL)
    #if defined(__CUDACC__)
        #define GPUMON_BACKEND_CUDA
    #endif
#endif

// 2. Core (InitOptions, State, Logging)
#include "core/common.hpp"

// 3. Backend Selection
#if defined(GPUMON_BACKEND_CUDA)
    #include "backends/cuda.hpp"
#elif defined(GPUMON_BACKEND_OPENCL)
    #error "GPUMON: OpenCL backend not implemented."
#else
    #error "GPUMON Error: No backend selected. Define GPUMON_BACKEND_CUDA."
#endif

// 4. Monitor (ScopedMonitor, Init, Shutdown)
#include "core/monitor.hpp"

#endif
