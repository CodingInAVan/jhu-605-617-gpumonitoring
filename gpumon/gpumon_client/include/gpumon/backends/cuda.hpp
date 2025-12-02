#ifndef GPUMON_CUDA_HPP
#define GPUMON_CUDA_HPP
#include <cuda_runtime.h>
#include <vector>
#include "../core/common.hpp"

namespace gpumon {
    namespace backend {
        inline void initialize() {}

        inline void shutdown() {}

        inline void synchronize() {
            cudaDeviceSynchronize();
        }

        inline std::vector<detail::MemorySnapshot> get_memory_snapshots() {
            std::vector<detail::MemorySnapshot> snapshots;

            int deviceCount = 0;
            if (cudaError_t err = cudaGetDeviceCount(&deviceCount); err != cudaSuccess || deviceCount == 0) return snapshots;

            // Save current device to restore later (polite behavior)
            int currentDevice = -1;
            cudaGetDevice(&currentDevice);

            for (int dev = 0; dev < deviceCount; ++dev) {
                if (cudaSetDevice(dev) != cudaSuccess) continue;

                size_t free = 0, total = 0;
                if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
                    detail::MemorySnapshot snap;
                    snap.deviceId = dev;
                    snap.freeMiB = free / (1024 * 1024);
                    snap.totalMiB = total / (1024 * 1024);
                    snap.usedMiB = snap.totalMiB - snap.freeMiB;
                    snap.valid = true;
                    snapshots.push_back(snap);
                }
            }

            // Restore context
            if (currentDevice >= 0) cudaSetDevice(currentDevice);

            return snapshots;
        }

    }

    namespace detail {
        inline void logKernelEvent(
        const std::string& kernelName,
        const int64_t tsStartNs,
        const int64_t tsEndNs,
        const dim3& grid,
        const dim3& block,
        const size_t sharedMemBytes,
        const std::string& cudaError,
        const std::string& tag = "")
        {
            State& state = getState();
            if (!state.initialized) return;

            std::ostringstream oss;
            oss << R"({"type":"kernel",)"
                << "\"pid\":" << state.pid << ","
                << R"("app":")" << escapeJson(state.appName) << "\","
                << R"("kernel":")" << escapeJson(kernelName) << "\","
                << "\"ts_start_ns\":" << tsStartNs << ","
                << "\"ts_end_ns\":" << tsEndNs << ","
                << "\"duration_ns\":" << (tsEndNs - tsStartNs) << ","
                << "\"grid\":[" << grid.x << "," << grid.y << "," << grid.z << "],"
                << "\"block\":[" << block.x << "," << block.y << "," << block.z << "],"
                << "\"shared_mem_bytes\":" << sharedMemBytes;

            if (!tag.empty()) oss << R"(,"tag":")" << escapeJson(tag) << "\"";

            oss << R"(,"cuda_error":")" << escapeJson(cudaError) << "\"}";
            writeLogLine(oss.str());
        }

        inline const char* getCudaErrorString(const cudaError_t error) {
            return ::cudaGetErrorString(error);
        }
    }
}

#define GPUMON_LAUNCH(kernel, grid, block, sharedMem, stream, ...) \
    do { \
        int64_t _ts = gpumon::detail::getTimestampNs(); \
        kernel<<<grid, block, sharedMem, stream>>>(__VA_ARGS__); \
        cudaError_t _err = cudaGetLastError(); \
        cudaDeviceSynchronize(); \
        int64_t _te = gpumon::detail::getTimestampNs(); \
        gpumon::detail::logKernelEvent(#kernel, _ts, _te, grid, block, sharedMem, gpumon::detail::getCudaErrorString(_err)); \
    } while(0)

// wraps a single kernel launch with custom tag
#define GPUMON_LAUNCH_TAGGED(tag, kernel, grid, block, sharedMem, stream, ...) \
    do { \
        int64_t _ts = gpumon::detail::getTimestampNs(); \
        kernel<<<grid, block, sharedMem, stream>>>(__VA_ARGS__); \
        cudaError_t _err = cudaGetLastError(); \
        cudaDeviceSynchronize(); \
        int64_t _te = gpumon::detail::getTimestampNs(); \
        gpumon::detail::logKernelEvent(#kernel, _ts, _te, grid, block, sharedMem, gpumon::detail::getCudaErrorString(_err), tag); \
    } while(0)

#endif
