#ifndef GPUMON_HPP
#define GPUMON_HPP

#include <cuda_runtime.h>
#include <string>
#include <fstream>
#include <mutex>
#include <chrono>
#include <cstdint>
#include <sstream>
#include <functional>
#include <iomanip>
#include <utility>
#include <thread>
#include <atomic>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace gpumon {

// ============================================================================
// Public API Structures
// ============================================================================

struct InitOptions {
    std::string appName;
    std::string logFilePath;   // where NDJSON logs are written (can be empty to use default)
    // How often to collect samples within a scope (in milliseconds).
    // Set to 0 to disable background sampling.
    uint32_t sampleIntervalMs = 0;
};

// ============================================================================
// Internal State
// ============================================================================

namespace detail {

    struct State {
        std::string appName;
        std::ofstream logFile;
        std::mutex logMutex;
        int32_t pid;
        bool initialized;
        uint32_t sampleIntervalMs; // Store the interval

        State() : pid(0), initialized(false), sampleIntervalMs(0) {}
    };

    inline State& getState() {
        static State state;
        return state;
    }

    inline std::mutex& getInitMutex() {
        static std::mutex m;
        return m;
    }

    inline int32_t getPid() {
    #ifdef _WIN32
        return static_cast<int32_t>(_getpid());
    #else
        return static_cast<int32_t>(getpid());
    #endif
    }

    inline std::string getDefaultLogPath(const std::string& appName, const int32_t pid) {
        const char* logDirEnv = std::getenv("GPUMON_LOG_DIR");

        if (!logDirEnv || logDirEnv[0] == '\0') return "";

        const std::string logDir = logDirEnv;
        std::ostringstream oss;
        oss << logDir;

    #ifdef _WIN32
        if (!logDir.empty() && logDir.back() != '\\' && logDir.back() != '/') {
            oss << "\\";
        }
    #else
        if (!logDir.empty() && logDir.back() != '/') {
            oss << "/";
        }
    #endif

        oss << "gpumon_" << appName << "_" << pid << ".log";
        return oss.str();
    }

    inline int64_t getTimestampNs() {
        auto now = std::chrono::steady_clock::now();
        auto duration = now.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    }

    // Structure to hold GPU memory snapshot
    struct MemorySnapshot {
        size_t freeMiB = 0;
        size_t totalMiB = 0;
        size_t usedMiB = 0;
        int deviceId = 0;
        bool valid = false;
    };

    // Query current GPU memory usage across all visible devices
    inline std::vector<MemorySnapshot> getMemorySnapshots() {
        std::vector<MemorySnapshot> snapshots;

        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) {
            return snapshots;
        }

        // Save current device to restore later
        int currentDevice = -1;
        err = cudaGetDevice(&currentDevice);
        if (err != cudaSuccess) {
            // If we can't get current device, default to 0
            currentDevice = 0;
        }

        for (int dev = 0; dev < deviceCount; ++dev) {
            // Set device before querying memory
            err = cudaSetDevice(dev);
            if (err != cudaSuccess) continue;

            size_t freeMem = 0, totalMem = 0;

            if (cudaMemGetInfo(&freeMem, &totalMem) == cudaSuccess) {
                MemorySnapshot snap;
                snap.deviceId = dev;
                snap.freeMiB = freeMem / (1024 * 1024);
                snap.totalMiB = totalMem / (1024 * 1024);
                snap.usedMiB = snap.totalMiB - snap.freeMiB;
                snap.valid = true;
                snapshots.push_back(snap);
            }
        }

        // Restore original device (ignore errors on restore)
        if (currentDevice >= 0) {
            cudaSetDevice(currentDevice);
        }

        return snapshots;
    }

    inline void writeLogLine(const std::string& jsonLine) {
        State& state = getState();
        std::lock_guard lock(state.logMutex);
        if (state.logFile.is_open()) {
            state.logFile << jsonLine << '\n';
            state.logFile.flush();
        }
    }

    inline std::string escapeJson(const std::string& str) {
        std::ostringstream oss;
        for (const char c : str) {
            // (Simplified for brevity, same as before)
            switch (c) {
                case '"':  oss << "\\\""; break;
                case '\\': oss << "\\\\"; break;
                default:   oss << c; break;
            }
        }
        return oss.str();
    }

    // Helper to write memory array to JSON stream
    inline void writeMemoryJson(std::ostringstream& oss, const std::vector<MemorySnapshot>& snapshots) {
        if (snapshots.empty()) return;
        oss << ",\"memory\":[";
        for (size_t i = 0; i < snapshots.size(); ++i) {
            if (i > 0) oss << ",";
            const auto& mem = snapshots[i];
            oss << "{\"device\":" << mem.deviceId
                << ",\"used_mib\":" << mem.usedMiB
                << ",\"free_mib\":" << mem.freeMiB
                << ",\"total_mib\":" << mem.totalMiB
                << "}";
        }
        oss << "]";
    }

} // namespace detail

// ============================================================================
// Public API Functions
// ============================================================================

inline bool init(const InitOptions& opts) {
    auto& state = detail::getState();
    std::lock_guard initLock(detail::getInitMutex());

    if (state.initialized) {
        return false; // Already initialized
    }

    state.appName = opts.appName;
    state.pid = detail::getPid();
    state.sampleIntervalMs = opts.sampleIntervalMs;

    std::string logPath = opts.logFilePath;
    if (logPath.empty()) {
        logPath = detail::getDefaultLogPath(state.appName, state.pid);
    }

    // If no log path configured, initialize in silent mode (no logging)
    if (logPath.empty()) {
        state.initialized = true;
        return true;
    }

    state.logFile.open(logPath, std::ios::out | std::ios::app);
    if (!state.logFile.is_open()) {
        // Failed to open log file - operate in silent mode
        state.initialized = true;
        return true;
    }

    state.initialized = true;

    // Write initialization event with log path for reference
    std::ostringstream oss;
    oss << R"({"type":"init",)"
        << "\"pid\":" << state.pid << ","
        << R"("app":")" << detail::escapeJson(state.appName) << "\","
        << R"("logPath":")" << detail::escapeJson(logPath) << "\","
        << "\"ts_ns\":" << detail::getTimestampNs()
        << "}";
    detail::writeLogLine(oss.str());

    return true;
}

inline void shutdown() {
    detail::State& state = detail::getState();
    
    if (!state.initialized || !state.logFile.is_open()) {
        return;
    }
    
    // Write shutdown event
    std::ostringstream oss;
    oss << R"({"type":"shutdown",)"
        << "\"pid\":" << state.pid << ","
        << R"("app":")" << detail::escapeJson(state.appName) << "\","
        << "\"ts_ns\":" << detail::getTimestampNs()
        << "}";
    detail::writeLogLine(oss.str());
    
    std::lock_guard lock(state.logMutex);
    state.initialized = false;
    if (state.logFile.is_open()) {
        state.logFile.close();
    }
}

// ============================================================================
// RAII Scoped Monitor (recommended for block-style monitoring)
// ============================================================================

class ScopedMonitor {
public:
    explicit ScopedMonitor(std::string  name, std::string  tag = "")
        : name_(std::move(name)), tag_(std::move(tag)), tsStart_(detail::getTimestampNs()) {
        const detail::State& state = detail::getState();
        if (!state.initialized) return;

        // Log Scope Begin
        logScopeEvent("scope_begin", tsStart_);

        // Start sampling thread if configured
        if (state.sampleIntervalMs > 0) {
            samplerThread_ = std::thread(&ScopedMonitor::samplingLoop, this, state.sampleIntervalMs);
        }
    }

    ~ScopedMonitor() {
        const detail::State& state = detail::getState();

        // Stop Sampling Thread
        if (samplerThread_.joinable()) {
            stopSampling_ = true;
            samplerThread_.join();
        }

        if (!state.initialized) return;

        // Log Scope End
        logScopeEvent("scope_end", detail::getTimestampNs());
    }

    // Non-copyable, non-movable
    ScopedMonitor(const ScopedMonitor&) = delete;
    ScopedMonitor& operator=(const ScopedMonitor&) = delete;

private:
    std::string name_;
    std::string tag_;
    int64_t tsStart_;

    // Threading member variables
    std::atomic<bool> stopSampling_;
    std::thread samplerThread_;

    void logScopeEvent(const char* type, int64_t timestamp) {
        const detail::State& state = detail::getState();
        const auto memSnapshots = detail::getMemorySnapshots();

        std::ostringstream oss;
        oss << "{\"type\":\"" << type << "\","
            << "\"pid\":" << state.pid << ","
            << R"("app":")" << detail::escapeJson(state.appName) << "\","
            << R"("name":")" << detail::escapeJson(name_) << "\"";

        if (!tag_.empty()) oss << R"(,"tag":")" << detail::escapeJson(tag_) << "\"";

        if (std::string(type) == "scope_end") {
            oss << ",\"ts_start_ns\":" << tsStart_
                << ",\"ts_end_ns\":" << timestamp
                << ",\"duration_ns\":" << (timestamp - tsStart_);
        } else {
            oss << ",\"ts_ns\":" << timestamp;
        }

        detail::writeMemoryJson(oss, memSnapshots);
        oss << "}";
        detail::writeLogLine(oss.str());
    }

    void samplingLoop(uint32_t intervalMs) {
        const detail::State& state = detail::getState();

        while (!stopSampling_) {
            // Sleep first (or sleep at end).
            // We usually sleep first to avoid capturing a sample immediately after scope_begin
            std::this_thread::sleep_for(std::chrono::milliseconds(intervalMs));

            if (stopSampling_) break;

            auto memSnapshots = detail::getMemorySnapshots();
            const int64_t now = detail::getTimestampNs();

            std::ostringstream oss;
            oss << R"({"type":"scope_sample",)"
                << "\"pid\":" << state.pid << ","
                << R"("app":")" << detail::escapeJson(state.appName) << "\","
                << R"("name":")" << detail::escapeJson(name_) << "\"";

            if (!tag_.empty()) oss << R"(,"tag":")" << detail::escapeJson(tag_) << "\"";

            oss << ",\"ts_ns\":" << now;
            detail::writeMemoryJson(oss, memSnapshots);
            oss << "}";

            detail::writeLogLine(oss.str());
        }
    }
};

// Lambda-based wrapper for functional-style monitoring
inline void monitor(const std::string& name, std::function<void()> fn, const std::string& tag = "") {
    ScopedMonitor monitor(name, tag);
    fn();
}

// Internal kernel logging function
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

    if (!tag.empty()) {
        oss << R"(,"tag":")" << escapeJson(tag) << "\"";
    }

    oss << R"(,"cuda_error":")" << escapeJson(cudaError) << "\""
        << "}";
    writeLogLine(oss.str());
}

inline const char* getCudaErrorString(const cudaError_t error) {
    return cudaGetErrorString(error);
}

} // namespace detail

} // namespace gpumon

// ============================================================================
// GPUMON_LAUNCH Macros
// ============================================================================

// wraps a single kernel launch
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

// Block-style macro using RAII
// Usage: GPUMON_SCOPE("name") { /* your code with kernel launches */ }
// Usage with tag: GPUMON_SCOPE_TAGGED("name", "tag") { /* your code */ }
#define GPUMON_SCOPE(name) if (gpumon::ScopedMonitor _scope{name}; true)

#define GPUMON_SCOPE_TAGGED(name, tag) if (gpumon::ScopedMonitor _scope{name, tag}; true)

#endif // GPUMON_HPP
