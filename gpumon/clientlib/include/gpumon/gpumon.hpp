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
    std::string logFilePath;   // where NDJSON logs are written
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
    
    State() : pid(0), initialized(false) {}
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

inline int64_t getTimestampNs() {
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

inline void writeLogLine(const std::string& jsonLine) {
    State& state = getState();
    std::lock_guard<std::mutex> lock(state.logMutex);
    if (state.logFile.is_open()) {
        state.logFile << jsonLine << '\n';
        state.logFile.flush();
    }
}

inline std::string escapeJson(const std::string& str) {
    std::ostringstream oss;
    for (char c : str) {
        switch (c) {
            case '"':  oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (c >= 0x20) {
                    oss << c;
                } else {
                    unsigned char uc = static_cast<unsigned char>(c);
                    oss << "\\u"
                        << std::hex
                        << std::setw(4)
                        << std::setfill('0')
                        << static_cast<int>(uc);
                }
        }
    }
    return oss.str();
}

} // namespace detail

// ============================================================================
// Public API Functions
// ============================================================================

inline bool init(const InitOptions& opts) {
    // must call init from a single thread
    auto& state = detail::getState();
    auto& initMutex = detail::getInitMutex();

    std::lock_guard<std::mutex> initLock(initMutex);
    
    if (state.initialized) {
        return false; // Already initialized
    }
    
    state.appName = opts.appName;
    state.pid = detail::getPid();
    
    state.logFile.open(opts.logFilePath, std::ios::out | std::ios::app);
    if (!state.logFile.is_open()) {
        return false;
    }
    
    state.initialized = true;
    
    // Write initialization event
    std::ostringstream oss;
    oss << "{\"type\":\"init\","
        << "\"pid\":" << state.pid << ","
        << "\"app\":\"" << detail::escapeJson(state.appName) << "\","
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
    oss << "{\"type\":\"shutdown\","
        << "\"pid\":" << state.pid << ","
        << "\"app\":\"" << detail::escapeJson(state.appName) << "\","
        << "\"ts_ns\":" << detail::getTimestampNs()
        << "}";
    detail::writeLogLine(oss.str());
    
    std::lock_guard<std::mutex> lock(state.logMutex);
    state.initialized = false;
    if (state.logFile.is_open()) {
        state.logFile.close();
    }
}

inline void beginRegion(const std::string& name, const std::string& tag = "") {
    detail::State& state = detail::getState();
    if (!state.initialized) return;

    std::ostringstream oss;
    oss << "{\"type\":\"region_begin\","
        << "\"pid\":" << state.pid << ","
        << "\"app\":\"" << detail::escapeJson(state.appName) << "\","
        << "\"name\":\"" << detail::escapeJson(name) << "\"";

    if (!tag.empty()) {
        oss << ",\"tag\":\"" << detail::escapeJson(tag) << "\"";
    }

    oss << ",\"ts_ns\":" << detail::getTimestampNs()
        << "}";
    detail::writeLogLine(oss.str());
}

inline void endRegion(const std::string& tag = "") {
    detail::State& state = detail::getState();
    if (!state.initialized) return;

    std::ostringstream oss;
    oss << "{\"type\":\"region_end\","
        << "\"pid\":" << state.pid << ","
        << "\"app\":\"" << detail::escapeJson(state.appName) << "\"";

    if (!tag.empty()) {
        oss << ",\"tag\":\"" << detail::escapeJson(tag) << "\"";
    }

    oss << ",\"ts_ns\":" << detail::getTimestampNs()
        << "}";
    detail::writeLogLine(oss.str());
}

// ============================================================================
// RAII Scoped Monitor (recommended for block-style monitoring)
// ============================================================================

class ScopedMonitor {
public:
    explicit ScopedMonitor(const std::string& name, const std::string& tag = "")
        : name_(name), tag_(tag), tsStart_(detail::getTimestampNs()) {
        detail::State& state = detail::getState();
        if (!state.initialized) return;

        std::ostringstream oss;
        oss << "{\"type\":\"scope_begin\","
            << "\"pid\":" << state.pid << ","
            << "\"app\":\"" << detail::escapeJson(state.appName) << "\","
            << "\"name\":\"" << detail::escapeJson(name_) << "\"";

        if (!tag_.empty()) {
            oss << ",\"tag\":\"" << detail::escapeJson(tag_) << "\"";
        }

        oss << ",\"ts_ns\":" << tsStart_
            << "}";
        detail::writeLogLine(oss.str());
    }

    ~ScopedMonitor() {
        detail::State& state = detail::getState();
        if (!state.initialized) return;

        int64_t tsEnd = detail::getTimestampNs();
        std::ostringstream oss;
        oss << "{\"type\":\"scope_end\","
            << "\"pid\":" << state.pid << ","
            << "\"app\":\"" << detail::escapeJson(state.appName) << "\","
            << "\"name\":\"" << detail::escapeJson(name_) << "\"";

        if (!tag_.empty()) {
            oss << ",\"tag\":\"" << detail::escapeJson(tag_) << "\"";
        }

        oss << ",\"ts_start_ns\":" << tsStart_ << ","
            << "\"ts_end_ns\":" << tsEnd << ","
            << "\"duration_ns\":" << (tsEnd - tsStart_)
            << "}";
        detail::writeLogLine(oss.str());
    }

    // Non-copyable, non-movable
    ScopedMonitor(const ScopedMonitor&) = delete;
    ScopedMonitor& operator=(const ScopedMonitor&) = delete;
    ScopedMonitor(ScopedMonitor&&) = delete;
    ScopedMonitor& operator=(ScopedMonitor&&) = delete;

private:
    std::string name_;
    std::string tag_;
    int64_t tsStart_;
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
    int64_t tsStartNs,
    int64_t tsEndNs,
    const dim3& grid,
    const dim3& block,
    size_t sharedMemBytes,
    const std::string& cudaError,
    const std::string& tag = "")
{
    State& state = getState();
    if (!state.initialized) return;

    std::ostringstream oss;
    oss << "{\"type\":\"kernel\","
        << "\"pid\":" << state.pid << ","
        << "\"app\":\"" << escapeJson(state.appName) << "\","
        << "\"kernel\":\"" << escapeJson(kernelName) << "\","
        << "\"ts_start_ns\":" << tsStartNs << ","
        << "\"ts_end_ns\":" << tsEndNs << ","
        << "\"duration_ns\":" << (tsEndNs - tsStartNs) << ","
        << "\"grid\":[" << grid.x << "," << grid.y << "," << grid.z << "],"
        << "\"block\":[" << block.x << "," << block.y << "," << block.z << "],"
        << "\"shared_mem_bytes\":" << sharedMemBytes;

    if (!tag.empty()) {
        oss << ",\"tag\":\"" << escapeJson(tag) << "\"";
    }

    oss << ",\"cuda_error\":\"" << escapeJson(cudaError) << "\""
        << "}";
    writeLogLine(oss.str());
}

inline const char* getCudaErrorString(cudaError_t error) {
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
        int64_t _gpumon_ts_start = gpumon::detail::getTimestampNs(); \
        kernel<<<grid, block, sharedMem, stream>>>(__VA_ARGS__); \
        cudaError_t _gpumon_err = cudaGetLastError(); \
        cudaError_t _gpumon_sync_err = cudaDeviceSynchronize(); \
        if (_gpumon_sync_err != cudaSuccess && _gpumon_err == cudaSuccess) { \
            _gpumon_err = _gpumon_sync_err; \
        } \
        int64_t _gpumon_ts_end = gpumon::detail::getTimestampNs(); \
        gpumon::detail::logKernelEvent( \
            #kernel, \
            _gpumon_ts_start, \
            _gpumon_ts_end, \
            grid, \
            block, \
            sharedMem, \
            gpumon::detail::getCudaErrorString(_gpumon_err) \
        ); \
    } while(0)

#define GPUMON_LAUNCH_ASYNC(Kernel, grid, block, sharedMem, stream, ...) \
    do { \
        int64_t _gpumon_ts_start = gpumon::detail::getTimestampNs(); \
        kernel<<<grid, block, sharedMem, stream>>>(__VA_ARGS__); \
        cudaError_t _gpumon_err = cudaGetLastError(); \
        int64_t _gpumon_ts_end = gpumon::detail::getTimestampNs(); \
        gpumon::detail::logKernelEvent( \
        #kernel, \
        _gpumon_ts_start, \
        _gpumon_ts_end, \
        grid, \
        block, \
        sharedMem, \
        gpumon::detail::getCudaErrorString(_gpumon_err) \
        ); \
    } while(0)

// wraps a single kernel launch with custom tag
#define GPUMON_LAUNCH_TAGGED(tag, kernel, grid, block, sharedMem, stream, ...) \
    do { \
        int64_t _gpumon_ts_start = gpumon::detail::getTimestampNs(); \
        kernel<<<grid, block, sharedMem, stream>>>(__VA_ARGS__); \
        cudaError_t _gpumon_err = cudaGetLastError(); \
        cudaError_t _gpumon_sync_err = cudaDeviceSynchronize(); \
        if (_gpumon_sync_err != cudaSuccess && _gpumon_err == cudaSuccess) { \
            _gpumon_err = _gpumon_sync_err; \
        } \
        int64_t _gpumon_ts_end = gpumon::detail::getTimestampNs(); \
        gpumon::detail::logKernelEvent( \
            #kernel, \
            _gpumon_ts_start, \
            _gpumon_ts_end, \
            grid, \
            block, \
            sharedMem, \
            gpumon::detail::getCudaErrorString(_gpumon_err), \
            tag \
        ); \
    } while(0)

// Block-style macro using RAII
// Usage: GPUMON_SCOPE("name") { /* your code with kernel launches */ }
// Usage with tag: GPUMON_SCOPE_TAGGED("name", "tag") { /* your code */ }
#define GPUMON_SCOPE(name) \
    if (gpumon::ScopedMonitor _gpumon_scope{name}; true)

#define GPUMON_SCOPE_TAGGED(name, tag) \
    if (gpumon::ScopedMonitor _gpumon_scope{name, tag}; true)

#endif // GPUMON_HPP
