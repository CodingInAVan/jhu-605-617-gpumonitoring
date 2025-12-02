#ifndef GPUMON_COMMON_HPP
#define GPUMON_COMMON_HPP

#include <string>
#include <fstream>
#include <mutex>
#include <chrono>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <atomic>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace gpumon {

    // ============================================================================
    // Data Structures
    // ============================================================================

    struct InitOptions {
        std::string appName;
        std::string logFilePath;
        uint32_t sampleIntervalMs = 0; // 0 to disable background sampling
    };

    namespace detail {

        // Unified memory snapshot structure for all backends
        struct MemorySnapshot {size_t freeMiB = 0;
            size_t totalMiB = 0;
            size_t usedMiB = 0;
            int deviceId = 0;
            bool valid = false;
        };

        // ============================================================================
        // Internal State (Singleton)
        // ============================================================================

        struct State {
            std::string appName;
            std::ofstream logFile;
            std::mutex logMutex;
            int32_t pid;
            std::atomic<bool> initialized;
            uint32_t sampleIntervalMs;

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

        // ============================================================================
        // Platform Utilities
        // ============================================================================

        inline int32_t getPid() {
        #ifdef _WIN32
            return static_cast<int32_t>(_getpid());
        #else
            return static_cast<int32_t>(getpid());
        #endif
        }

        inline int64_t getTimestampNs() {
            const auto now = std::chrono::steady_clock::now();
            const auto duration = now.time_since_epoch();
            return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
        }

        inline std::string getDefaultLogPath(const std::string& appName, const int32_t pid) {
            const char* logDirEnv = std::getenv("GPUMON_LOG_DIR");
            if (!logDirEnv || logDirEnv[0] == '\0') return "";

            const std::string logDir = logDirEnv;
            std::ostringstream oss;
            oss << logDir;
        #ifdef _WIN32
            if (!logDir.empty() && logDir.back() != '\\' && logDir.back() != '/') oss << "\\";
        #else
            if (!logDir.empty() && logDir.back() != '/') oss << "/";
        #endif
            oss << "gpumon_" << appName << "_" << pid << ".log";
            return oss.str();
        }

        // ============================================================================
        // JSON & Logging Utilities
        // ============================================================================

        inline std::string escapeJson(const std::string& str) {
            std::ostringstream oss;
            for (const char c : str) {
                switch (c) {
                    case '"':  oss << "\\\""; break;
                    case '\\': oss << "\\\\"; break;
                    case '\b': oss << "\\b"; break;
                    case '\f': oss << "\\f"; break;
                    case '\n': oss << "\\n"; break;
                    case '\r': oss << "\\r"; break;
                    case '\t': oss << "\\t"; break;
                    default:   oss << c; break;
                }
            }
            return oss.str();
        }

        inline void writeLogLine(const std::string& jsonLine) {
            State& state = getState();
            // Use lock to ensure thread-safe writing from sampler + main thread
            std::lock_guard lock(state.logMutex);
            if (state.logFile.is_open()) {
                state.logFile << jsonLine << '\n';
                state.logFile.flush();
            }
        }

        // Helper to format the memory array into JSON
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
} // namespace gpumon

#endif // GPUMON_COMMON_HPP