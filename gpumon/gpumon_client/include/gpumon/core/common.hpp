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
        // Snapshot of Device State (Identity + Memory)
        struct DeviceSnapshot {
            int deviceId = 0;
            std::string name;
            std::string uuid; // Extracted via Runtime API
            int pciBusId = 0;

            size_t freeMiB = 0;
            size_t totalMiB = 0;
            size_t usedMiB = 0;

            unsigned int gpuUtil = 0;      // %
            unsigned int memUtil = 0;      // %
            unsigned int tempC = 0;        // Celsius
            unsigned int powermW = 0;      // Milliwatts
            unsigned int clockGfx = 0;     // MHz
            unsigned int clockSm = 0;      // MHz
            unsigned int clockMem = 0;     // MHz
        };

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
        inline void writeDeviceJson(std::ostringstream& oss, const std::vector<DeviceSnapshot>& snapshots) {
            if (snapshots.empty()) return;
            oss << ",\"devices\":[";
            for (size_t i = 0; i < snapshots.size(); ++i) {
                if (i > 0) oss << ",";
            const auto& s = snapshots[i];
                oss << "{\"id\":" << s.deviceId
                    << ",\"name\":\"" << escapeJson(s.name) << "\""
                    << ",\"uuid\":\"" << s.uuid << "\""
                    << ",\"pci_bus\":" << s.pciBusId
                    << ",\"used_mib\":" << s.usedMiB
                    << ",\"free_mib\":" << s.freeMiB
                    << ",\"total_mib\":" << s.totalMiB
                    << ",\"util_gpu\":" << s.gpuUtil
                    << ",\"util_mem\":" << s.memUtil
                    << ",\"temp_c\":" << s.tempC
                    << ",\"power_mw\":" << s.powermW
                    << ",\"clk_gfx\":" << s.clockGfx
                    << ",\"clk_sm\":" << s.clockSm
                    << ",\"clk_mem\":" << s.clockMem
                    << "}";
            }
            oss << "]";
        }

    } // namespace detail
} // namespace gpumon

#endif // GPUMON_COMMON_HPP