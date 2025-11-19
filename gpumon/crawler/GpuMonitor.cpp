#include "GpuMonitor.h"
#include "Utils.h"
#include "NvidiaSmiHelper.h"
#include "ClientLogReader.h"
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <vector>
#include <map>
#include <numeric>
#include <climits>
#include <filesystem>

#ifdef _WIN32
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <unistd.h>
#endif

using util::nowIso8601Utc;

// Static stop flag definition
std::atomic<bool> GpuMonitor::stop_{false};

// GPU-level metrics (one per GPU)
struct GpuMetrics {
    std::vector<unsigned int> gpuUtilPercent;
    std::vector<unsigned int> memUtilPercent;
    std::vector<unsigned int> temperatureCelsius;
    std::vector<unsigned int> powerMilliwatts;
    std::vector<unsigned int> graphicsClockMHz;
    std::vector<unsigned int> smClockMHz;
    std::vector<unsigned int> memClockMHz;
    uint64_t totalMemoryMiB = 0;
    uint64_t usedTotalMemoryMiB = 0;
    uint64_t freeMemoryMiB = 0;

    void addSample(unsigned int gpuUtil, unsigned int memUtil, unsigned int temp,
                   unsigned int power, unsigned int gfxClock, unsigned int sm,
                   unsigned int mem, uint64_t total, uint64_t used, uint64_t free) {
        gpuUtilPercent.push_back(gpuUtil);
        memUtilPercent.push_back(memUtil);
        temperatureCelsius.push_back(temp);
        powerMilliwatts.push_back(power);
        graphicsClockMHz.push_back(gfxClock);
        smClockMHz.push_back(sm);
        memClockMHz.push_back(mem);
        totalMemoryMiB = total;
        usedTotalMemoryMiB = used;
        freeMemoryMiB = free;
    }
};

// Process-level metrics (one per process)
struct ProcessMetrics {
    unsigned int pid = 0;
    std::string processName;
    std::vector<uint64_t> usedMemoryMiB;

    void addSample(uint64_t procMem) {
        usedMemoryMiB.push_back(procMem);
    }
};

static uint64_t average(const std::vector<uint64_t>& vals) {
    if (vals.empty()) return 0;
    return std::accumulate(vals.begin(), vals.end(), 0ULL) / vals.size();
}

static uint64_t average(const std::vector<unsigned int>& vals) {
    if (vals.empty()) return 0;
    return std::accumulate(vals.begin(), vals.end(), 0ULL) / vals.size();
}

static uint64_t maxVal(const std::vector<uint64_t>& vals) {
    if (vals.empty()) return 0;
    return *std::max_element(vals.begin(), vals.end());
}

static uint64_t maxVal(const std::vector<unsigned int>& vals) {
    if (vals.empty()) return 0;
    return *std::max_element(vals.begin(), vals.end());
}

void GpuMonitor::requestStop() { stop_.store(true, std::memory_order_relaxed); }
bool GpuMonitor::isStopRequested() { return stop_.load(std::memory_order_relaxed); }

static uint32_t getenvOrUint(const char* key, uint32_t def) {
    const std::string v = util::getenvOr(key, "");
    if (v.empty()) return def;
    try { return static_cast<uint32_t>(std::stoul(v)); } catch (...) { return def; }
}

static std::string getHostName() {
    char host[256];
    if (gethostname(host, sizeof(host)) == 0) {
        return host;
    }
    return "unknown";
}

// Build GPU-level metrics JSON
static std::string buildGpuMetricJson(const std::string& ts,
                                      const std::string& hostname,
                                      const std::string& gpuId,
                                      const std::string& gpuName,
                                      const GpuMetrics& metrics) {
    std::ostringstream j;
    j << "{\"timestamp\":\"" << ts << "\","
      << "\"hostname\":\"" << hostname << "\","
      << "\"gpuId\":\"" << gpuId << "\","
      << "\"gpuName\":\"" << gpuName << "\","
      << "\"metricType\":\"gpu\"";

    // Memory metrics
    j << ",\"totalMemoryMiB\":" << metrics.totalMemoryMiB
      << ",\"usedTotalMemoryMiB\":" << metrics.usedTotalMemoryMiB
      << ",\"freeMemoryMiB\":" << metrics.freeMemoryMiB;

    // Utilization metrics (average only)
    j << ",\"gpuUtilPercent\":" << average(metrics.gpuUtilPercent)
      << ",\"memUtilPercent\":" << average(metrics.memUtilPercent);

    // Temperature metrics (average only)
    j << ",\"temperatureCelsius\":" << average(metrics.temperatureCelsius);

    // Power metrics (average only)
    j << ",\"powerMilliwatts\":" << average(metrics.powerMilliwatts);

    // Clock metrics (average only)
    j << ",\"graphicsClockMHz\":" << average(metrics.graphicsClockMHz)
      << ",\"smClockMHz\":" << average(metrics.smClockMHz)
      << ",\"memClockMHz\":" << average(metrics.memClockMHz);

    j << '}';

    // Debug output
    std::string result = j.str();
    std::cout << "DEBUG GPU JSON: " << result << std::endl;

    return result;
}

// Structure to hold enriched process data from clientlib
struct EnrichedProcessData {
    std::string appName;
    std::string kernelName;
    std::string regionName;
    std::string scopeName;
    std::string tag;
};

// Helper: find matching clientlib events for a PID within timestamp range
static EnrichedProcessData getEnrichedDataForProcess(
    const std::vector<std::unique_ptr<ClientLogReader>>& logReaders,
    unsigned int pid,
    int64_t startNs,
    int64_t endNs)
{
    EnrichedProcessData enriched;

    for (const auto& reader : logReaders) {
        if (!reader || !reader->isValid()) continue;

        auto events = reader->getEventsForPid(static_cast<int32_t>(pid), startNs, endNs);

        // Aggregate data from events
        for (const auto& event : events) {
            if (!event.appName.empty() && enriched.appName.empty()) {
                enriched.appName = event.appName;
            }
            if (!event.kernelName.empty()) {
                if (!enriched.kernelName.empty()) enriched.kernelName += ",";
                enriched.kernelName += event.kernelName;
            }
            if (!event.regionName.empty() && enriched.regionName.empty()) {
                enriched.regionName = event.regionName;
            }
            if (!event.scopeName.empty() && enriched.scopeName.empty()) {
                enriched.scopeName = event.scopeName;
            }
            if (!event.tag.empty() && enriched.tag.empty()) {
                enriched.tag = event.tag;
            }
        }

        // If we found data, no need to check other log files
        if (!enriched.appName.empty()) break;
    }

    return enriched;
}

// Build process-level metrics JSON
static std::string buildProcessMetricJson(const std::string& ts,
                                          const std::string& hostname,
                                          const std::string& gpuId,
                                          const std::string& gpuName,
                                          unsigned int pid,
                                          const std::string& processName,
                                          const ProcessMetrics& metrics,
                                          const EnrichedProcessData& enriched = {}) {
    std::ostringstream j;
    j << "{\"timestamp\":\"" << ts << "\","
      << "\"hostname\":\"" << hostname << "\","
      << "\"gpuId\":\"" << gpuId << "\","
      << "\"gpuName\":\"" << gpuName << "\","
      << "\"metricType\":\"process\"";

    // Add pid and processName
    j << ",\"pid\":" << pid
      << ",\"processName\":\"" << processName << "\"";
    j << ",\"processUsedMemoryMiB\":" << average(metrics.usedMemoryMiB);

    // Add enriched data from clientlib if available
    if (!enriched.appName.empty()) {
        j << ",\"appName\":\"" << enriched.appName << "\"";
    }
    if (!enriched.kernelName.empty()) {
        j << ",\"kernelName\":\"" << enriched.kernelName << "\"";
    }
    if (!enriched.regionName.empty()) {
        j << ",\"regionName\":\"" << enriched.regionName << "\"";
    }
    if (!enriched.scopeName.empty()) {
        j << ",\"scopeName\":\"" << enriched.scopeName << "\"";
    }
    if (!enriched.tag.empty()) {
        j << ",\"tag\":\"" << enriched.tag << "\"";
    }

    j << '}';

    // Debug output
    std::string result = j.str();
    std::cout << "DEBUG PROCESS JSON: " << result << std::endl;

    return result;
}

GpuMonitor::GpuMonitor(std::unique_ptr<IMetricsSender> sender, const std::vector<std::string>& clientLogPaths)
    : sender_(std::move(sender)), clientLogPaths_(clientLogPaths) {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        throw std::runtime_error("WSAStartup failed");
    }
#endif
    hostname_ = getHostName();
}

// Helper: collect a single sample for a GPU and store in the metrics maps
static void collectSampleForIndex(unsigned int i, nvmlDevice_t device,
                                  const char* uuid, const char* deviceName,
                                  std::map<std::string, GpuMetrics>& gpuMetrics,
                                  std::map<std::string, std::map<unsigned int, ProcessMetrics>>& processMetrics) {
    // Get running processes on this GPU via NVML
    unsigned int infoCount = 0;
    std::vector<nvmlProcessInfo_t> processes;

    // First call to get count
    nvmlReturn_t result = nvmlDeviceGetComputeRunningProcesses(device, &infoCount, nullptr);
    if (result == NVML_SUCCESS || result == NVML_ERROR_INSUFFICIENT_SIZE) {
        if (infoCount > 0) {
            processes.resize(infoCount);
            result = nvmlDeviceGetComputeRunningProcesses(device, &infoCount, processes.data());
            if (result == NVML_SUCCESS) {
                processes.resize(infoCount);
            } else {
                processes.clear();
            }
        }
    }

    // If no compute processes, check graphics processes
    if (processes.empty()) {
        infoCount = 0;
        result = nvmlDeviceGetGraphicsRunningProcesses(device, &infoCount, nullptr);
        if (result == NVML_SUCCESS || result == NVML_ERROR_INSUFFICIENT_SIZE) {
            if (infoCount > 0) {
                processes.resize(infoCount);
                result = nvmlDeviceGetGraphicsRunningProcesses(device, &infoCount, processes.data());
                if (result == NVML_SUCCESS) {
                    processes.resize(infoCount);
                } else {
                    processes.clear();
                }
            }
        }
    }

    // Query per-process memory via nvidia-smi (workaround for Windows WDDM)
    // Build a map: PID -> memory usage (MiB) for fast lookup
    std::map<unsigned int, uint64_t> smiMemoryMap;
    std::vector<SmiProcessInfo> smiProcesses = queryProcessMemoryViaSmi(i);
    for (const auto& smiProc : smiProcesses) {
        smiMemoryMap[smiProc.pid] = smiProc.usedMemoryMiB;
    }

    // Collect GPU-level metrics
    nvmlMemory_t memoryInfo{};
    nvmlDeviceGetMemoryInfo(device, &memoryInfo);

    nvmlUtilization_t utilization{};
    nvmlDeviceGetUtilizationRates(device, &utilization);

    unsigned int temperature = 0;
    nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);

    unsigned int powerMilliwatts = 0;
    nvmlDeviceGetPowerUsage(device, &powerMilliwatts);

    unsigned int graphicsClock = 0, smClock = 0, memClock = 0;
    nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &graphicsClock);
    nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &smClock);
    nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &memClock);

    std::string gpuKey = uuid;

    // Add GPU-level sample
    auto& gpuMetric = gpuMetrics[gpuKey];
    gpuMetric.addSample(utilization.gpu, utilization.memory, temperature,
                       powerMilliwatts, graphicsClock, smClock, memClock,
                       memoryInfo.total / 1024 / 1024,
                       memoryInfo.used / 1024 / 1024,
                       memoryInfo.free / 1024 / 1024);

    // Add process-level samples
    for (const auto& proc : processes) {
        auto& procMetric = processMetrics[gpuKey][proc.pid];

        if (procMetric.processName.empty()) {
            char procNameBuf[256];
            unsigned int len = sizeof(procNameBuf);
            if (nvmlSystemGetProcessName(proc.pid, procNameBuf, len) == NVML_SUCCESS) {
                procMetric.processName = procNameBuf;
            } else {
                procMetric.processName = "unknown";
            }
            procMetric.pid = proc.pid;
        }

        // Determine per-process memory usage:
        // 1. Try NVML first (works on Linux/TCC mode)
        // 2. Fall back to nvidia-smi data (Windows WDDM workaround)
        // 3. Default to 0 if neither is available
        uint64_t memMiB = 0;

        // Check if NVML provides valid memory data
        if (proc.usedGpuMemory != ULLONG_MAX && proc.usedGpuMemory != 0xFFFFFFFFFFFFFFFFULL) {
            memMiB = proc.usedGpuMemory / (1024 * 1024);
        } else {
            // NVML memory not available (Windows WDDM), use nvidia-smi data
            auto it = smiMemoryMap.find(proc.pid);
            if (it != smiMemoryMap.end()) {
                memMiB = it->second;
            } else {
                // Process not found in nvidia-smi output, use 0
                memMiB = 0;
            }
        }

        procMetric.addSample(memMiB);
    }
}

void GpuMonitor::runLoop() const {
    // 1. Initialize NVML
    checkNvmlError(nvmlInit_v2(), "nvmlInit_v2");
    std::cout << "✅ NVML Initialized Successfully." << std::endl;

    unsigned int deviceCount;
    checkNvmlError(nvmlDeviceGetCount_v2(&deviceCount), "nvmlDeviceGetCount_v2");
    if (deviceCount == 0) {
        std::cout << "No NVIDIA devices found on this system." << std::endl;
        nvmlShutdown();
        return;
    }

    std::cout << "Found " << deviceCount << " NVIDIA device(s)." << std::endl;
    std::cout << "---" << std::endl;

    // Read control env vars
    const uint32_t aggregationIntervalMs = getenvOrUint("METRICS_INTERVAL_MS", 5000);  // How often to send aggregated metrics
    const uint32_t sampleIntervalMs = getenvOrUint("METRICS_SAMPLE_INTERVAL_MS", 500);  // How often to sample
    const uint32_t maxIters = getenvOrUint("METRICS_MAX_ITERS", 0); // 0 = infinite
    const bool oneShot = util::getenvOr("METRICS_ONESHOT", "0") == std::string("1");

    // Store device handles and metadata
    struct DeviceInfo {
        nvmlDevice_t handle;
        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        char uuid[NVML_DEVICE_UUID_BUFFER_SIZE];
    };
    std::vector<DeviceInfo> devices(deviceCount);

    for (unsigned int i = 0; i < deviceCount; ++i) {
        checkNvmlError(nvmlDeviceGetHandleByIndex_v2(i, &devices[i].handle), "nvmlDeviceGetHandleByIndex_v2");
        checkNvmlError(nvmlDeviceGetName(devices[i].handle, devices[i].name, NVML_DEVICE_NAME_BUFFER_SIZE), "nvmlDeviceGetName");
        checkNvmlError(nvmlDeviceGetUUID(devices[i].handle, devices[i].uuid, NVML_DEVICE_UUID_BUFFER_SIZE), "nvmlDeviceGetUUID");
        std::cout << "GPU " << i << ": " << devices[i].name << " [" << devices[i].uuid << "]" << std::endl;
    }

    // Warm-up utilization sampling to avoid always-0 readings on first pass
    for (unsigned int i = 0; i < deviceCount; ++i) {
        nvmlUtilization_t dummy{};
        nvmlDeviceGetUtilizationRates(devices[i].handle, &dummy);
    }

    // Ensure sample interval is at least 100ms (NVML requirement)
    static constexpr unsigned int kMinNvmlSampleIntervalMs = 100u;
    const auto actualSampleIntervalMs = max(sampleIntervalMs, kMinNvmlSampleIntervalMs);

    // 2.5. Initialize client log readers if log paths are provided
    std::vector<std::unique_ptr<ClientLogReader>> logReaders;
    for (const auto& logPath : clientLogPaths_) {
        auto reader = std::make_unique<ClientLogReader>(logPath);
        if (reader->isValid()) {
            std::cout << "Initialized log reader for: " << logPath << std::endl;
            logReaders.push_back(std::move(reader));
        } else {
            std::cout << "Warning: Could not open log file: " << logPath << std::endl;
        }
    }

    // 3. Start the monitoring loop with sampling and aggregation
    uint32_t iter = 0;
    while (!isStopRequested()) {
        // Map: GPU UUID -> GpuMetrics
        std::map<std::string, GpuMetrics> gpuMetrics;
        // Map: GPU UUID -> PID -> ProcessMetrics
        std::map<std::string, std::map<unsigned int, ProcessMetrics>> processMetrics;

        // Collect samples over the aggregation interval
        const auto aggregationStart = std::chrono::steady_clock::now();
        const std::string aggregationTimestamp = nowIso8601Utc();

        while (true) {
            // Collect one sample from each GPU
            for (unsigned int i = 0; i < deviceCount; i++) {
                collectSampleForIndex(i, devices[i].handle, devices[i].uuid, devices[i].name,
                                     gpuMetrics, processMetrics);
            }

            // Check if we've reached the aggregation interval
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - aggregationStart
            ).count();

            if (elapsed >= aggregationIntervalMs) {
                break;
            }

            // Sleep until next sample (but check for stop request)
            std::this_thread::sleep_for(std::chrono::milliseconds(actualSampleIntervalMs));
            if (isStopRequested()) break;
        }

        // Send aggregated metrics
        std::cout << "Sending aggregated metrics for " << gpuMetrics.size() << " GPU(s)" << std::endl;

        // Send GPU-level metrics (one per GPU)
        for (const auto& gpuEntry : gpuMetrics) {
            const std::string& gpuUuid = gpuEntry.first;
            const GpuMetrics& metrics = gpuEntry.second;

            // Find device name for this GPU
            std::string gpuName = "unknown";
            for (const auto& dev : devices) {
                if (gpuUuid == dev.uuid) {
                    gpuName = dev.name;
                    break;
                }
            }

            std::cout << "  GPU: " << gpuName << ", Samples: " << metrics.gpuUtilPercent.size() << std::endl;

            sender_->send(buildGpuMetricJson(
                aggregationTimestamp, hostname_, gpuUuid, gpuName, metrics
            ));
        }

        // Send process-level metrics (one per process)
        for (const auto& gpuEntry : processMetrics) {
            const std::string& gpuUuid = gpuEntry.first;
            const auto& processMap = gpuEntry.second;

            // Find device name for this GPU
            std::string gpuName = "unknown";
            for (const auto& dev : devices) {
                if (gpuUuid == dev.uuid) {
                    gpuName = dev.name;
                    break;
                }
            }

            for (const auto& procEntry : processMap) {
                const unsigned int pid = procEntry.first;
                const ProcessMetrics& metrics = procEntry.second;

                // Get enriched data from clientlib logs if available
                // Use a reasonable time window around the aggregation interval
                auto now = std::chrono::steady_clock::now();
                int64_t endNs = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
                int64_t startNs = endNs - (aggregationIntervalMs * 1000000LL); // Convert ms to ns

                EnrichedProcessData enriched = getEnrichedDataForProcess(logReaders, pid, startNs, endNs);

                std::cout << "  Process on " << gpuName << ": PID=" << pid
                          << ", Name=" << metrics.processName
                          << ", Samples: " << metrics.usedMemoryMiB.size();

                if (!enriched.appName.empty()) {
                    std::cout << ", App=" << enriched.appName;
                }
                if (!enriched.kernelName.empty()) {
                    std::cout << ", Kernel=" << enriched.kernelName;
                }
                if (!enriched.tag.empty()) {
                    std::cout << ", Tag=" << enriched.tag;
                }
                std::cout << std::endl;

                sender_->send(buildProcessMetricJson(
                    aggregationTimestamp, hostname_, gpuUuid, gpuName,
                    pid, metrics.processName, metrics, enriched
                ));
            }
        }

        ++iter;
        if (oneShot || (maxIters > 0 && iter >= maxIters)) break;
    }

    // Shutdown NVML on exit
    nvmlShutdown();
}
