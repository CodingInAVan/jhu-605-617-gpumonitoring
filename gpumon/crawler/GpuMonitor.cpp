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
#include <set>
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
    int64_t scopeMemDeltaMiB = 0;  // Memory allocated/freed during scope
    int64_t scopeMemUsedMiB = 0;   // Memory used at end of scope
    bool hasMemoryData = false;
};

// Helper: get all process events from all log readers within timestamp range
static std::map<int32_t, std::vector<ClientEvent>> getAllProcessEventsFromReaders(
    const std::vector<std::unique_ptr<ClientLogReader>>& logReaders,
    int64_t startNs,
    int64_t endNs)
{
    std::map<int32_t, std::vector<ClientEvent>> allProcessEvents;

    std::cout << "[getAllProcessEventsFromReaders] Collecting events from " << logReaders.size()
              << " log readers, time range: " << startNs << " to " << endNs << std::endl;

    for (size_t i = 0; i < logReaders.size(); ++i) {
        const auto& reader = logReaders[i];
        if (!reader || !reader->isValid()) {
            std::cout << "[getAllProcessEventsFromReaders] Skipping invalid reader " << i << std::endl;
            continue;
        }

        auto readerEvents = reader->getAllProcessEvents(startNs, endNs);
        std::cout << "[getAllProcessEventsFromReaders] Reader " << i
                  << " found events for " << readerEvents.size() << " processes" << std::endl;

        // Merge events from this reader into the combined map
        for (auto& [pid, events] : readerEvents) {
            allProcessEvents[pid].insert(
                allProcessEvents[pid].end(),
                std::make_move_iterator(events.begin()),
                std::make_move_iterator(events.end())
            );
        }
    }

    std::cout << "[getAllProcessEventsFromReaders] Total: " << allProcessEvents.size()
              << " processes with events" << std::endl;

    return allProcessEvents;
}

// Helper: enrich process data from clientlib events
static EnrichedProcessData enrichProcessData(const std::vector<ClientEvent>& events, unsigned int pid)
{
    std::cout << "[enrichProcessData] Processing " << events.size() << " events for PID=" << pid << std::endl;

    EnrichedProcessData enriched;

    // Aggregate data from events
    for (const auto& event : events) {
        std::cout << "[enrichProcessData] Processing event: type=" << event.type;
        if (!event.appName.empty()) std::cout << ", app=" << event.appName;
        if (!event.kernelName.empty()) std::cout << ", kernel=" << event.kernelName;
        if (!event.scopeName.empty()) std::cout << ", scope=" << event.scopeName;
        if (!event.tag.empty()) std::cout << ", tag=" << event.tag;
        if (event.memDeltaMiB != 0) std::cout << ", memDelta=" << event.memDeltaMiB << "MiB";
        std::cout << std::endl;

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
        // Collect memory data from scope_end events
        if (event.type == "scope_end") {
            if (event.memDeltaMiB != 0) {
                enriched.scopeMemDeltaMiB += event.memDeltaMiB;
            }
            // Also capture the absolute memory usage at scope end
            if (event.memEndUsedMiB > 0) {
                enriched.scopeMemUsedMiB = event.memEndUsedMiB;
                enriched.hasMemoryData = true;
            }
        }
    }

    std::cout << "[enrichProcessData] Summary for PID " << pid
              << ": appName=" << (enriched.appName.empty() ? "(none)" : enriched.appName)
              << ", kernels=" << (enriched.kernelName.empty() ? "(none)" : enriched.kernelName)
              << ", scope=" << (enriched.scopeName.empty() ? "(none)" : enriched.scopeName)
              << ", tag=" << (enriched.tag.empty() ? "(none)" : enriched.tag)
              << ", memDelta=" << enriched.scopeMemDeltaMiB << " MiB"
              << ", memUsed=" << enriched.scopeMemUsedMiB << " MiB"
              << std::endl;

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
    if (enriched.hasMemoryData) {
        if (enriched.scopeMemDeltaMiB != 0) {
            j << ",\"scopeMemDeltaMiB\":" << enriched.scopeMemDeltaMiB;
        }
        if (enriched.scopeMemUsedMiB > 0) {
            j << ",\"scopeMemUsedMiB\":" << enriched.scopeMemUsedMiB;
        }
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
            memMiB = 0;
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

        // Read new events from all log files into cache
        for (auto& reader : logReaders) {
            size_t newEvents = reader->readNewEvents();
            if (newEvents > 0) {
                std::cout << "[ClientLog] Read " << newEvents << " new events. Total cached: "
                          << reader->getCachedEventCount() << std::endl;
            }
        }

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

        // Get all process events from clientlib logs for this time window
        auto now = std::chrono::steady_clock::now();
        int64_t endNs = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        int64_t startNs = endNs - (aggregationIntervalMs * 1000000LL); // Convert ms to ns

        std::map<int32_t, std::vector<ClientEvent>> allProcessEvents =
            getAllProcessEventsFromReaders(logReaders, startNs, endNs);

        // Track which PIDs from logs we've already processed with NVML data
        std::set<int32_t> processedPids;

        // Send process-level metrics (one per process) - NVML processes with enrichment
        for (const auto&[fst, snd] : processMetrics) {
            const std::string& gpuUuid = fst;
            const auto& processMap = snd;

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

                // Enrich this process with clientlib events (if available)
                EnrichedProcessData enriched;
                auto eventIt = allProcessEvents.find(static_cast<int32_t>(pid));
                if (eventIt != allProcessEvents.end()) {
                    enriched = enrichProcessData(eventIt->second, pid);
                    processedPids.insert(static_cast<int32_t>(pid));
                }

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

        // Send metrics for processes that have log events but weren't captured by NVML
        // (e.g., processes that finished, or short-lived processes)
        for (const auto& [pid, events] : allProcessEvents) {
            if (processedPids.find(pid) != processedPids.end()) {
                continue; // Already processed with NVML data
            }

            // Enrich from log events only
            EnrichedProcessData enriched = enrichProcessData(events, pid);

            // Create empty process metrics (no NVML data available)
            ProcessMetrics emptyMetrics;
            emptyMetrics.pid = pid;
            emptyMetrics.processName = enriched.appName.empty() ? "unknown" : enriched.appName;

            std::cout << "  Process (log-only): PID=" << pid
                      << ", App=" << enriched.appName;
            if (!enriched.kernelName.empty()) {
                std::cout << ", Kernel=" << enriched.kernelName;
            }
            if (!enriched.tag.empty()) {
                std::cout << ", Tag=" << enriched.tag;
            }
            std::cout << " (not in NVML)" << std::endl;

            // Send with unknown GPU (we don't know which GPU this process used)
            sender_->send(buildProcessMetricJson(
                aggregationTimestamp, hostname_, "unknown", "unknown",
                pid, emptyMetrics.processName, emptyMetrics, enriched
            ));
        }

        ++iter;
        if (oneShot || (maxIters > 0 && iter >= maxIters)) break;
    }

    // Shutdown NVML on exit
    nvmlShutdown();
}
