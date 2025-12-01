#include "GpuMonitor.h"
#include "Utils.h"
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

struct DeviceInfo {
    nvmlDevice_t handle{};
    std::string name;
    std::string uuid;
};

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

    void addSample(const nvmlUtilization_t& util, const nvmlMemory_t& mem,
                   unsigned int temp, unsigned int power,
                   unsigned int gfx, unsigned int sm, unsigned int mclk) {
        gpuUtilPercent.push_back(util.gpu);
        memUtilPercent.push_back(util.memory);
        temperatureCelsius.push_back(temp);
        powerMilliwatts.push_back(power);
        graphicsClockMHz.push_back(gfx);
        smClockMHz.push_back(sm);
        memClockMHz.push_back(mclk);
        totalMemoryMiB = mem.total / 1024 / 1024;
        usedTotalMemoryMiB = mem.used / 1024 / 1024;
        freeMemoryMiB = mem.free / 1024 / 1024;
    }
};

// Process-level metrics (one per process)
struct ProcessMetrics {
    unsigned int pid = 0;
    std::string processName;
    std::vector<uint64_t> usedMemoryMiB;

    void addSample(const uint64_t procMem) {
        usedMemoryMiB.push_back(procMem);
    }
};

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

// ============================================================================
// Static Helpers
// ============================================================================

std::atomic<bool> GpuMonitor::stop_{false};

template<typename T>
static uint64_t average(const std::vector<T>& vals) {
    if (vals.empty()) return 0;
    return std::accumulate(vals.begin(), vals.end(), 0ULL) / vals.size();
}

static uint32_t getenvOrUint(const char* key, uint32_t def) {
    const std::string v = util::getenvOr(key, "");
    if (v.empty()) return def;
    try { return static_cast<uint32_t>(std::stoul(v)); } catch (...) { return def; }
}

static std::string getHostName() {
    char host[256];
    if (gethostname(host, sizeof(host)) == 0) return host;
    return "unknown";
}

// ============================================================================
// GpuMonitor Implementation
// ============================================================================

GpuMonitor::GpuMonitor(std::unique_ptr<IMetricsSender> sender, const std::vector<std::string>& clientLogPaths)
    : sender_(std::move(sender)), clientLogPaths_(clientLogPaths) {
#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
    hostname_ = getHostName();

    // check environment variable for debug mode (supports "1" or "true")
    const char* dbg = std::getenv("GPUMON_DEBUG");
    debugMode_ = (dbg && (std::string(dbg) == "1" || std::string(dbg) == "true"));

    if (debugMode_) {
        std::cout << "GpuMonitor: Debug mode enabled" << std::endl;
    }
}

void GpuMonitor::requestStop() { stop_.store(true, std::memory_order_relaxed); }
bool GpuMonitor::isStopRequested() { return stop_.load(std::memory_order_relaxed); }

// ----------------------------------------------------------------------------
// Phase 1: Initialization
// ----------------------------------------------------------------------------

std::vector<DeviceInfo> GpuMonitor::initializeNvml() {
    checkNvmlError(nvmlInit_v2(), "nvmlInit_v2");

    unsigned int deviceCount = 0;
    checkNvmlError(nvmlDeviceGetCount_v2(&deviceCount), "nvmlDeviceGetCount_v2");

    std::vector<DeviceInfo> devices(deviceCount);
    char nameBuf[NVML_DEVICE_NAME_BUFFER_SIZE];
    char uuidBuf[NVML_DEVICE_UUID_BUFFER_SIZE];

    for (unsigned int i = 0; i < deviceCount; ++i) {
        checkNvmlError(nvmlDeviceGetHandleByIndex_v2(i, &devices[i].handle), "GetHandle");
        checkNvmlError(nvmlDeviceGetName(devices[i].handle, nameBuf, sizeof(nameBuf)), "GetName");
        checkNvmlError(nvmlDeviceGetUUID(devices[i].handle, uuidBuf, sizeof(uuidBuf)), "GetUUID");

        devices[i].name = nameBuf;
        devices[i].uuid = uuidBuf;

        // Warm-up utilization
        nvmlUtilization_t dummy;
        nvmlDeviceGetUtilizationRates(devices[i].handle, &dummy);

        std::cout << "GPU " << i << ": " << devices[i].name << " [" << devices[i].uuid << "]\n";
    }
    return devices;
}

void GpuMonitor::initializeLogReaders(std::vector<std::unique_ptr<gpumon::ClientLogReader>>& readers) const {
    for (const auto& logPath : clientLogPaths_) {
        if (auto reader = std::make_unique<gpumon::ClientLogReader>(logPath, debugMode_); reader->isValid()) {
            if (debugMode_) {
                std::cout << "GpuMonitor: Initialized log reader for " << logPath << std::endl;
            }
            readers.push_back(std::move(reader));
        } else {
            if (debugMode_) {
                std::cout << "[GPUMON] Failed to open log: " << logPath << std::endl;
            }
        }
    }
}
// ----------------------------------------------------------------------------
// Phase 2: Data Collection (Sampling)
// ----------------------------------------------------------------------------
static std::vector<nvmlProcessInfo_t> getRunningProcesses(nvmlDevice_t device) {
    unsigned int infoCount = 0;
    // Try Compute Processes
    auto res = nvmlDeviceGetComputeRunningProcesses(device, &infoCount, nullptr);
    if ((res == NVML_SUCCESS || res == NVML_ERROR_INSUFFICIENT_SIZE) && infoCount > 0) {
        std::vector<nvmlProcessInfo_t> procs(infoCount);
        if (nvmlDeviceGetComputeRunningProcesses(device, &infoCount, procs.data()) == NVML_SUCCESS) {
            procs.resize(infoCount);
            return procs;
        }
    }

    // Fallback to Graphics Processes
    infoCount = 0;
    res = nvmlDeviceGetGraphicsRunningProcesses(device, &infoCount, nullptr);
    if ((res == NVML_SUCCESS || res == NVML_ERROR_INSUFFICIENT_SIZE) && infoCount > 0) {
        std::vector<nvmlProcessInfo_t> procs(infoCount);
        if (nvmlDeviceGetGraphicsRunningProcesses(device, &infoCount, procs.data()) == NVML_SUCCESS) {
            procs.resize(infoCount);
            return procs;
        }
    }
    return {};
}

void GpuMonitor::collectWindowSamples(
    uint32_t durationMs,
    uint32_t sampleIntervalMs,
    const std::vector<DeviceInfo>& devices,
    std::map<std::string, GpuMetrics>& outGpuMetrics,
    std::map<std::string, std::map<unsigned int, ProcessMetrics>>& outProcMetrics
) {

    auto start = std::chrono::steady_clock::now();

    while (!isStopRequested()) {
        for (const auto& dev : devices) {
            // 1. Collect GPU Metrics
            nvmlMemory_t memInfo{};
            nvmlUtilization_t util{};
            unsigned int temp = 0, power = 0;
            unsigned int clkGfx = 0, clkSm = 0, clkMem = 0;

            nvmlDeviceGetMemoryInfo(dev.handle, &memInfo);
            nvmlDeviceGetUtilizationRates(dev.handle, &util);
            nvmlDeviceGetTemperature(dev.handle, NVML_TEMPERATURE_GPU, &temp);
            nvmlDeviceGetPowerUsage(dev.handle, &power);
            nvmlDeviceGetClockInfo(dev.handle, NVML_CLOCK_GRAPHICS, &clkGfx);
            nvmlDeviceGetClockInfo(dev.handle, NVML_CLOCK_SM, &clkSm);
            nvmlDeviceGetClockInfo(dev.handle, NVML_CLOCK_MEM, &clkMem);

            outGpuMetrics[dev.uuid].addSample(util, memInfo, temp, power, clkGfx, clkSm, clkMem);

            // 2. Collect Process Metrics
            auto procs = getRunningProcesses(dev.handle);
            for (const auto& p : procs) {
                auto& procMetric = outProcMetrics[dev.uuid][p.pid];

                if (procMetric.processName.empty()) {
                    char nameBuf[256];
                    if (
                        constexpr unsigned int len = sizeof(nameBuf);
                        nvmlSystemGetProcessName(p.pid, nameBuf, len) == NVML_SUCCESS
                    ) {
                        procMetric.processName = nameBuf;
                    } else {
                        procMetric.processName = "unknown";
                    }
                    procMetric.pid = p.pid;
                }

                uint64_t memMiB = 0;
                if (p.usedGpuMemory != ULLONG_MAX) {
                    memMiB = p.usedGpuMemory / (1024 * 1024);
                }
                procMetric.addSample(memMiB);
            }
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();

        if (elapsed >= durationMs) break;

        std::this_thread::sleep_for(std::chrono::milliseconds(sampleIntervalMs));
    }
}
// ----------------------------------------------------------------------------
// Phase 3: Reporting & Enrichment
// ----------------------------------------------------------------------------

static EnrichedProcessData enrichData(const std::vector<gpumon::ClientEvent>& events) {
    EnrichedProcessData data;
    for (const auto& e : events) {
        if (!e.appName.empty() && data.appName.empty()) data.appName = e.appName;
        if (!e.kernelName.empty()) {
            if (!data.kernelName.empty()) data.kernelName += ",";
            data.kernelName += e.kernelName;
        }
        if (!e.regionName.empty() && data.regionName.empty()) data.regionName = e.regionName;
        if (!e.scopeName.empty() && data.scopeName.empty()) data.scopeName = e.scopeName;
        if (!e.tag.empty() && data.tag.empty()) data.tag = e.tag;

        if (e.type == "scope_end") {
            data.scopeMemDeltaMiB += e.memDeltaMiB;
            if (e.memEndUsedMiB > 0) {
                data.scopeMemUsedMiB = e.memEndUsedMiB;
                data.hasMemoryData = true;
            }
        }
    }
    return data;
}

void GpuMonitor::sendMetric(const std::string& json) const {
    if (debugMode_) {
        std::cout << "[GPUMON DEBUG JSON] " << json << std::endl;
    }
    if (sender_) {
        sender_->send(json);
    }
}

void GpuMonitor::processAndSendMetrics(
    int64_t startNs,
    int64_t endNs,
    const std::string& timestampIso,
    const std::vector<DeviceInfo>& devices,
    const std::vector<std::unique_ptr<gpumon::ClientLogReader>>& readers,
    const std::map<std::string, GpuMetrics>& gpuMetrics,
    const std::map<std::string, std::map<unsigned int, ProcessMetrics>>& processMetrics
) const {

    // 1. Send GPU Metrics
    for (const auto& entity : gpuMetrics) {
        const std::string& uuid = entity.first;
        const GpuMetrics& m = entity.second;

        std::string name = "unknown";
        auto it = std::find_if(devices.begin(), devices.end(), [&](const DeviceInfo& d){ return d.uuid == uuid; });
        if (it != devices.end()) name = it->name;

        std::ostringstream j;
        j << R"({"timestamp":")" << timestampIso << R"(","hostname":")" << hostname_
          << R"(","gpuId":")" << uuid << R"(","gpuName":")" << name
          << R"(","metricType":"gpu")"
          << ",\"totalMemoryMiB\":" << m.totalMemoryMiB
          << ",\"usedTotalMemoryMiB\":" << m.usedTotalMemoryMiB
          << ",\"freeMemoryMiB\":" << m.freeMemoryMiB
          << ",\"gpuUtilPercent\":" << average(m.gpuUtilPercent)
          << ",\"memUtilPercent\":" << average(m.memUtilPercent)
          << ",\"temperatureCelsius\":" << average(m.temperatureCelsius)
          << ",\"powerMilliwatts\":" << average(m.powerMilliwatts)
          << "}";
        sendMetric(j.str());
    }

    // 2. Aggregate Log Events
    std::map<int32_t, std::vector<gpumon::ClientEvent>> logEvents;
    for (const auto& r : readers) {
        if (!r || !r->isValid()) continue;
        auto events = r->getAllProcessEvents(startNs, endNs);
        for (auto& [pid, evs] : events) {
            logEvents[pid].insert(logEvents[pid].end(),
                                std::make_move_iterator(evs.begin()),
                                std::make_move_iterator(evs.end()));
        }
    }

    std::set<int32_t> processedPids;

    // 3. Send NVML Process Metrics (Enriched)
    for (const auto& gpuEntry : processMetrics) {
        const std::string& uuid = gpuEntry.first;
        const auto& procs = gpuEntry.second;

        std::string gpuName = "unknown";
        if (
            auto it = std::find_if(devices.begin(), devices.end(), [&](const DeviceInfo& d){ return d.uuid == uuid; });
            it != devices.end()
        ) gpuName = it->name;

        for (const auto& procEntry : procs) {
            const unsigned int pid = procEntry.first;
            const ProcessMetrics& m = procEntry.second;

            EnrichedProcessData enriched;
            if (logEvents.count(pid)) {
                enriched = enrichData(logEvents[pid]);
                processedPids.insert(pid);
            }

            std::ostringstream j;
            j << R"({"timestamp":")" << timestampIso << R"(","hostname":")" << hostname_
              << R"(","gpuId":")" << uuid << R"(","gpuName":")" << gpuName
              << R"(","metricType":"process","pid":)" << pid
              << R"(,"processName":")" << m.processName << "\""
              << ",\"processUsedMemoryMiB\":" << average(m.usedMemoryMiB);

            if (!enriched.appName.empty()) j << R"(,"appName":")" << enriched.appName << "\"";
            if (!enriched.kernelName.empty()) j << R"(,"kernelName":")" << enriched.kernelName << "\"";
            if (!enriched.tag.empty()) j << R"(,"tag":")" << enriched.tag << "\"";
            if (enriched.hasMemoryData) j << ",\"scopeMemDeltaMiB\":" << enriched.scopeMemDeltaMiB;

            j << "}";
            sendMetric(j.str());
        }
    }

    // 4. Send Orphan Log Metrics (No NVML match)
    for (const auto& entry : logEvents) {
        int32_t pid = entry.first;
        const auto& events = entry.second;

        if (processedPids.count(pid)) continue;

        // 1. Try to find which GPU this orphan belongs to
        std::string orphanGpuUuid = "unknown";
        std::string orphanGpuName = "unknown";

        for (const auto& e : events) {
            // If the event has a valid device ID, map it to our NVML device list
            if (e.deviceId >= 0 && e.deviceId < static_cast<int32_t>(devices.size())) {
                orphanGpuUuid = devices[e.deviceId].uuid;
                orphanGpuName = devices[e.deviceId].name;
                break; // Found it
            }
        }

        EnrichedProcessData enriched = enrichData(events);
        std::ostringstream j;
        j << "{\"timestamp\":\"" << timestampIso << "\",\"hostname\":\"" << hostname_
          << "\",\"gpuId\":\"" << orphanGpuUuid << "\",\"gpuName\":\"" << orphanGpuName << "\""
          << ",\"metricType\":\"process\",\"pid\":" << pid
          << ",\"processName\":\"" << (enriched.appName.empty() ? "unknown" : enriched.appName) << "\""
          << ",\"processUsedMemoryMiB\":0"; // NVML didn't see it, so memory is effectively 0 from sampling perspective

        if (!enriched.appName.empty()) j << ",\"appName\":\"" << enriched.appName << "\"";
        if (!enriched.kernelName.empty()) j << ",\"kernelName\":\"" << enriched.kernelName << "\"";
        if (!enriched.tag.empty()) j << ",\"tag\":\"" << enriched.tag << "\"";
        // Also send scope memory data if available in the log
        if (enriched.hasMemoryData) j << ",\"scopeMemDeltaMiB\":" << enriched.scopeMemDeltaMiB;

        j << "}";
        sendMetric(j.str());
    }
}

void GpuMonitor::runLoop() const {
    auto devices = initializeNvml();
    if (devices.empty()) {
        nvmlShutdown();
        return;
    }

    const uint32_t aggIntervalMs = getenvOrUint("METRICS_INTERVAL_MS", 5000);
    const uint32_t sampIntervalMs = max(100u, getenvOrUint("METRICS_SAMPLE_INTERVAL_MS", 500));

    std::vector<std::unique_ptr<gpumon::ClientLogReader>> logReaders;
    initializeLogReaders(logReaders);

    // Initialize time tracking
    const auto lastAggregationTime = std::chrono::steady_clock::now();
    int64_t lastEndNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        lastAggregationTime.time_since_epoch()).count();

    while (!isStopRequested()) {
        // 1. Refresh Logs
        for (auto& r : logReaders) r->readNewEvents();

        // 2. Prepare Storage
        std::map<std::string, GpuMetrics> gpuMetrics;
        std::map<std::string, std::map<unsigned int, ProcessMetrics>> procMetrics;

        // 3. Collect Data
        collectWindowSamples(aggIntervalMs, sampIntervalMs, devices, gpuMetrics, procMetrics);

        // 4. Calculate Time Window
        auto now = std::chrono::steady_clock::now();
        const int64_t currentEndNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();
        std::string tsIso = nowIso8601Utc();

        // 5. Send
        processAndSendMetrics(
            lastEndNs,      // Start from where we left off
            currentEndNs,   // End at now
            tsIso,
            devices, logReaders, gpuMetrics, procMetrics
        );

        // Update tracking
        lastEndNs = currentEndNs;
    }

    nvmlShutdown();
}
