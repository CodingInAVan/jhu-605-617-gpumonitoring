#pragma once
#include <string>
#include <memory>
#include <stdexcept>
#include <atomic>
#include <vector>
#include <nvml.h>

#include "ClientLogReader.h"
#include "MetricsSender.h"

inline void checkNvmlError(nvmlReturn_t result, const std::string &functionName) {
    if (result != NVML_SUCCESS) {
        throw std::runtime_error(functionName + " failed with error:  " + nvmlErrorString(result));
    }
}

struct GpuMetrics;
struct ProcessMetrics;
struct DeviceInfo;

class GpuMonitor {
public:
    explicit GpuMonitor(std::unique_ptr<IMetricsSender> sender, const std::vector<std::string>& clientLogPaths = {});
    ~GpuMonitor() = default;

    void runLoop() const;

    // Cancellation control
    static void requestStop();
    static bool isStopRequested();

private:
    [[nodiscard]] static std::vector<DeviceInfo> initializeNvml();
    void initializeLogReaders(std::vector<std::unique_ptr<gpumon::ClientLogReader>>& readers) const;

    static void collectWindowSamples(
        uint32_t durationMs,
        uint32_t sampleIntervalMs,
        const std::vector<DeviceInfo>& devices,
        std::map<std::string, GpuMetrics>& outGpuMetrics,
        std::map<std::string, std::map<unsigned int, ProcessMetrics>>& outProcMetrics
    ) ;

    void sendMetric(const std::string &json) const;

    // Merges NVML data with Client Logs and sends JSON
    void processAndSendMetrics(
        int64_t startNs,
        int64_t endNs,
        const std::string& timestampIso,
        const std::vector<DeviceInfo>& devices,
        const std::vector<std::unique_ptr<gpumon::ClientLogReader>>& readers,
        const std::map<std::string, GpuMetrics>& gpuMetrics,
        const std::map<std::string, std::map<unsigned int, ProcessMetrics>>& processMetrics
    ) const;

    static std::atomic<bool> stop_;
    std::unique_ptr<IMetricsSender> sender_;
    std::string hostname_;
    std::vector<std::string> clientLogPaths_;
    bool debugMode_;
};
