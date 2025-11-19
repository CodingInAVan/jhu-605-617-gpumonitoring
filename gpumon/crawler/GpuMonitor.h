#pragma once
#include <string>
#include <memory>
#include <stdexcept>
#include <atomic>
#include <vector>
#include <nvml.h>
#include "MetricsSender.h"

inline void checkNvmlError(nvmlReturn_t result, const std::string &functionName) {
    if (result != NVML_SUCCESS) {
        throw std::runtime_error(functionName + " failed with error:  " + nvmlErrorString(result));
    }
}

class GpuMonitor {
public:
    explicit GpuMonitor(std::unique_ptr<IMetricsSender> sender, const std::vector<std::string>& clientLogPaths = {});
    ~GpuMonitor() = default;

    void runLoop() const;

    // Cancellation control
    static void requestStop();
    static bool isStopRequested();

private:
    static std::atomic<bool> stop_;
    std::unique_ptr<IMetricsSender> sender_;
    std::string hostname_;
    std::vector<std::string> clientLogPaths_;
};
