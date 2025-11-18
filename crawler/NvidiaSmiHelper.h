#pragma once

#include <vector>
#include <string>

/**
 * Structure representing per-process GPU memory information from nvidia-smi.
 *
 * On Windows WDDM, per-process GPU memory reporting is often unavailable
 * through both NVML and nvidia-smi. This is a known limitation of the
 * Windows Display Driver Model.
 */
struct SmiProcessInfo {
    unsigned int pid;
    std::string processName;
    uint64_t usedMemoryMiB;
};

/**
 * Queries per-process GPU memory usage via nvidia-smi for a specific GPU device.
 *
 * This function is a workaround for Windows WDDM systems where NVML cannot
 * provide per-process GPU memory (nvmlProcessInfo_t::usedGpuMemory returns
 * NVML_VALUE_NOT_AVAILABLE).
 *
 * @param deviceIndex The GPU device index (0-based)
 * @return A vector of SmiProcessInfo structs containing process information,
 *         or an empty vector if nvidia-smi fails or memory data is unavailable
 *
 * @note On most Windows WDDM systems, this will return an empty vector or
 *       entries with 0 memory because per-process memory tracking is not
 *       supported by the driver model.
 */
std::vector<SmiProcessInfo> queryProcessMemoryViaSmi(unsigned int deviceIndex);
