#include "ClientLogReader.h"

#include <filesystem>
#include <iostream>
#include <utility>

#include "JsonUtils.h"

namespace gpumon {
    ClientLogReader::ClientLogReader(std::string logFilePath, const bool debugMode)
        : logFilePath_(std::move(logFilePath)), debugMode_(debugMode), lastPosition_(0) {
    }

    bool ClientLogReader::isValid() const {
        return std::filesystem::exists(logFilePath_);
    }

    std::optional<ClientEvent> ClientLogReader::parseLine(const std::string& line) {
        if (line.empty() || line[0] != '{') return std::nullopt;

        ClientEvent event;

        event.type = JsonUtils::extractString(line, "type");
        if (event.type.empty()) return std::nullopt;

        event.pid = static_cast<int32_t>(JsonUtils::extractInt(line, "pid"));
        event.appName = JsonUtils::extractString(line, "app");
        event.tag = JsonUtils::extractString(line, "tag");
        event.tsNs = JsonUtils::extractInt(line, "ts_ns");

        // 2. Event Type Specific Parsing
        if (event.type == "kernel") {
            event.kernelName = JsonUtils::extractString(line, "kernel");
            event.tsStartNs = JsonUtils::extractInt(line, "ts_start_ns");
            event.tsEndNs = JsonUtils::extractInt(line, "ts_end_ns");
            event.durationNs = JsonUtils::extractInt(line, "duration_ns");
        }
        else if (event.type == "scope_begin" || event.type == "scope_end") {
            event.scopeName = JsonUtils::extractString(line, "name");
            event.tsStartNs = JsonUtils::extractInt(line, "ts_start_ns");
            event.tsEndNs = JsonUtils::extractInt(line, "ts_end_ns");
            event.durationNs = JsonUtils::extractInt(line, "duration_ns");

            // 3. Memory Parsing (Consolidated)
            if (event.type == "scope_begin") {
                event.memStartUsedMiB = JsonUtils::extractNestedInt(line, "memory", "used_mib");
                event.deviceId = static_cast<int32_t>(JsonUtils::extractNestedInt(line, "memory", "device"));
            } else { // scope_end
                event.memEndUsedMiB = JsonUtils::extractNestedInt(line, "memory", "used_mib");

                // Try to extract device ID if we don't have it (fallback)
                if (event.deviceId == -1) {
                    event.deviceId = static_cast<int32_t>(JsonUtils::extractNestedInt(line, "memory", "device"));
                }

                // It will be 0 here, but can be calculated later by comparing start/end events if needed.
                event.memDeltaMiB = 0;
            }
        }

        return event;
    }

    size_t ClientLogReader::readNewEvents() {
        std::ifstream file(logFilePath_, std::ios::binary);
        if (!file.is_open()) return 0;

        file.seekg(0, std::ios::end);
        if (const std::streampos currentSize = file.tellg(); currentSize < lastPosition_) {
            if (debugMode_) {
                std::cout << "[ClientLogReader] Log rotation detected. Resetting cursor." << std::endl;
            }
            lastPosition_ = 0;
        }

        file.seekg(lastPosition_);

        size_t count = 0;
        std::string line;
        while (std::getline(file, line)) {
            if (auto event = parseLine(line); event.has_value()) {
                if (debugMode_) {
                    std::cout << "[LogEvent] PID:" << event->pid << " Type:" << event->type
                          << " App:" << event->appName;
                    if (!event->kernelName.empty()) std::cout << " Kernel:" << event->kernelName;
                    if (!event->scopeName.empty()) std::cout << " Scope:" << event->scopeName;

                    if (event->type == "scope_begin") {
                        std::cout << " MemStart:" << event->memStartUsedMiB << " Dev:" << event->deviceId;
                    } else if (event->type == "scope_end") {
                        std::cout << " MemEnd:" << event->memEndUsedMiB;
                    }
                    std::cout << std::endl;
                }
                cachedEvents_.push_back(event.value());
                count++;
            }
        }

        if (file.eof()) file.clear();
        lastPosition_ = file.tellg();
        return count;
    }

    std::map<int32_t, std::vector<ClientEvent> > ClientLogReader::getAllProcessEvents(int64_t startNs, int64_t endNs) const {
        std::map<int32_t, std::vector<ClientEvent>> result;

        for (const auto& event : cachedEvents_) {
            // For durations, check overlaps. For points, check timestamp.
            int64_t evtTime = event.tsNs;

            // Use start time for ranges to determine if they started in this window
            if (event.tsStartNs > 0) evtTime = event.tsStartNs;

            if (evtTime >= startNs && evtTime <= endNs) {
                result[event.pid].push_back(event);
            }
        }

        return result;
    }

    void ClientLogReader::pruneOldEvents(int64_t olderThanNs) {
        if (cachedEvents_.empty()) return;

        const auto it = std::remove_if(cachedEvents_.begin(), cachedEvents_.end(), [olderThanNs](const ClientEvent& e) {
            const int64_t timeToCheck = (e.tsEndNs >0) ? e.tsEndNs : e.tsNs;
            return timeToCheck < olderThanNs;
        });

        if (it != cachedEvents_.end()) {
            if (debugMode_) std::cout << "[ClientLogReader] Pruned " << std::distance(it, cachedEvents_.end()) << " old events." << std::endl;
            cachedEvents_.erase(it, cachedEvents_.end());
        }
    }
}
