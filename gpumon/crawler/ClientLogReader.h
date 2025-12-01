#pragma once
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <optional>
#include <cstdint>

namespace gpumon {

// Structure representing a clientlib event from NDJSON log
struct ClientEvent {
    std::string type;           // "init", "kernel", "region_begin", "region_end", "scope_begin", "scope_end", "shutdown"
    int32_t pid = 0;
    std::string appName;
    int64_t tsNs = 0;           // timestamp in nanoseconds (steady_clock)

    // Kernel-specific fields
    std::string kernelName;
    int64_t tsStartNs = 0;
    int64_t tsEndNs = 0;
    int64_t durationNs = 0;

    // Region/scope-specific fields
    std::string regionName;  // from region_begin/region_end
    std::string scopeName;   // from scope_begin/scope_end

    // Memory tracking fields
    int32_t deviceId = -1;
    int64_t memStartUsedMiB = 0;
    int64_t memEndUsedMiB = 0;
    int64_t memDeltaMiB = 0;

    // Common optional field
    std::string tag;
};

class ClientLogReader {
public:
    explicit ClientLogReader(std::string logFilePath, bool debugMode);
    ~ClientLogReader() = default;

    // Read all new events from the log file (since last read) and cache them
    // Returns the number of new events read
    size_t readNewEvents();

    // Get all process events within a time window from cached events
    // Returns a map: PID -> list of events for that process
    [[nodiscard]] std::map<int32_t, std::vector<ClientEvent>> getAllProcessEvents(int64_t startNs, int64_t endNs) const;

    void pruneOldEvents(int64_t olderThanNs);

    // Check if log file exists and is readable
    [[nodiscard]] bool isValid() const;

    // Get total number of cached events
    [[nodiscard]] size_t getCachedEventCount() const { return cachedEvents_.size(); }

private:
    std::string logFilePath_;
    bool debugMode_;
    std::streampos lastPosition_;
    std::vector<ClientEvent> cachedEvents_;  // Cache of all events read so far

    // Parse a single NDJSON line into a ClientEvent
    static std::optional<ClientEvent> parseLine(const std::string& line);
};

}