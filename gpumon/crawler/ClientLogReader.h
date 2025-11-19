#pragma once
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <optional>
#include <cstdint>

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

    // Common optional field
    std::string tag;
};

class ClientLogReader {
public:
    explicit ClientLogReader(std::string  logFilePath);

    // Read all new events from the log file (since last read)
    std::vector<ClientEvent> readNewEvents();

    // Get events for a specific PID within a time window
    std::vector<ClientEvent> getEventsForPid(int32_t pid, int64_t startNs, int64_t endNs) const;

    // Check if log file exists and is readable
    bool isValid() const;

private:
    std::string logFilePath_;
    std::streampos lastPosition_;

    // Parse a single NDJSON line into a ClientEvent
    static std::optional<ClientEvent> parseLine(const std::string& line);

    // Helper to extract JSON string value
    static std::string extractJsonString(const std::string& json, const std::string& key);

    // Helper to extract JSON integer value
    static int64_t extractJsonInt(const std::string& json, const std::string& key);
};
