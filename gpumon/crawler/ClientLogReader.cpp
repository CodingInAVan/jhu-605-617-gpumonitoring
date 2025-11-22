#include "ClientLogReader.h"
#include <iostream>
#include <utility>

ClientLogReader::ClientLogReader(std::string  logFilePath)
    : logFilePath_(std::move(logFilePath)), lastPosition_(0) {
}

bool ClientLogReader::isValid() const {
    const std::ifstream file(logFilePath_);
    return file.good();
}

std::string ClientLogReader::extractJsonString(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\":\"";
    const size_t pos = json.find(needle);
    if (pos == std::string::npos) return "";

    size_t start = pos + needle.length();
    size_t end = start;

    // Handle escaped quotes
    while (end < json.length()) {
        if (json[end] == '\\' && end + 1 < json.length()) {
            end += 2; // Skip escaped character
            continue;
        }
        if (json[end] == '"') {
            break;
        }
        end++;
    }

    return json.substr(start, end - start);
}

int64_t ClientLogReader::extractJsonInt(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\":";
    const size_t pos = json.find(needle);
    if (pos == std::string::npos) return 0;

    const size_t start = pos + needle.length();
    size_t end = start;

    // Find end of number (comma, space, or closing brace)
    while (end < json.length() &&
           json[end] != ',' &&
           json[end] != '}' &&
           json[end] != ' ' &&
           json[end] != '\r' &&
           json[end] != '\n') {
        end++;
    }

    const std::string numStr = json.substr(start, end - start);
    try {
        return std::stoll(numStr);
    } catch (...) {
        return 0;
    }
}

std::optional<ClientEvent> ClientLogReader::parseLine(const std::string& line) {
    if (line.empty() || line[0] != '{') return std::nullopt;

    ClientEvent event;
    std::cout << "[ClientLogReader] Parsing line: " << line << std::endl;
    // Extract common fields
    event.type = extractJsonString(line, "type");
    event.pid = static_cast<int32_t>(extractJsonInt(line, "pid"));
    event.appName = extractJsonString(line, "app");
    event.tag = extractJsonString(line, "tag");

    // Extract timestamp (varies by event type)
    if (line.find("\"ts_ns\":") != std::string::npos) {
        event.tsNs = extractJsonInt(line, "ts_ns");
    }

    // Extract type-specific fields
    if (event.type == "kernel") {
        event.kernelName = extractJsonString(line, "kernel");
        event.tsStartNs = extractJsonInt(line, "ts_start_ns");
        event.tsEndNs = extractJsonInt(line, "ts_end_ns");
        event.durationNs = extractJsonInt(line, "duration_ns");
    } else if (event.type == "region_begin" || event.type == "region_end") {
        event.regionName = extractJsonString(line, "name");
    } else if (event.type == "scope_begin" || event.type == "scope_end") {
        event.scopeName = extractJsonString(line, "name");
        event.tsStartNs = extractJsonInt(line, "ts_start_ns");
        event.tsEndNs = extractJsonInt(line, "ts_end_ns");
        event.durationNs = extractJsonInt(line, "duration_ns");

        // Extract memory data (simplified: use first device if multiple)
        // mem_start:[{"device":0,"used_mib":1024,...}]
        if (event.type == "scope_begin" && line.find("\"mem_start\":") != std::string::npos) {
            // Simple extraction: find first "used_mib" value in mem_start array
            size_t memStartPos = line.find("\"mem_start\":");
            if (memStartPos != std::string::npos) {
                size_t usedPos = line.find("\"used_mib\":", memStartPos);
                if (usedPos != std::string::npos) {
                    size_t colonPos = usedPos + 11; // length of "used_mib":
                    // Skip whitespace after colon
                    while (colonPos < line.length() && (line[colonPos] == ' ' || line[colonPos] == '\t')) {
                        colonPos++;
                    }
                    size_t endPos = colonPos;
                    while (endPos < line.length() && (std::isdigit(line[endPos]) || line[endPos] == '-')) {
                        endPos++;
                    }
                    if (endPos > colonPos) {
                        std::string valStr = line.substr(colonPos, endPos - colonPos);
                        try {
                            event.memStartUsedMiB = std::stoll(valStr);
                            std::cout << "[ClientLogReader] Parsed mem_start used_mib: " << event.memStartUsedMiB << " from: " << valStr << std::endl;
                        } catch (...) {
                            std::cerr << "[ClientLogReader] Failed to parse mem_start used_mib from: " << valStr << std::endl;
                        }
                    }
                }
            }
        }

        // mem_end:[{"device":0,"used_mib":2048,...}]
        if (event.type == "scope_end" && line.find("\"mem_end\":") != std::string::npos) {
            std::cout << "line = " << line << std::endl;

            size_t memEndPos = line.find("\"mem_end\":");
            if (memEndPos != std::string::npos) {
                size_t usedPos = line.find("\"used_mib\":", memEndPos);
                if (usedPos != std::string::npos) {
                    size_t colonPos = usedPos + 11;
                    // Skip whitespace after colon
                    while (colonPos < line.length() && (line[colonPos] == ' ' || line[colonPos] == '\t')) {
                        colonPos++;
                    }
                    size_t endPos = colonPos;
                    while (endPos < line.length() && (std::isdigit(line[endPos]) || line[endPos] == '-')) {
                        endPos++;
                    }
                    if (endPos > colonPos) {
                        std::string valStr = line.substr(colonPos, endPos - colonPos);
                        try {
                            event.memEndUsedMiB = std::stoll(valStr);
                            std::cout << "[ClientLogReader] Parsed mem_end used_mib: " << event.memEndUsedMiB << " from: " << valStr << std::endl;
                        } catch (...) {
                            std::cerr << "[ClientLogReader] Failed to parse mem_end used_mib from: " << valStr << std::endl;
                        }
                    }
                }
            }
        }

        // mem_delta:[{"device":0,"delta_mib":512}]
        if (event.type == "scope_end" && line.find("\"mem_delta\":") != std::string::npos) {
            size_t memDeltaPos = line.find("\"mem_delta\":");
            if (memDeltaPos != std::string::npos) {
                size_t deltaPos = line.find("\"delta_mib\":", memDeltaPos);
                if (deltaPos != std::string::npos) {
                    size_t colonPos = deltaPos + 12; // length of "delta_mib":
                    // Skip whitespace after colon
                    while (colonPos < line.length() && (line[colonPos] == ' ' || line[colonPos] == '\t')) {
                        colonPos++;
                    }
                    size_t endPos = colonPos;
                    while (endPos < line.length() && (std::isdigit(line[endPos]) || line[endPos] == '-')) {
                        endPos++;
                    }
                    if (endPos > colonPos) {
                        std::string valStr = line.substr(colonPos, endPos - colonPos);
                        try {
                            event.memDeltaMiB = std::stoll(valStr);
                            std::cout << "[ClientLogReader] Parsed mem_delta delta_mib: " << event.memDeltaMiB << " from: " << valStr << std::endl;
                        } catch (...) {
                            std::cerr << "[ClientLogReader] Failed to parse mem_delta delta_mib from: " << valStr << std::endl;
                        }
                    }
                }
            }
        }
    }

    return event;
}

size_t ClientLogReader::readNewEvents() {
    size_t newEventCount = 0;

    std::ifstream file(logFilePath_, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ClientLogReader::readNewEvents] Failed to open: " << logFilePath_ << std::endl;
        return 0;
    }

    // Seek to last read position
    std::cout << "[ClientLogReader::readNewEvents] Reading from position: " << lastPosition_
              << " in file: " << logFilePath_ << std::endl;
    file.seekg(lastPosition_);

    std::string line;
    while (std::getline(file, line)) {
        std::cout << "[ClientLogReader::readNewEvents] Read line: " << line << std::endl;
        auto event = parseLine(line);
        if (event.has_value()) {
            cachedEvents_.push_back(event.value());
            newEventCount++;
            std::cout << "[ClientLogReader::readNewEvents] Added event to cache: type=" << event->type
                      << ", pid=" << event->pid << std::endl;
        }
    }

    // Update last position
    if (file.eof()) {
        file.clear();
    }
    lastPosition_ = file.tellg();
    std::cout << "[ClientLogReader::readNewEvents] Read " << newEventCount
              << " new events. Total cached: " << cachedEvents_.size()
              << ". New position: " << lastPosition_ << std::endl;

    return newEventCount;
}

std::map<int32_t, std::vector<ClientEvent>> ClientLogReader::getAllProcessEvents(
    const int64_t startNs, const int64_t endNs) {
    std::map<int32_t, std::vector<ClientEvent>> processEvents;

    std::cout << "[ClientLogReader::getAllProcessEvents] Searching " << cachedEvents_.size()
              << " cached events in time range: " << startNs << " to " << endNs << std::endl;

    // Group all events by PID within the time window
    for (const auto& event : cachedEvents_) {
        // Filter by timestamp window
        int64_t eventTime = event.tsNs;
        if (event.type == "kernel" || event.type == "scope_end") {
            // Use start time for range check
            eventTime = event.tsStartNs;
        }

        if (eventTime >= startNs && eventTime <= endNs) {
            processEvents[event.pid].push_back(event);
        }
    }

    std::cout << "[ClientLogReader::getAllProcessEvents] Found events for " << processEvents.size()
              << " processes" << std::endl;
    for (const auto& [pid, events] : processEvents) {
        std::cout << "  PID " << pid << ": " << events.size() << " events" << std::endl;
    }

    return processEvents;
}
