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
    }

    return event;
}

std::vector<ClientEvent> ClientLogReader::readNewEvents() {
    std::vector<ClientEvent> events;

    std::ifstream file(logFilePath_, std::ios::binary);
    if (!file.is_open()) {
        return events;
    }

    // Seek to last read position
    file.seekg(lastPosition_);

    std::string line;
    while (std::getline(file, line)) {
        auto event = parseLine(line);
        if (event.has_value()) {
            events.push_back(event.value());
        }
    }

    // Update last position
    lastPosition_ = file.tellg();

    return events;
}

std::vector<ClientEvent> ClientLogReader::getEventsForPid(const int32_t pid,
    const int64_t startNs, const int64_t endNs) const {
    std::vector<ClientEvent> matchingEvents;

    std::ifstream file(logFilePath_, std::ios::binary);
    if (!file.is_open()) {
        return matchingEvents;
    }

    std::string line;
    while (std::getline(file, line)) {
        auto event = parseLine(line);
        if (!event.has_value()) continue;

        // Filter by PID
        if (event->pid != pid) continue;

        // Filter by timestamp window
        int64_t eventTime = event->tsNs;
        if (event->type == "kernel" || event->type == "scope_end") {
            // Use start time for range check
            eventTime = event->tsStartNs;
        }

        if (eventTime >= startNs && eventTime <= endNs) {
            matchingEvents.push_back(event.value());
        }
    }

    return matchingEvents;
}
