#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>
#include "Utils.h"
#include "MetricsSender.h"
#include "GpuMonitor.h"
#include "Config.h"
#ifdef _WIN32
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

static Channel channelFromEnv() {
    // Default to HTTP posting to /metrics
    return Channel::HTTP;
}

// Parse clientlib log paths from environment variable or default locations
static std::vector<std::string> getClientLogPaths() {
    std::vector<std::string> logPaths;

    // Check environment variable for explicit paths (comma-separated)
    std::string envPaths = util::getenvOr("GPUMON_CLIENT_LOGS", "");
    if (!envPaths.empty()) {
        std::stringstream ss(envPaths);
        std::string path;
        while (std::getline(ss, path, ',')) {
            // Trim whitespace
            size_t start = path.find_first_not_of(" \t\r\n");
            size_t end = path.find_last_not_of(" \t\r\n");
            if (start != std::string::npos) {
                path = path.substr(start, end - start + 1);
                logPaths.push_back(path);
            }
        }
    }

    // If no explicit paths, check default locations
    if (logPaths.empty()) {
        std::vector<std::string> defaultPaths = {
            "gpumon.log",                    // Current directory
            "./gpumon.log",
        };

#ifdef _WIN32
        const char* appdata = std::getenv("APPDATA");
        if (appdata) {
            defaultPaths.push_back(std::string(appdata) + "\\gpumon\\gpumon.log");
        }
        defaultPaths.push_back("C:\\ProgramData\\gpumon\\gpumon.log");
#else
        const char* home = std::getenv("HOME");
        if (home) {
            defaultPaths.push_back(std::string(home) + "/.gpumon/gpumon.log");
        }
        defaultPaths.push_back("/var/log/gpumon/gpumon.log");
#endif

        // Only add paths that exist
        for (const auto& path : defaultPaths) {
            if (fs::exists(path)) {
                logPaths.push_back(path);
            }
        }
    }

    return logPaths;
}

static std::string buildSelfTestSampleJson(const std::string& hostname) {
    // Minimal valid metric per backend expectations
    std::ostringstream j;
    j << "{";
    j << "\"timestamp\":\"" << util::nowIso8601Utc() << "\",";
    j << "\"metric\":\"power\",";
    j << "\"hostname\":\"" << util::escapeJson(hostname) << "\",";
    j << "\"gpuName\":\"GPU\",";
    j << "\"watts\":100.0";
    j << "}";
    return j.str();
}

int main(int argc, char** argv) {
    try {
        // Resolve settings
        Settings settings = config::resolveFromSources(argc, argv);

        if (settings.setKey) {
            // Force re-prompt and overwrite stored key
            settings.apiKey.clear();
            settings = config::interactiveSetup(settings);
            std::cout << "API key saved. Ingestion enabled." << std::endl;
            if (!settings.selfTest) return 0; // if not continuing with self-test
        }

        if (settings.apiKey.empty()) {
            // Interactive first run prompt
            settings = config::interactiveSetup(settings);
        }

        // Prepare sender
        auto sender = makeSender(channelFromEnv(), settings.backendUrl, settings.apiKey);

        if (settings.selfTest) {
            // Send a single metric and print response behavior (HTTP sender prints errors/status)
            std::string host = "host";
            char buf[256];
            if (gethostname(buf, sizeof(buf)) == 0) host = buf;
            std::string sample = buildSelfTestSampleJson(host);
            sender->send(sample);
            std::cout << "[OK] Self-test request sent." << std::endl;
            return 0;
        }

        // Normal run
        // Get clientlib log paths for enrichment
        std::vector<std::string> logPaths = getClientLogPaths();
        if (!logPaths.empty()) {
            std::cout << "Will monitor " << logPaths.size() << " clientlib log file(s) for enrichment:" << std::endl;
            for (const auto& path : logPaths) {
                std::cout << "  - " << path << std::endl;
            }
        } else {
            std::cout << "No clientlib log files found. Process metrics will not be enriched." << std::endl;
            std::cout << "Set GPUMON_CLIENT_LOGS environment variable to specify log file paths." << std::endl;
        }

        GpuMonitor monitor(std::move(sender), logPaths);
        monitor.runLoop();
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << std::endl;
        return 1;
    }
}
