#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
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

static Channel channelFromEnv() {
    // Default to HTTP posting to /metrics
    return Channel::HTTP;
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
#ifdef _WIN32
            char buf[256];
            if (gethostname(buf, sizeof(buf)) == 0) host = buf;
#else
            char buf[256];
            if (gethostname(buf, sizeof(buf)) == 0) host = buf;
#endif
            std::string sample = buildSelfTestSampleJson(host);
            sender->send(sample);
            std::cout << "[OK] Self-test request sent." << std::endl;
            return 0;
        }

        // Normal run
        GpuMonitor monitor(std::move(sender));
        monitor.runLoop();
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << std::endl;
        return 1;
    }
}
