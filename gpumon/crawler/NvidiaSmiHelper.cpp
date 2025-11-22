#include "NvidiaSmiHelper.h"
#include <cstdio>
#include <sstream>
#include <iostream>

// Helper function to trim whitespace from a string
static void trim(std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end = s.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) {
        s.clear();
    } else {
        s = s.substr(start, end - start + 1);
    }
}

std::vector<SmiProcessInfo> queryProcessMemoryViaSmi(unsigned int deviceIndex) {
    std::vector<SmiProcessInfo> result;

    // Build the nvidia-smi command
    // Using "-q -d PIDS" provides detailed process information in text format
    std::ostringstream cmd;
    cmd << "nvidia-smi -q -i " << deviceIndex << " -d PIDS";

    std::string cmdStr = cmd.str();
    std::cout << "[DEBUG] Executing: " << cmdStr << std::endl;

    // Execute command using _popen (Windows-compatible)
    FILE* pipe = _popen(cmdStr.c_str(), "r");
    if (!pipe) {
        std::cerr << "[WARNING] Failed to execute nvidia-smi for GPU " << deviceIndex << std::endl;
        return result;
    }

    // Read all output into a buffer
    std::string fullOutput;
    char buffer[512];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        fullOutput += buffer;
    }

    int exitCode = _pclose(pipe);

    if (exitCode != 0 || fullOutput.empty()) {
        std::cout << "[INFO] nvidia-smi did not return process data for GPU " << deviceIndex << std::endl;
        return result;
    }

    std::istringstream outputStream(fullOutput);
    std::string line;
    int validEntries = 0;
    int skippedEntries = 0;

    SmiProcessInfo currentProcess;
    bool inProcessBlock = false;

    while (std::getline(outputStream, line)) {
        trim(line);
        if (line.empty()) continue;

        // Check if we're starting a new process block
        if (line.find("Process ID") != std::string::npos) {
            // Save previous process if it was valid
            if (inProcessBlock && currentProcess.pid != 0 && currentProcess.usedMemoryMiB > 0) {
                result.push_back(currentProcess);
                validEntries++;
                std::cout << "[DEBUG] Parsed: PID=" << currentProcess.pid
                          << ", Name=" << currentProcess.processName
                          << ", Memory=" << currentProcess.usedMemoryMiB << " MiB" << std::endl;
            } else if (inProcessBlock) {
                skippedEntries++;
            }

            // Start new process
            currentProcess = SmiProcessInfo{0, "", 0};
            inProcessBlock = true;

            // Extract PID
            if (size_t colonPos = line.find(':'); colonPos != std::string::npos) {
                std::string pidStr = line.substr(colonPos + 1);
                trim(pidStr);
                try {
                    currentProcess.pid = static_cast<unsigned int>(std::stoul(pidStr));
                } catch (...) {}
            }
        }
        else if (inProcessBlock && line.find("Process Name") != std::string::npos) {
            if (size_t colonPos = line.find(':'); colonPos != std::string::npos) {
                std::string procName = line.substr(colonPos + 1);
                trim(procName);
                currentProcess.processName = procName;
            }
        }
        else if (inProcessBlock && line.find("Used GPU Memory") != std::string::npos) {
            size_t colonPos = line.find(':');
            if (colonPos != std::string::npos) {
                std::string memStr = line.substr(colonPos + 1);
                trim(memStr);

                // Skip N/A or Not Supported
                if (memStr.find("N/A") != std::string::npos ||
                    memStr.find("Not Supported") != std::string::npos) {
                    continue;
                }

                // Remove " MiB" suffix if present
                size_t mibPos = memStr.find(" MiB");
                if (mibPos != std::string::npos) {
                    memStr = memStr.substr(0, mibPos);
                    trim(memStr);
                }

                try {
                    currentProcess.usedMemoryMiB = std::stoull(memStr);
                } catch (...) {}
            }
        }
    }

    if (inProcessBlock && currentProcess.pid != 0 && currentProcess.usedMemoryMiB > 0) {
        result.push_back(currentProcess);
        validEntries++;
        std::cout << "[DEBUG] Parsed: PID=" << currentProcess.pid
                  << ", Name=" << currentProcess.processName
                  << ", Memory=" << currentProcess.usedMemoryMiB << " MiB" << std::endl;
    } else if (inProcessBlock) {
        skippedEntries++;
    }

    std::cout << "[DEBUG] Parsed " << validEntries << " process(es), skipped " << skippedEntries
              << " (memory N/A)" << std::endl;

    if (result.empty() && skippedEntries > 0) {
        std::cout << "[INFO] Per-process GPU memory is not available on this system (Windows WDDM limitation)." << std::endl;
    }

    return result;
}
