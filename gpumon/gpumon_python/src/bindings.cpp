#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define GPUMON_BACKEND_CUDA // Python users likely on CUDA
#include "gpumon/gpumon.hpp"

namespace py = pybind11;

class PyScope {
public:
    PyScope(std::string name, std::string tag)
        : name_(std::move(name)), tag_(std::move(tag)), monitor_(nullptr) {}

    // Called when entering 'with gpumon.Scope(...):'
    void enter() {
        // Create the C++ object. It logs "scope_begin" immediately.
        monitor_ = new gpumon::ScopedMonitor(name_, tag_);
    }

    // Called when exiting 'with' block
    void exit(py::object type, py::object value, py::object traceback) {
        // Deleting the object triggers the destructor.
        // It logs "scope_end" and calls cudaDeviceSynchronize().
        if (monitor_) {
            delete monitor_;
            monitor_ = nullptr;
        }
    }

private:
    std::string name_;
    std::string tag_;
    gpumon::ScopedMonitor* monitor_;
};

PYBIND11_MODULE(gpumon_py, m) {
    m.doc() = "Python bindings for GPUMon Client";

    // 1. Init Function
    m.def("init", [](std::string appName, std::string logPath, int sampleIntervalMs) {
        gpumon::InitOptions opts;
        opts.appName = appName;
        opts.logFilePath = logPath;
        opts.sampleIntervalMs = sampleIntervalMs;
        return gpumon::init(opts);
    }, py::arg("app_name"), py::arg("log_path") = "", py::arg("sample_interval_ms") = 0);

    // 2. Shutdown Function
    m.def("shutdown", &gpumon::shutdown);

    // 3. Scope Context Manager
    py::class_<PyScope>(m, "Scope")
        .def(py::init<std::string, std::string>(), py::arg("name"), py::arg("tag") = "")
        .def("__enter__", &PyScope::enter)
        .def("__exit__", &PyScope::exit);
}