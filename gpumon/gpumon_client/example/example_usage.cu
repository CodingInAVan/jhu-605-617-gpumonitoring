#include <gpumon/gpumon.hpp>
#include <iostream>
#include <cuda_runtime.h>

__global__
void vectorAdd(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__
void vectorMul(int* a, int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

int main() {
    // Initialize GPUmon
    gpumon::InitOptions opts;
    opts.appName = "example_app";
    opts.logFilePath = "gpumon.log";
    
    if (!gpumon::init(opts)) {
        std::cerr << "Failed to initialize gpumon" << std::endl;
        return 1;
    }
    
    std::cout << "GPUmon initialized. Logs will be written to: " << opts.logFilePath << std::endl;
    
    const int n = 1024;
    const size_t bytes = n * sizeof(int);
    
    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Allocate and initialize host memory
    int* h_a = new int[n];
    int* h_b = new int[n];
    int* h_c = new int[n];
    
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // Copy to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Begin a region
    gpumon::beginRegion("computation");
    
    // Launch kernels with monitoring
    dim3 grid(4);
    dim3 block(256);
    
    std::cout << "Launching vectorAdd kernel..." << std::endl;
    GPUMON_LAUNCH(vectorAdd, grid, block, 0, 0, d_a, d_b, d_c, n);
    
    std::cout << "Launching vectorMul kernel..." << std::endl;
    GPUMON_LAUNCH(vectorMul, grid, block, 0, 0, d_a, d_b, d_c, n);
    
    // End the region
    gpumon::endRegion();
    
    // Copy results back
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // Verify results (simple check)
    std::cout << "First result: " << h_c[0] << " (expected: 0)" << std::endl;
    std::cout << "Second result: " << h_c[1] << " (expected: 2)" << std::endl;
    
    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // Shutdown GPUmon
    gpumon::shutdown();
    
    std::cout << "GPUmon shutdown complete." << std::endl;
    
    return 0;
}
