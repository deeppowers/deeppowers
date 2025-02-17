#include "cuda_device.hpp"
#include "cuda_stream.hpp"
#include <stdexcept>
#include <sstream>

namespace deeppowers {
namespace hal {

void CUDADevice::check_cuda_error(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ":" << line << ": "
            << cudaGetErrorString(error);
        throw std::runtime_error(oss.str());
    }
}

CUDADevice::CUDADevice(int device_id) : device_id_(device_id) {
    // Set current device
    CUDA_CHECK(cudaSetDevice(device_id_));
    
    // Create default stream
    CUDA_CHECK(cudaStreamCreate(&default_stream_));
}

CUDADevice::~CUDADevice() {
    try {
        // Release all allocated memory
        for (const auto& allocation : allocations_) {
            cudaFree(allocation.first);
        }
        allocations_.clear();

        // Destroy default stream
        if (default_stream_) {
            cudaStreamDestroy(default_stream_);
        }
    } catch (...) {
        // Destructor should not throw exceptions
    }
}

std::string CUDADevice::name() const {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id_));
    return prop.name;
}

size_t CUDADevice::total_memory() const {
    size_t total;
    CUDA_CHECK(cudaSetDevice(device_id_));
    CUDA_CHECK(cudaMemGetInfo(nullptr, &total));
    return total;
}

size_t CUDADevice::available_memory() const {
    size_t free;
    CUDA_CHECK(cudaSetDevice(device_id_));
    CUDA_CHECK(cudaMemGetInfo(&free, nullptr));
    return free;
}

void* CUDADevice::allocate(size_t size) {
    CUDA_CHECK(cudaSetDevice(device_id_));
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    allocations_[ptr] = size;
    return ptr;
}

void CUDADevice::deallocate(void* ptr) {
    if (!ptr) return;
    
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        throw std::runtime_error("Attempting to deallocate untracked CUDA memory");
    }
    
    CUDA_CHECK(cudaSetDevice(device_id_));
    CUDA_CHECK(cudaFree(ptr));
    allocations_.erase(it);
}

void CUDADevice::memcpy_to_device(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaSetDevice(device_id_));
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, default_stream_));
}

void CUDADevice::memcpy_to_host(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaSetDevice(device_id_));
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, default_stream_));
}

void CUDADevice::synchronize() {
    CUDA_CHECK(cudaSetDevice(device_id_));
    CUDA_CHECK(cudaDeviceSynchronize());
}

std::shared_ptr<Stream> CUDADevice::create_stream() {
    return std::make_shared<CUDAStream>(device_id_);
}

} // namespace hal
} // namespace deeppowers