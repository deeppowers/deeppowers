#pragma once

#include <cuda_runtime.h>
#include "../hal.hpp"
#include <unordered_map>

namespace deeppowers {
namespace hal {

class CUDADevice : public Device {
public:
    explicit CUDADevice(int device_id);
    ~CUDADevice() override;

    // Device interface implementation
    DeviceType type() const override { return DeviceType::CUDA; }
    std::string name() const override;
    size_t total_memory() const override;
    size_t available_memory() const override;

    void* allocate(size_t size) override;
    void deallocate(void* ptr) override;
    void memcpy_to_device(void* dst, const void* src, size_t size) override;
    void memcpy_to_host(void* dst, const void* src, size_t size) override;
    void synchronize() override;
    std::shared_ptr<Stream> create_stream() override;

    // CUDA specific methods
    int device_id() const { return device_id_; }
    cudaStream_t get_default_stream() const { return default_stream_; }

private:
    int device_id_;
    cudaStream_t default_stream_;
    std::unordered_map<void*, size_t> allocations_;  // Track memory allocations
    
    // Error checking helper function
    static void check_cuda_error(cudaError_t error, const char* file, int line);
};

// Error checking macro
#define CUDA_CHECK(err) CUDADevice::check_cuda_error(err, __FILE__, __LINE__)

} // namespace hal
} // namespace deeppowers 