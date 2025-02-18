#pragma once

#include "../stream.hpp"
#include <hip/hip_runtime.h>

namespace deeppowers {
namespace hal {

class ROCmDevice;

/**
 * ROCm stream implementation for asynchronous operations
 */
class ROCmStream : public Stream {
public:
    /**
     * Create a new ROCm stream
     * @param device ROCm device pointer
     */
    explicit ROCmStream(ROCmDevice* device);
    ~ROCmStream() override;

    // Stream interface implementations
    void synchronize() override;
    bool query() override;
    
    // Memory operations
    void memcpy_host_to_device_async(void* dst, const void* src, size_t size) override;
    void memcpy_device_to_host_async(void* dst, const void* src, size_t size) override;
    void memcpy_device_to_device_async(void* dst, const void* src, size_t size) override;
    void memset_async(void* ptr, int value, size_t size) override;

    // ROCm specific methods
    hipStream_t rocm_stream() const { return stream_; }

private:
    ROCmDevice* device_;
    hipStream_t stream_;
};

}} // namespace deeppowers::hal 