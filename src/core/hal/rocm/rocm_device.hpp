#pragma once

#include "../device.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <string>
#include <memory>

namespace deeppowers {
namespace hal {

// Forward declarations
class ROCmStream;
class ROCmEvent;
class ROCmKernel;

/**
 * ROCm device implementation for AMD GPUs
 */
class ROCmDevice : public Device {
public:
    /**
     * Initialize ROCm device with specified device ID
     * @param device_id AMD GPU device ID
     */
    explicit ROCmDevice(int device_id = 0);
    ~ROCmDevice() override;

    // Device interface implementations
    void synchronize() override;
    std::string name() const override;
    std::string vendor() const override;
    size_t total_memory() const override;
    size_t free_memory() const override;
    int compute_capability() const override;
    
    // Memory management
    void* allocate(size_t size) override;
    void deallocate(void* ptr) override;
    void memcpy_host_to_device(void* dst, const void* src, size_t size) override;
    void memcpy_device_to_host(void* dst, const void* src, size_t size) override;
    void memcpy_device_to_device(void* dst, const void* src, size_t size) override;
    void memset(void* ptr, int value, size_t size) override;

    // Stream management
    std::shared_ptr<Stream> create_stream() override;
    std::shared_ptr<Event> create_event() override;

    // Kernel management
    std::shared_ptr<Kernel> create_kernel(const std::string& name,
                                        const std::string& code) override;
    std::shared_ptr<Kernel> load_kernel(const std::string& path) override;

    // ROCm specific methods
    hipDevice_t rocm_device() const { return device_; }
    hipCtx_t rocm_context() const { return context_; }

private:
    int device_id_;
    hipDevice_t device_;
    hipCtx_t context_;
    
    // Initialize ROCm device and context
    void initialize();
    
    // Check ROCm error status
    static void check_rocm_error(hipError_t error, const char* file, int line);
};

// Macro for ROCm error checking
#define ROCM_CHECK(x) ROCmDevice::check_rocm_error(x, __FILE__, __LINE__)

}} // namespace deeppowers::hal 