#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace deeppowers {
namespace hal {

// Forward declarations
class Device;
class Tensor;
class Kernel;
class Stream;

// Device type enum
enum class DeviceType {
    CPU,
    CUDA,
    ROCM,
    METAL,
    ONEAPI
};

// Data type enum
enum class DataType {
    FLOAT32,
    FLOAT16,
    INT8,
    INT32,
    BOOL
};

// Device abstract base class
class Device {
public:
    virtual ~Device() = default;

    // Device information
    virtual DeviceType type() const = 0;
    virtual std::string name() const = 0;
    virtual size_t total_memory() const = 0;
    virtual size_t available_memory() const = 0;

    // Memory management
    virtual void* allocate(size_t size) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void memcpy_to_device(void* dst, const void* src, size_t size) = 0;
    virtual void memcpy_to_host(void* dst, const void* src, size_t size) = 0;

    // Device synchronization
    virtual void synchronize() = 0;

    // Stream management
    virtual std::shared_ptr<Stream> create_stream() = 0;
};

// Tensor abstract base class
class Tensor {
public:
    virtual ~Tensor() = default;

    // Tensor information
    virtual const std::vector<int64_t>& shape() const = 0;
    virtual DataType dtype() const = 0;
    virtual size_t size_in_bytes() const = 0;
    virtual void* data() = 0;
    virtual const void* data() const = 0;

    // Data operations
    virtual void copy_from_host(const void* src) = 0;
    virtual void copy_to_host(void* dst) const = 0;
};

// Kernel abstract base class
class Kernel {
public:
    virtual ~Kernel() = default;

    // Kernel information
    virtual std::string name() const = 0;
    
    // Parameter setting
    virtual void set_arg(int index, size_t size, const void* value) = 0;
    
    // Launch configuration
    struct LaunchConfig {
        std::vector<size_t> grid_dim;
        std::vector<size_t> block_dim;
        size_t shared_memory_bytes;
    };

    // Kernel execution
    virtual void launch(const LaunchConfig& config, Stream* stream = nullptr) = 0;
};

// Stream abstract base class
class Stream {
public:
    virtual ~Stream() = default;

    // Stream operations
    virtual void synchronize() = 0;
    virtual bool query() = 0;  // Check if all operations in the stream are complete
};

} // namespace hal
} // namespace deeppowers