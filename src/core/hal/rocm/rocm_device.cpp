#include "rocm_device.hpp"
#include "rocm_stream.hpp"
#include "rocm_event.hpp"
#include "rocm_kernel.hpp"
#include <stdexcept>
#include <sstream>

namespace deeppowers {
namespace hal {

ROCmDevice::ROCmDevice(int device_id)
    : device_id_(device_id)
    , device_(nullptr)
    , context_(nullptr) {
    initialize();
}

ROCmDevice::~ROCmDevice() {
    if (context_) {
        ROCM_CHECK(hipCtxDestroy(context_));
    }
}

void ROCmDevice::initialize() {
    // Get number of devices
    int device_count;
    ROCM_CHECK(hipGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        throw std::runtime_error("No ROCm capable devices found");
    }
    
    if (device_id_ >= device_count) {
        throw std::runtime_error("Invalid device ID");
    }
    
    // Set device
    ROCM_CHECK(hipSetDevice(device_id_));
    ROCM_CHECK(hipGetDevice(&device_));
    
    // Create context
    ROCM_CHECK(hipCtxCreate(&context_, 0, device_));
}

void ROCmDevice::synchronize() {
    ROCM_CHECK(hipDeviceSynchronize());
}

std::string ROCmDevice::name() const {
    hipDeviceProp_t props;
    ROCM_CHECK(hipGetDeviceProperties(&props, device_id_));
    return props.name;
}

std::string ROCmDevice::vendor() const {
    return "AMD";
}

size_t ROCmDevice::total_memory() const {
    size_t total;
    ROCM_CHECK(hipMemGetInfo(nullptr, &total));
    return total;
}

size_t ROCmDevice::free_memory() const {
    size_t free;
    ROCM_CHECK(hipMemGetInfo(&free, nullptr));
    return free;
}

int ROCmDevice::compute_capability() const {
    hipDeviceProp_t props;
    ROCM_CHECK(hipGetDeviceProperties(&props, device_id_));
    return props.major * 100 + props.minor * 10;
}

void* ROCmDevice::allocate(size_t size) {
    void* ptr;
    ROCM_CHECK(hipMalloc(&ptr, size));
    return ptr;
}

void ROCmDevice::deallocate(void* ptr) {
    if (ptr) {
        ROCM_CHECK(hipFree(ptr));
    }
}

void ROCmDevice::memcpy_host_to_device(void* dst, const void* src, size_t size) {
    ROCM_CHECK(hipMemcpy(dst, src, size, hipMemcpyHostToDevice));
}

void ROCmDevice::memcpy_device_to_host(void* dst, const void* src, size_t size) {
    ROCM_CHECK(hipMemcpy(dst, src, size, hipMemcpyDeviceToHost));
}

void ROCmDevice::memcpy_device_to_device(void* dst, const void* src, size_t size) {
    ROCM_CHECK(hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice));
}

void ROCmDevice::memset(void* ptr, int value, size_t size) {
    ROCM_CHECK(hipMemset(ptr, value, size));
}

std::shared_ptr<Stream> ROCmDevice::create_stream() {
    return std::make_shared<ROCmStream>(this);
}

std::shared_ptr<Event> ROCmDevice::create_event() {
    return std::make_shared<ROCmEvent>(this);
}

std::shared_ptr<Kernel> ROCmDevice::create_kernel(
    const std::string& name,
    const std::string& code) {
    return std::make_shared<ROCmKernel>(this, name, code);
}

std::shared_ptr<Kernel> ROCmDevice::load_kernel(const std::string& path) {
    return std::make_shared<ROCmKernel>(this, path);
}

void ROCmDevice::check_rocm_error(hipError_t error, const char* file, int line) {
    if (error != hipSuccess) {
        std::ostringstream oss;
        oss << "ROCm error " << error << ": "
            << hipGetErrorString(error)
            << " at " << file << ":" << line;
        throw std::runtime_error(oss.str());
    }
}

}} // namespace deeppowers::hal 