#include "rocm_stream.hpp"
#include "rocm_device.hpp"
#include <stdexcept>

namespace deeppowers {
namespace hal {

ROCmStream::ROCmStream(ROCmDevice* device)
    : device_(device)
    , stream_(nullptr) {
    ROCM_CHECK(hipStreamCreate(&stream_));
}

ROCmStream::~ROCmStream() {
    if (stream_) {
        ROCM_CHECK(hipStreamDestroy(stream_));
    }
}

void ROCmStream::synchronize() {
    ROCM_CHECK(hipStreamSynchronize(stream_));
}

bool ROCmStream::query() {
    hipError_t status = hipStreamQuery(stream_);
    if (status == hipSuccess) {
        return true;
    } else if (status == hipErrorNotReady) {
        return false;
    } else {
        ROCM_CHECK(status);
        return false;
    }
}

void ROCmStream::memcpy_host_to_device_async(void* dst, const void* src, size_t size) {
    ROCM_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice, stream_));
}

void ROCmStream::memcpy_device_to_host_async(void* dst, const void* src, size_t size) {
    ROCM_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToHost, stream_));
}

void ROCmStream::memcpy_device_to_device_async(void* dst, const void* src, size_t size) {
    ROCM_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToDevice, stream_));
}

void ROCmStream::memset_async(void* ptr, int value, size_t size) {
    ROCM_CHECK(hipMemsetAsync(ptr, value, size, stream_));
}

}} // namespace deeppowers::hal 