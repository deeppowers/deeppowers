#include "cuda_stream.hpp"
#include "cuda_device.hpp"

namespace deeppowers {
namespace hal {

CUDAStream::CUDAStream(int device_id) : device_id_(device_id) {
    CUDA_CHECK(cudaSetDevice(device_id_));
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

CUDAStream::~CUDAStream() {
    try {
        if (stream_) {
            CUDA_CHECK(cudaSetDevice(device_id_));
            CUDA_CHECK(cudaStreamDestroy(stream_));
        }
    } catch (...) {
        // Destructor should not throw exceptions
    }
}

void CUDAStream::synchronize() {
    CUDA_CHECK(cudaSetDevice(device_id_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

bool CUDAStream::query() {
    CUDA_CHECK(cudaSetDevice(device_id_));
    cudaError_t status = cudaStreamQuery(stream_);
    if (status == cudaSuccess) {
        return true;
    } else if (status == cudaErrorNotReady) {
        return false;
    } else {
        CUDA_CHECK(status);
        return false;  // Should not reach here
    }
}

} // namespace hal
} // namespace deeppowers