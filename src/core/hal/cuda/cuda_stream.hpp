#pragma once

#include <cuda_runtime.h>
#include "../hal.hpp"

namespace deeppowers {
namespace hal {

class CUDAStream : public Stream {
public:
    explicit CUDAStream(int device_id);
    ~CUDAStream() override;

    // Stream interface implementation
    void synchronize() override;
    bool query() override;

    // CUDA specific methods
    cudaStream_t get_cuda_stream() const { return stream_; }

private:
    int device_id_;
    cudaStream_t stream_;
};

} // namespace hal
} // namespace deeppowers 