#include "cuda_tensor.hpp"
#include <numeric>
#include <stdexcept>

namespace deeppowers {
namespace hal {

size_t CUDATensor::get_element_size(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32:
            return 4;
        case DataType::FLOAT16:
            return 2;
        case DataType::INT8:
            return 1;
        case DataType::INT32:
            return 4;
        case DataType::BOOL:
            return 1;
        default:
            throw std::runtime_error("Unsupported data type");
    }
}

size_t CUDATensor::compute_size_in_bytes(const std::vector<int64_t>& shape, DataType dtype) {
    size_t num_elements = std::accumulate(shape.begin(), shape.end(), 
                                        static_cast<size_t>(1), std::multiplies<size_t>());
    return num_elements * get_element_size(dtype);
}

size_t CUDATensor::compute_num_elements() const {
    return std::accumulate(shape_.begin(), shape_.end(), 
                          static_cast<size_t>(1), std::multiplies<size_t>());
}

CUDATensor::CUDATensor(CUDADevice* device,
                       const std::vector<int64_t>& shape,
                       DataType dtype)
    : device_(device)
    , shape_(shape)
    , dtype_(dtype)
    , data_(nullptr)
    , size_in_bytes_(compute_size_in_bytes(shape, dtype)) {
    
    if (!device_) {
        throw std::runtime_error("Device cannot be null");
    }

    // Allocate device memory
    data_ = device_->allocate(size_in_bytes_);
}

CUDATensor::~CUDATensor() {
    if (data_) {
        device_->deallocate(data_);
        data_ = nullptr;
    }
}

void CUDATensor::copy_from_host(const void* src) {
    if (!src) {
        throw std::runtime_error("Source pointer cannot be null");
    }
    device_->memcpy_to_device(data_, src, size_in_bytes_);
}

void CUDATensor::copy_to_host(void* dst) const {
    if (!dst) {
        throw std::runtime_error("Destination pointer cannot be null");
    }
    device_->memcpy_to_host(dst, data_, size_in_bytes_);
}

} // namespace hal
} // namespace deeppowers