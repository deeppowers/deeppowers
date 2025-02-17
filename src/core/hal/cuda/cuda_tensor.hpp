#pragma once

#include "../hal.hpp"
#include "cuda_device.hpp"
#include <memory>

namespace deeppowers {
namespace hal {

class CUDATensor : public Tensor {
public:
    CUDATensor(CUDADevice* device,
               const std::vector<int64_t>& shape,
               DataType dtype);
    
    ~CUDATensor() override;

    // Tensor interface implementation
    const std::vector<int64_t>& shape() const override { return shape_; }
    DataType dtype() const override { return dtype_; }
    size_t size_in_bytes() const override { return size_in_bytes_; }
    void* data() override { return data_; }
    const void* data() const override { return data_; }

    void copy_from_host(const void* src) override;
    void copy_to_host(void* dst) const override;

    // CUDA specific methods
    CUDADevice* device() const { return device_; }
    
    // Helper methods
    static size_t compute_size_in_bytes(const std::vector<int64_t>& shape, DataType dtype);
    static size_t get_element_size(DataType dtype);

private:
    CUDADevice* device_;              // Owned device
    std::vector<int64_t> shape_;      // Tensor shape
    DataType dtype_;                  // Data type
    void* data_;                      // Device memory pointer
    size_t size_in_bytes_;            // Total number of bytes
    
    // Compute total number of elements
    size_t compute_num_elements() const;
};

} // namespace hal
} // namespace deeppowers 