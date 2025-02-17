#pragma once

#include "../hal.hpp"
#include "cuda_device.hpp"
#include "cuda_stream.hpp"
#include <cuda.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace deeppowers {
namespace hal {

class CUDAKernel : public Kernel {
public:
    CUDAKernel(CUDADevice* device,
               const std::string& name,
               const std::string& ptx_source,
               const std::string& entry_point);
    
    ~CUDAKernel() override;

    // Kernel interface implementation
    std::string name() const override { return name_; }
    void set_arg(int index, size_t size, const void* value) override;
    void launch(const LaunchConfig& config, Stream* stream = nullptr) override;

    // CUDA specific methods
    CUDADevice* device() const { return device_; }
    CUfunction function() const { return function_; }

private:
    CUDADevice* device_;              // Owned device
    std::string name_;                // Kernel name
    CUmodule module_;                 // CUDA module
    CUfunction function_;             // CUDA function
    
    // Parameter cache
    struct ParamInfo {
        size_t size;
        std::vector<uint8_t> data;
    };
    std::unordered_map<int, ParamInfo> params_;  // Parameter cache
    std::vector<void*> param_ptrs_;              // Parameter pointer array

    // Helper method
    void compile_ptx(const std::string& ptx_source);
    void prepare_parameters();
};

// Utility class for compiling and loading CUDA kernels
class CUDAKernelBuilder {
public:
    static std::shared_ptr<CUDAKernel> build_from_source(
        CUDADevice* device,
        const std::string& source,
        const std::string& kernel_name,
        const std::vector<std::string>& compile_options = {});

    static std::shared_ptr<CUDAKernel> build_from_ptx(
        CUDADevice* device,
        const std::string& ptx_source,
        const std::string& kernel_name);

private:
    static std::string compile_to_ptx(
        const std::string& source,
        const std::vector<std::string>& options);
};

} // namespace hal
} // namespace deeppowers 