#pragma once

#include "../kernel.hpp"
#include <hip/hip_runtime.h>
#include <string>
#include <vector>

namespace deeppowers {
namespace hal {

class ROCmDevice;
class ROCmStream;

/**
 * ROCm kernel implementation for GPU compute kernels
 */
class ROCmKernel : public Kernel {
public:
    /**
     * Create a new ROCm kernel from source code
     * @param device ROCm device pointer
     * @param name Kernel function name
     * @param code HIP kernel source code
     */
    ROCmKernel(ROCmDevice* device, const std::string& name, const std::string& code);
    
    /**
     * Load a ROCm kernel from a pre-compiled binary file
     * @param device ROCm device pointer
     * @param path Path to the binary file
     */
    ROCmKernel(ROCmDevice* device, const std::string& path);
    
    ~ROCmKernel() override;

    // Kernel interface implementations
    void launch(const std::vector<size_t>& grid_dim,
               const std::vector<size_t>& block_dim,
               const std::vector<void*>& args,
               size_t shared_memory = 0,
               Stream* stream = nullptr) override;
               
    void set_constant(const std::string& name, const void* data, size_t size) override;

private:
    ROCmDevice* device_;
    hipModule_t module_;
    hipFunction_t function_;
    std::string name_;
    
    // Compile kernel source code
    void compile(const std::string& code);
    
    // Load pre-compiled kernel
    void load_binary(const std::string& path);
};

}} // namespace deeppowers::hal 