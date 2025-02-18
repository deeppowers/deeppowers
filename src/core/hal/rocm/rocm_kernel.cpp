#include "rocm_kernel.hpp"
#include "rocm_device.hpp"
#include "rocm_stream.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <hip/hiprtc.h>

namespace deeppowers {
namespace hal {

ROCmKernel::ROCmKernel(ROCmDevice* device, const std::string& name, const std::string& code)
    : device_(device)
    , module_(nullptr)
    , function_(nullptr)
    , name_(name) {
    compile(code);
}

ROCmKernel::ROCmKernel(ROCmDevice* device, const std::string& path)
    : device_(device)
    , module_(nullptr)
    , function_(nullptr) {
    load_binary(path);
}

ROCmKernel::~ROCmKernel() {
    if (module_) {
        ROCM_CHECK(hipModuleUnload(module_));
    }
}

void ROCmKernel::compile(const std::string& code) {
    // Create program
    hiprtcProgram prog;
    ROCM_CHECK(hiprtcCreateProgram(&prog, code.c_str(), name_.c_str(), 0, nullptr, nullptr));
    
    // Set compilation options
    std::vector<const char*> options;
    options.push_back("--gpu-architecture=gfx900");  // Default to Vega architecture
    
    // Compile
    hiprtcResult result = hiprtcCompileProgram(prog, options.size(), options.data());
    
    // Get compilation log
    size_t log_size;
    ROCM_CHECK(hiprtcGetProgramLogSize(prog, &log_size));
    if (log_size > 1) {
        std::vector<char> log(log_size);
        ROCM_CHECK(hiprtcGetProgramLog(prog, log.data()));
        if (result != HIPRTC_SUCCESS) {
            throw std::runtime_error("Kernel compilation failed: " + std::string(log.data()));
        }
    }
    
    // Get PTX code
    size_t code_size;
    ROCM_CHECK(hiprtcGetCodeSize(prog, &code_size));
    std::vector<char> ptx(code_size);
    ROCM_CHECK(hiprtcGetCode(prog, ptx.data()));
    
    // Load module and get function
    ROCM_CHECK(hipModuleLoadData(&module_, ptx.data()));
    ROCM_CHECK(hipModuleGetFunction(&function_, module_, name_.c_str()));
    
    // Cleanup
    ROCM_CHECK(hiprtcDestroyProgram(&prog));
}

void ROCmKernel::load_binary(const std::string& path) {
    // Read binary file
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open kernel binary file: " + path);
    }
    
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> binary(size);
    file.read(binary.data(), size);
    
    // Load module
    ROCM_CHECK(hipModuleLoadData(&module_, binary.data()));
    
    // Get function
    size_t pos = path.find_last_of("/\\");
    name_ = path.substr(pos + 1);
    pos = name_.find_last_of(".");
    if (pos != std::string::npos) {
        name_ = name_.substr(0, pos);
    }
    ROCM_CHECK(hipModuleGetFunction(&function_, module_, name_.c_str()));
}

void ROCmKernel::launch(const std::vector<size_t>& grid_dim,
                       const std::vector<size_t>& block_dim,
                       const std::vector<void*>& args,
                       size_t shared_memory,
                       Stream* stream) {
    if (grid_dim.size() != 3 || block_dim.size() != 3) {
        throw std::runtime_error("Grid and block dimensions must be 3D");
    }
    
    // Prepare launch configuration
    dim3 grid(grid_dim[0], grid_dim[1], grid_dim[2]);
    dim3 block(block_dim[0], block_dim[1], block_dim[2]);
    
    // Get stream
    hipStream_t rocm_stream = nullptr;
    if (stream) {
        rocm_stream = static_cast<ROCmStream*>(stream)->rocm_stream();
    }
    
    // Launch kernel
    void** kernel_args = const_cast<void**>(args.data());
    ROCM_CHECK(hipModuleLaunchKernel(function_,
                                    grid.x, grid.y, grid.z,
                                    block.x, block.y, block.z,
                                    shared_memory, rocm_stream,
                                    kernel_args, nullptr));
}

void ROCmKernel::set_constant(const std::string& name, const void* data, size_t size) {
    hipDeviceptr_t ptr;
    size_t bytes;
    ROCM_CHECK(hipModuleGetGlobal(&ptr, &bytes, module_, name.c_str()));
    
    if (bytes != size) {
        throw std::runtime_error("Constant buffer size mismatch");
    }
    
    ROCM_CHECK(hipMemcpyHtoD(ptr, data, size));
}

}} // namespace deeppowers::hal 