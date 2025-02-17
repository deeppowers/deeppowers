#include "cuda_kernel.hpp"
#include <nvrtc.h>
#include <sstream>
#include <cstring>

namespace deeppowers {
namespace hal {

// NVRTC error checking helper function
static void check_nvrtc_error(nvrtcResult result, const char* file, int line) {
    if (result != NVRTC_SUCCESS) {
        std::ostringstream oss;
        oss << "NVRTC error at " << file << ":" << line << ": "
            << nvrtcGetErrorString(result);
        throw std::runtime_error(oss.str());
    }
}

#define NVRTC_CHECK(err) check_nvrtc_error(err, __FILE__, __LINE__)

// CUDA driver API error checking helper function
static void check_cuda_driver_error(CUresult result, const char* file, int line) {
    if (result != CUDA_SUCCESS) {
        const char* error_string;
        cuGetErrorString(result, &error_string);
        std::ostringstream oss;
        oss << "CUDA driver error at " << file << ":" << line << ": "
            << error_string;
        throw std::runtime_error(oss.str());
    }
}

#define CUDA_DRIVER_CHECK(err) check_cuda_driver_error(err, __FILE__, __LINE__)

CUDAKernel::CUDAKernel(CUDADevice* device,
                       const std::string& name,
                       const std::string& ptx_source,
                       const std::string& entry_point)
    : device_(device)
    , name_(name)
    , module_(nullptr)
    , function_(nullptr) {
    
    if (!device_) {
        throw std::runtime_error("Device cannot be null");
    }

    compile_ptx(ptx_source);
    
    // Get kernel function
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&function_, module_, entry_point.c_str()));
}

CUDAKernel::~CUDAKernel() {
    if (module_) {
        cuModuleUnload(module_);
        module_ = nullptr;
    }
}

void CUDAKernel::compile_ptx(const std::string& ptx_source) {
    // Initialize CUDA driver API
    static bool initialized = false;
    if (!initialized) {
        CUDA_DRIVER_CHECK(cuInit(0));
        initialized = true;
    }

    // Load PTX
    CUDA_DRIVER_CHECK(cuModuleLoadData(&module_, ptx_source.c_str()));
}

void CUDAKernel::set_arg(int index, size_t size, const void* value) {
    if (!value) {
        throw std::runtime_error("Parameter value cannot be null");
    }

    // Store parameter
    auto& param = params_[index];
    param.size = size;
    param.data.resize(size);
    std::memcpy(param.data.data(), value, size);
}

void CUDAKernel::prepare_parameters() {
    // Clear old parameter pointers
    param_ptrs_.clear();
    
    // Prepare parameters in index order
    for (int i = 0; i < params_.size(); ++i) {
        auto it = params_.find(i);
        if (it == params_.end()) {
            throw std::runtime_error("Missing kernel parameter at index " + std::to_string(i));
        }
        param_ptrs_.push_back(it->second.data.data());
    }
}

void CUDAKernel::launch(const LaunchConfig& config, Stream* stream) {
    if (config.grid_dim.size() != 3 || config.block_dim.size() != 3) {
        throw std::runtime_error("Grid and block dimensions must be 3D");
    }

    // Prepare parameters
    prepare_parameters();

    // Get CUDA stream
    CUstream cuda_stream = nullptr;
    if (stream) {
        auto cuda_stream_ptr = dynamic_cast<CUDAStream*>(stream);
        if (!cuda_stream_ptr) {
            throw std::runtime_error("Invalid stream type");
        }
        cuda_stream = cuda_stream_ptr->get_cuda_stream();
    }

    // Launch kernel
    CUDA_DRIVER_CHECK(cuLaunchKernel(function_,
                                    config.grid_dim[0], config.grid_dim[1], config.grid_dim[2],
                                    config.block_dim[0], config.block_dim[1], config.block_dim[2],
                                    config.shared_memory_bytes,
                                    cuda_stream,
                                    param_ptrs_.data(),
                                    nullptr));
}

std::string CUDAKernelBuilder::compile_to_ptx(
    const std::string& source,
    const std::vector<std::string>& options) {
    
    // Create NVRTC program
    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog,
                                  source.c_str(),
                                  "kernel.cu",
                                  0, nullptr, nullptr));

    // Prepare compile options
    std::vector<const char*> opts;
    for (const auto& opt : options) {
        opts.push_back(opt.c_str());
    }

    // Compile
    nvrtcResult compile_result = nvrtcCompileProgram(prog, opts.size(), opts.data());
    
    // Get compile log
    size_t log_size;
    NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &log_size));
    std::string log(log_size, '\0');
    NVRTC_CHECK(nvrtcGetProgramLog(prog, &log[0]));
    
    if (compile_result != NVRTC_SUCCESS) {
        nvrtcDestroyProgram(&prog);
        throw std::runtime_error("CUDA kernel compilation failed:\n" + log);
    }

    // Get PTX
    size_t ptx_size;
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
    std::string ptx(ptx_size, '\0');
    NVRTC_CHECK(nvrtcGetPTX(prog, &ptx[0]));

    // Clean up
    NVRTC_CHECK(nvrtcDestroyProgram(&prog));

    return ptx;
}

std::shared_ptr<CUDAKernel> CUDAKernelBuilder::build_from_source(
    CUDADevice* device,
    const std::string& source,
    const std::string& kernel_name,
    const std::vector<std::string>& compile_options) {
    
    // Compile source code to PTX
    std::string ptx = compile_to_ptx(source, compile_options);
    
    // Create kernel
    return build_from_ptx(device, ptx, kernel_name);
}

std::shared_ptr<CUDAKernel> CUDAKernelBuilder::build_from_ptx(
    CUDADevice* device,
    const std::string& ptx_source,
    const std::string& kernel_name) {
    
    return std::make_shared<CUDAKernel>(device, kernel_name, ptx_source, kernel_name);
}

} // namespace hal
} // namespace deeppowers 