#include "quantization_manager.hpp"
#include "quantization_kernels.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace deeppowers {

QuantizationManager::QuantizationManager(hal::Device* device)
    : device_(device) {
    
    if (!device_) {
        throw std::runtime_error("Device cannot be null");
    }
    
    // Create CUDA kernels
    int8_quant_kernel_ = device->create_kernel("quantize_tensor_int8_kernel");
    int4_quant_kernel_ = device->create_kernel("quantize_tensor_int4_kernel");
    dequant_kernel_ = device->create_kernel("dequantize_tensor_kernel");
    calibration_kernel_ = device->create_kernel("calibration_kernel");
}

std::unique_ptr<hal::Tensor> QuantizationManager::quantize_int8(
    const hal::Tensor* tensor,
    bool per_channel) {
    
    if (!tensor) return nullptr;
    
    // Create output tensor
    const auto& shape = tensor->shape();
    auto quantized = std::make_unique<hal::Tensor>(shape, hal::DataType::INT8, device_);
    
    // Calculate elements per channel
    int num_elements = 1;
    for (int64_t dim : shape) {
        num_elements *= dim;
    }
    int elements_per_channel = per_channel ? (num_elements / shape.back()) : num_elements;
    
    // Allocate quantization parameters
    int num_channels = per_channel ? shape.back() : 1;
    std::vector<float> scales(num_channels);
    std::vector<int8_t> zero_points(num_channels);
    
    // Launch quantization kernel
    launch_quantization_kernel(tensor, quantized.get(), per_channel);
    
    return quantized;
}

std::unique_ptr<hal::Tensor> QuantizationManager::quantize_int4(
    const hal::Tensor* tensor,
    bool per_channel) {
    
    if (!tensor) return nullptr;
    
    // Create output tensor (INT4 values are packed into INT8)
    const auto& shape = tensor->shape();
    std::vector<int64_t> packed_shape = shape;
    packed_shape.back() = (shape.back() + 1) / 2;  // Two INT4 values per INT8
    auto quantized = std::make_unique<hal::Tensor>(packed_shape, hal::DataType::INT8, device_);
    
    // Calculate elements per channel
    int num_elements = 1;
    for (int64_t dim : shape) {
        num_elements *= dim;
    }
    int elements_per_channel = per_channel ? (num_elements / shape.back()) : num_elements;
    
    // Allocate quantization parameters
    int num_channels = per_channel ? shape.back() : 1;
    std::vector<float> scales(num_channels);
    std::vector<int8_t> zero_points(num_channels);
    
    // Launch quantization kernel
    launch_quantization_kernel(tensor, quantized.get(), per_channel);
    
    return quantized;
}

std::unique_ptr<hal::Tensor> QuantizationManager::dequantize(
    const hal::Tensor* tensor) {
    
    if (!tensor) return nullptr;
    
    // Create output tensor
    const auto& shape = tensor->shape();
    auto dequantized = std::make_unique<hal::Tensor>(shape, hal::DataType::FLOAT32, device_);
    
    // Launch dequantization kernel
    launch_dequantization_kernel(tensor, dequantized.get());
    
    return dequantized;
}

void QuantizationManager::calibrate(
    const hal::Tensor* tensor,
    const std::string& tensor_name) {
    
    if (!tensor || quant_method_ != QuantizationMethod::POST_TRAINING) {
        return;
    }
    
    // Get or create quantization parameters
    auto& params = quant_params_[tensor_name];
    if (params.scales.empty()) {
        const auto& shape = tensor->shape();
        int num_channels = per_channel_ ? shape.back() : 1;
        
        params.scales.resize(num_channels);
        params.zero_points.resize(num_channels);
        params.min_vals.resize(num_channels, std::numeric_limits<float>::max());
        params.max_vals.resize(num_channels, std::numeric_limits<float>::lowest());
        params.running_means.resize(num_channels);
        params.running_vars.resize(num_channels);
    }
    
    // Launch calibration kernel
    launch_calibration_kernel(tensor, tensor_name);
    
    params.num_samples++;
}

void QuantizationManager::finalize_calibration() {
    for (auto& [tensor_name, params] : quant_params_) {
        update_quantization_params(tensor_name);
    }
    is_calibrated_ = true;
}

std::unique_ptr<hal::Tensor> QuantizationManager::maybe_quantize(
    const hal::Tensor* tensor,
    const std::string& tensor_name) {
    
    if (!tensor || quant_type_ == QuantizationType::NONE || 
        quant_method_ != QuantizationMethod::DYNAMIC) {
        return nullptr;
    }
    
    switch (quant_type_) {
        case QuantizationType::INT8:
            return quantize_int8(tensor, per_channel_);
        case QuantizationType::INT4:
            return quantize_int4(tensor, per_channel_);
        default:
            return nullptr;
    }
}

std::unique_ptr<hal::Tensor> QuantizationManager::maybe_dequantize(
    const hal::Tensor* tensor,
    const std::string& tensor_name) {
    
    if (!tensor || tensor->dtype() == hal::DataType::FLOAT32) {
        return nullptr;
    }
    
    return dequantize(tensor);
}

void QuantizationManager::launch_quantization_kernel(
    const hal::Tensor* input,
    hal::Tensor* output,
    bool per_channel) {
    
    // Calculate kernel parameters
    const auto& shape = input->shape();
    int num_elements = 1;
    for (int64_t dim : shape) {
        num_elements *= dim;
    }
    int elements_per_channel = per_channel ? (num_elements / shape.back()) : num_elements;
    
    // Configure kernel launch parameters
    dim3 grid((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    
    // Select and launch appropriate kernel
    if (quant_type_ == QuantizationType::INT8) {
        kernels::quantize_tensor_int8_kernel<<<grid, block>>>(
            static_cast<const float*>(input->data()),
            static_cast<int8_t*>(output->data()),
            nullptr,  // Scales will be computed by kernel
            nullptr,  // Zero points will be computed by kernel
            num_elements,
            elements_per_channel,
            per_channel);
    } else if (quant_type_ == QuantizationType::INT4) {
        kernels::quantize_tensor_int4_kernel<<<grid, block>>>(
            static_cast<const float*>(input->data()),
            static_cast<int8_t*>(output->data()),
            nullptr,  // Scales will be computed by kernel
            nullptr,  // Zero points will be computed by kernel
            num_elements,
            elements_per_channel,
            per_channel);
    }
}

void QuantizationManager::launch_dequantization_kernel(
    const hal::Tensor* input,
    hal::Tensor* output) {
    
    // Calculate kernel parameters
    const auto& shape = input->shape();
    int num_elements = 1;
    for (int64_t dim : shape) {
        num_elements *= dim;
    }
    
    const auto& scales = input->scales();
    int elements_per_channel = num_elements / scales.size();
    bool per_channel = scales.size() > 1;
    bool is_int4 = input->dtype() == hal::DataType::INT4;
    
    // Configure kernel launch parameters
    dim3 grid((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    
    // Launch dequantization kernel
    kernels::dequantize_tensor_kernel<<<grid, block>>>(
        static_cast<const int8_t*>(input->data()),
        static_cast<float*>(output->data()),
        scales.data(),
        input->zero_points().data(),
        num_elements,
        elements_per_channel,
        per_channel,
        is_int4);
}

void QuantizationManager::launch_calibration_kernel(
    const hal::Tensor* input,
    const std::string& tensor_name) {
    
    auto& params = quant_params_[tensor_name];
    
    // Calculate kernel parameters
    const auto& shape = input->shape();
    int num_elements = 1;
    for (int64_t dim : shape) {
        num_elements *= dim;
    }
    int elements_per_channel = per_channel_ ? (num_elements / shape.back()) : num_elements;
    
    // Configure kernel launch parameters
    dim3 grid((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    
    // Launch calibration kernel
    kernels::calibration_kernel<<<grid, block>>>(
        static_cast<const float*>(input->data()),
        params.min_vals.data(),
        params.max_vals.data(),
        params.running_means.data(),
        params.running_vars.data(),
        num_elements,
        elements_per_channel,
        per_channel_);
}

void QuantizationManager::update_quantization_params(
    const std::string& tensor_name) {
    
    auto& params = quant_params_[tensor_name];
    
    // Update quantization parameters based on collected statistics
    for (size_t i = 0; i < params.scales.size(); ++i) {
        float min_val = params.min_vals[i];
        float max_val = params.max_vals[i];
        
        if (symmetric_) {
            // Symmetric quantization
            float abs_max = std::max(std::abs(min_val), std::abs(max_val));
            params.scales[i] = abs_max / 127.0f;
            params.zero_points[i] = 0;
        } else {
            // Asymmetric quantization
            params.scales[i] = (max_val - min_val) / 255.0f;
            params.zero_points[i] = static_cast<int8_t>(-min_val / params.scales[i]);
        }
        
        // Update running statistics
        params.running_means[i] /= params.num_samples;
        params.running_vars[i] = params.running_vars[i] / params.num_samples - 
                                params.running_means[i] * params.running_means[i];
    }
}

} // namespace deeppowers