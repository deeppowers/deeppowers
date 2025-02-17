#pragma once

#include "../hal/hal.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace deeppowers {

// Quantization manager class
class QuantizationManager {
public:
    QuantizationManager(hal::Device* device);
    ~QuantizationManager() = default;
    
    // Quantization methods
    std::unique_ptr<hal::Tensor> quantize_int8(
        const hal::Tensor* tensor,
        bool per_channel = false);
        
    std::unique_ptr<hal::Tensor> quantize_int4(
        const hal::Tensor* tensor,
        bool per_channel = false);
        
    // Dequantization methods
    std::unique_ptr<hal::Tensor> dequantize(
        const hal::Tensor* tensor);
        
    // Calibration methods
    void calibrate(
        const hal::Tensor* tensor,
        const std::string& tensor_name);
        
    void finalize_calibration();
    
    // Runtime quantization
    std::unique_ptr<hal::Tensor> maybe_quantize(
        const hal::Tensor* tensor,
        const std::string& tensor_name);
        
    std::unique_ptr<hal::Tensor> maybe_dequantize(
        const hal::Tensor* tensor,
        const std::string& tensor_name);
        
    // Configuration methods
    void set_quantization_type(QuantizationType type) { quant_type_ = type; }
    void set_quantization_method(QuantizationMethod method) { quant_method_ = method; }
    void set_per_channel(bool per_channel) { per_channel_ = per_channel; }
    void set_symmetric(bool symmetric) { symmetric_ = symmetric; }
    
    // Status query
    bool is_calibrated() const { return is_calibrated_; }
    
private:
    // Internal helper methods
    void launch_quantization_kernel(
        const hal::Tensor* input,
        hal::Tensor* output,
        bool per_channel);
        
    void launch_dequantization_kernel(
        const hal::Tensor* input,
        hal::Tensor* output);
        
    void launch_calibration_kernel(
        const hal::Tensor* input,
        const std::string& tensor_name);
        
    void update_quantization_params(
        const std::string& tensor_name);
        
    // Quantization parameters structure
    struct QuantParams {
        std::vector<float> scales;           // Scaling factors
        std::vector<int8_t> zero_points;     // Zero points
        std::vector<float> min_vals;         // Minimum values
        std::vector<float> max_vals;         // Maximum values
        std::vector<float> running_means;    // Running means
        std::vector<float> running_vars;     // Running variances
        size_t num_samples = 0;              // Number of calibration samples
    };
    
    // Member variables
    hal::Device* device_;                    // Device for computation
    QuantizationType quant_type_ = QuantizationType::NONE;  // Quantization type
    QuantizationMethod quant_method_ = QuantizationMethod::NONE;  // Quantization method
    bool per_channel_ = false;               // Per-channel quantization flag
    bool symmetric_ = true;                  // Symmetric quantization flag
    bool is_calibrated_ = false;             // Calibration status
    
    // Quantization parameters cache
    std::unordered_map<std::string, QuantParams> quant_params_;
    
    // CUDA kernel cache
    hal::Kernel* int8_quant_kernel_ = nullptr;    // INT8 quantization kernel
    hal::Kernel* int4_quant_kernel_ = nullptr;    // INT4 quantization kernel
    hal::Kernel* dequant_kernel_ = nullptr;       // Dequantization kernel
    hal::Kernel* calibration_kernel_ = nullptr;   // Calibration kernel
};

} // namespace deeppowers 