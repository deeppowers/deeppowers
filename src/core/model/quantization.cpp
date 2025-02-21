#include "quantization.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace deeppowers {

Quantizer::Quantizer(const QuantConfig& config)
    : config_(config) {
}

CalibrationData Quantizer::calibrate(const std::vector<Tensor>& inputs) {
    CalibrationData calib_data;
    
    // Compute min/max values across all inputs
    for (const auto& input : inputs) {
        const float* data = input.data<float>();
        size_t size = input.size();
        
        if (config_.per_channel) {
            // Per-channel calibration
            size_t channels = input.shape()[0];
            size_t elements_per_channel = size / channels;
            
            for (size_t c = 0; c < channels; c++) {
                float min_val = std::numeric_limits<float>::max();
                float max_val = std::numeric_limits<float>::lowest();
                
                for (size_t i = 0; i < elements_per_channel; i++) {
                    float val = data[c * elements_per_channel + i];
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                }
                
                calib_data.min_vals.push_back(min_val);
                calib_data.max_vals.push_back(max_val);
                
                // Compute scale and zero point
                float scale;
                int32_t zero_point;
                compute_scale_zp({min_val, max_val}, scale, zero_point);
                
                calib_data.scales.push_back(scale);
                calib_data.zero_points.push_back(zero_point);
            }
        } else {
            // Per-tensor calibration
            float min_val = std::numeric_limits<float>::max();
            float max_val = std::numeric_limits<float>::lowest();
            
            for (size_t i = 0; i < size; i++) {
                min_val = std::min(min_val, data[i]);
                max_val = std::max(max_val, data[i]);
            }
            
            calib_data.min_vals = {min_val};
            calib_data.max_vals = {max_val};
            
            // Compute scale and zero point
            float scale;
            int32_t zero_point;
            compute_scale_zp({min_val, max_val}, scale, zero_point);
            
            calib_data.scales = {scale};
            calib_data.zero_points = {zero_point};
        }
    }
    
    return calib_data;
}

Tensor Quantizer::quantize_weights(const Tensor& weights,
                                 const CalibrationData& calib_data) {
    return config_.per_channel ? 
           quantize_per_channel(weights, calib_data) :
           quantize_per_tensor(weights, calib_data);
}

Tensor Quantizer::quantize_activations(const Tensor& activations,
                                     const CalibrationData& calib_data) {
    // Activations are always quantized per-tensor
    return quantize_per_tensor(activations, calib_data);
}

Tensor Quantizer::dequantize(const Tensor& quantized,
                           const CalibrationData& calib_data) {
    const auto& shape = quantized.shape();
    Tensor output(shape, DataType::FLOAT32);
    
    if (config_.per_channel) {
        size_t channels = shape[0];
        size_t elements_per_channel = quantized.size() / channels;
        
        for (size_t c = 0; c < channels; c++) {
            float scale = calib_data.scales[c];
            int32_t zero_point = calib_data.zero_points[c];
            
            const int8_t* q_data = quantized.data<int8_t>() + c * elements_per_channel;
            float* out_data = output.data<float>() + c * elements_per_channel;
            
            for (size_t i = 0; i < elements_per_channel; i++) {
                out_data[i] = scale * (q_data[i] - zero_point);
            }
        }
    } else {
        float scale = calib_data.scales[0];
        int32_t zero_point = calib_data.zero_points[0];
        
        const int8_t* q_data = quantized.data<int8_t>();
        float* out_data = output.data<float>();
        
        for (size_t i = 0; i < quantized.size(); i++) {
            out_data[i] = scale * (q_data[i] - zero_point);
        }
    }
    
    return output;
}

void Quantizer::compute_scale_zp(const std::vector<float>& values,
                               float& scale,
                               int32_t& zero_point) {
    float min_val = *std::min_element(values.begin(), values.end());
    float max_val = *std::max_element(values.begin(), values.end());
    
    // Compute scale
    float range = std::max(std::abs(min_val), std::abs(max_val));
    int num_bits = (config_.type == QuantizationType::INT4) ? 4 : 8;
    int max_quant = (1 << (num_bits - 1)) - 1;
    
    scale = range / max_quant;
    
    // Compute zero point
    if (config_.symmetric) {
        zero_point = 0;
    } else {
        zero_point = static_cast<int32_t>(std::round(-min_val / scale));
        zero_point = std::min(std::max(zero_point, -max_quant), max_quant);
    }
}

Tensor Quantizer::quantize_per_channel(const Tensor& input,
                                     const CalibrationData& calib_data) {
    const auto& shape = input.shape();
    Tensor output(shape, 
                 config_.type == QuantizationType::INT4 ? 
                 DataType::INT4 : DataType::INT8);
    
    size_t channels = shape[0];
    size_t elements_per_channel = input.size() / channels;
    
    for (size_t c = 0; c < channels; c++) {
        float scale = calib_data.scales[c];
        int32_t zero_point = calib_data.zero_points[c];
        
        const float* in_data = input.data<float>() + c * elements_per_channel;
        int8_t* out_data = output.data<int8_t>() + c * elements_per_channel;
        
        for (size_t i = 0; i < elements_per_channel; i++) {
            float scaled = std::round(in_data[i] / scale + zero_point);
            out_data[i] = static_cast<int8_t>(
                std::min(std::max(scaled, -128.0f), 127.0f));
        }
    }
    
    return output;
}

Tensor Quantizer::quantize_per_tensor(const Tensor& input,
                                    const CalibrationData& calib_data) {
    const auto& shape = input.shape();
    Tensor output(shape, 
                 config_.type == QuantizationType::INT4 ? 
                 DataType::INT4 : DataType::INT8);
    
    float scale = calib_data.scales[0];
    int32_t zero_point = calib_data.zero_points[0];
    
    const float* in_data = input.data<float>();
    int8_t* out_data = output.data<int8_t>();
    
    for (size_t i = 0; i < input.size(); i++) {
        float scaled = std::round(in_data[i] / scale + zero_point);
        out_data[i] = static_cast<int8_t>(
            std::min(std::max(scaled, -128.0f), 127.0f));
    }
    
    return output;
}

std::shared_ptr<Model> MixedPrecision::convert_model(
    const std::shared_ptr<Model>& model,
    int precision_bits) {
    // TODO: Implement model conversion to mixed precision
    return model;
}

std::unordered_map<std::string, int> MixedPrecision::auto_mixed_precision(
    const std::shared_ptr<Model>& model,
    const std::vector<Tensor>& calibration_data) {
    std::unordered_map<std::string, int> precision_map;
    
    // TODO: Implement automatic precision selection
    return precision_map;
}

bool MixedPrecision::check_accuracy_loss(
    const Tensor& orig,
    const Tensor& quantized,
    float threshold) {
    // TODO: Implement accuracy loss checking
    return false;
}

int MixedPrecision::determine_optimal_precision(
    const Tensor& weights,
    const Tensor& activations) {
    // TODO: Implement optimal precision determination
    return 8;
}

} // namespace deeppowers 