#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include "tensor.hpp"

namespace deeppowers {

/**
 * Supported quantization types
 */
enum class QuantizationType {
    INT8,       // 8-bit integer quantization
    INT4,       // 4-bit integer quantization
    MIXED_INT8, // Mixed INT8/FP16 quantization
    MIXED_INT4  // Mixed INT4/FP16 quantization
};

/**
 * Quantization configuration
 */
struct QuantConfig {
    QuantizationType type;
    float scale_threshold;      // Threshold for scale calculation
    bool per_channel;          // Whether to use per-channel quantization
    bool symmetric;            // Whether to use symmetric quantization
    std::vector<std::string> skip_layers;  // Layers to skip quantization
};

/**
 * Quantization calibration data
 */
struct CalibrationData {
    std::vector<float> scales;     // Quantization scales
    std::vector<int32_t> zero_points;  // Zero points for asymmetric quantization
    std::vector<float> min_vals;   // Minimum values per channel
    std::vector<float> max_vals;   // Maximum values per channel
};

/**
 * Model quantization class
 */
class Quantizer {
public:
    /**
     * Create quantizer with configuration
     * @param config Quantization configuration
     */
    explicit Quantizer(const QuantConfig& config);
    
    /**
     * Calibrate quantization parameters using sample inputs
     * @param inputs Sample input tensors
     * @return Calibration data
     */
    CalibrationData calibrate(const std::vector<Tensor>& inputs);
    
    /**
     * Quantize weights using calibration data
     * @param weights Input weights tensor
     * @param calib_data Calibration data
     * @return Quantized weights tensor
     */
    Tensor quantize_weights(const Tensor& weights,
                          const CalibrationData& calib_data);
    
    /**
     * Quantize activations at runtime
     * @param activations Input activations tensor
     * @param calib_data Calibration data
     * @return Quantized activations tensor
     */
    Tensor quantize_activations(const Tensor& activations,
                              const CalibrationData& calib_data);
    
    /**
     * Dequantize tensor back to floating point
     * @param quantized Quantized tensor
     * @param calib_data Calibration data
     * @return Dequantized floating point tensor
     */
    Tensor dequantize(const Tensor& quantized,
                     const CalibrationData& calib_data);

private:
    QuantConfig config_;
    
    // Helper functions
    void compute_scale_zp(const std::vector<float>& values,
                         float& scale,
                         int32_t& zero_point);
    
    Tensor quantize_per_channel(const Tensor& input,
                              const CalibrationData& calib_data);
    
    Tensor quantize_per_tensor(const Tensor& input,
                             const CalibrationData& calib_data);
};

/**
 * Mixed precision computation support
 */
class MixedPrecision {
public:
    /**
     * Convert model to mixed precision
     * @param model Input model
     * @param precision_bits Bits for reduced precision (4 or 8)
     * @return Mixed precision model
     */
    static std::shared_ptr<Model> convert_model(
        const std::shared_ptr<Model>& model,
        int precision_bits);
    
    /**
     * Automatically determine optimal precision for each layer
     * @param model Input model
     * @param calibration_data Sample inputs for calibration
     * @return Layer precision mapping
     */
    static std::unordered_map<std::string, int> auto_mixed_precision(
        const std::shared_ptr<Model>& model,
        const std::vector<Tensor>& calibration_data);

private:
    // Helper functions for precision selection
    static bool check_accuracy_loss(const Tensor& orig,
                                  const Tensor& quantized,
                                  float threshold);
    
    static int determine_optimal_precision(const Tensor& weights,
                                        const Tensor& activations);
};

} // namespace deeppowers 