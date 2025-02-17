#pragma once

#include "../hal/hal.hpp"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace deeppowers {

// Model types
enum class ModelType {
    DECODER_ONLY,    // Decoder-only models (e.g., GPT series)
    ENCODER_ONLY,    // Encoder-only models (e.g., BERT series)
    ENCODER_DECODER  // Encoder-decoder models (e.g., T5 series)
};

// Quantization data types
enum class QuantizationType {
    NONE,       // No quantization
    INT8,       // 8-bit integer quantization
    INT4,       // 4-bit integer quantization
    FP16        // 16-bit floating point
};

// Quantization method
enum class QuantizationMethod {
    NONE,                   // No quantization
    POST_TRAINING,         // Post-training quantization
    QUANTIZATION_AWARE,    // Quantization-aware training
    DYNAMIC               // Dynamic quantization at runtime
};

// Quantization configuration
struct QuantizationConfig {
    QuantizationType type = QuantizationType::NONE;
    QuantizationMethod method = QuantizationMethod::NONE;
    bool per_channel = false;           // Whether to use per-channel quantization
    bool symmetric = true;              // Whether to use symmetric quantization
    float calibration_ratio = 0.01f;    // Ratio of data to use for calibration
    std::vector<std::string> excluded_ops;  // Operations to exclude from quantization
};

// Model configuration
struct ModelConfig {
    ModelType type = ModelType::DECODER_ONLY;  // Model type
    size_t hidden_size = 768;                  // Hidden layer size
    size_t num_layers = 12;                    // Number of layers
    size_t num_attention_heads = 12;           // Number of attention heads
    size_t vocab_size = 50257;                 // Vocabulary size
    size_t max_position_embeddings = 2048;     // Maximum position embeddings
    float attention_dropout = 0.1f;            // Attention dropout rate
    float hidden_dropout = 0.1f;               // Hidden layer dropout rate
    
    // Quantization configuration
    QuantizationConfig quant_config;
    bool use_fp16 = false;                     // Whether to use FP16
    bool use_int8 = false;                     // Whether to use INT8
    std::string quantization_method = "";      // Quantization method
};

// Model weights
class ModelWeights {
public:
    virtual ~ModelWeights() = default;
    
    // Loading and saving
    virtual void load_from_file(const std::string& path) = 0;
    virtual void save_to_file(const std::string& path) const = 0;
    
    // Weight access
    virtual hal::Tensor* get_weight(const std::string& name) = 0;
    virtual const hal::Tensor* get_weight(const std::string& name) const = 0;
    
    // Memory management
    virtual size_t total_size_in_bytes() const = 0;
    virtual void to_device(hal::Device* device) = 0;
    virtual void to_host() = 0;
};

// Model class
class Model {
public:
    Model(const ModelConfig& config, hal::Device* device);
    virtual ~Model() = default;

    // Disable copying
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Model operations
    virtual void load_weights(const std::string& path) = 0;
    virtual void save_weights(const std::string& path) const = 0;
    virtual void to_device(hal::Device* device) = 0;
    virtual void to_host() = 0;

    // Configuration access
    const ModelConfig& config() const { return config_; }
    ModelConfig& mutable_config() { return config_; }
    
    // Device access
    hal::Device* device() const { return device_; }
    void set_device(hal::Device* device) { device_ = device; }

protected:
    ModelConfig config_;
    hal::Device* device_;
    std::unique_ptr<ModelWeights> weights_;
};

// Model factory
class ModelFactory {
public:
    static std::unique_ptr<Model> create_model(
        const std::string& model_name,
        const ModelConfig& config,
        hal::Device* device);
        
    static void register_model_creator(
        const std::string& model_name,
        std::function<std::unique_ptr<Model>(
            const ModelConfig&, hal::Device*)> creator);

private:
    static std::unordered_map<std::string,
        std::function<std::unique_ptr<Model>(
            const ModelConfig&, hal::Device*)>> model_creators_;
};

} // namespace deeppowers 