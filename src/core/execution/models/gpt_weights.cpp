#include "gpt_weights.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

namespace deeppowers {

GPTWeights::GPTWeights(hal::Device* device)
    : device_(device) {
    if (!device_) {
        throw std::runtime_error("Device cannot be null");
    }
}

GPTWeights::~GPTWeights() {
    clear_weights();
}

void GPTWeights::load_from_file(const std::string& path) {
    // Load weight file
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open weight file: " + path);
    }

    try {
        // Read configuration information
        nlohmann::json config_json;
        file >> config_json;
        
        // Parse configuration
        ModelConfig config;
        config.hidden_size = config_json["hidden_size"];
        config.num_layers = config_json["num_layers"];
        config.num_attention_heads = config_json["num_attention_heads"];
        config.vocab_size = config_json["vocab_size"];
        
        // Initialize weights
        init_weights(config);
        
        // Read weight data
        for (auto& [name, tensor] : weight_map_) {
            // Read tensor shape and data
            std::vector<int64_t> shape;
            size_t shape_size;
            file.read(reinterpret_cast<char*>(&shape_size), sizeof(size_t));
            shape.resize(shape_size);
            file.read(reinterpret_cast<char*>(shape.data()), 
                     shape_size * sizeof(int64_t));
            
            // Allocate host memory and read data
            size_t data_size = tensor->size_in_bytes();
            std::vector<uint8_t> host_data(data_size);
            file.read(reinterpret_cast<char*>(host_data.data()), data_size);
            
            // Copy data to device
            tensor->copy_from_host(host_data.data());
        }
    } catch (const std::exception& e) {
        clear_weights();
        throw std::runtime_error("Failed to load weights: " + std::string(e.what()));
    }
}

void GPTWeights::save_to_file(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to create weight file: " + path);
    }

    try {
        // Save configuration information
        nlohmann::json config_json;
        // TODO: Extract configuration information from weights
        file << config_json.dump(4);
        
        // Save weight data
        for (const auto& [name, tensor] : weight_map_) {
            // Save tensor shape
            const auto& shape = tensor->shape();
            size_t shape_size = shape.size();
            file.write(reinterpret_cast<const char*>(&shape_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(shape.data()),
                      shape_size * sizeof(int64_t));
            
            // Allocate host memory and copy data
            size_t data_size = tensor->size_in_bytes();
            std::vector<uint8_t> host_data(data_size);
            tensor->copy_to_host(host_data.data());
            
            // Write data
            file.write(reinterpret_cast<const char*>(host_data.data()), data_size);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to save weights: " + std::string(e.what()));
    }
}

hal::Tensor* GPTWeights::get_weight(const std::string& name) {
    auto it = weight_map_.find(name);
    return (it != weight_map_.end()) ? it->second : nullptr;
}

const hal::Tensor* GPTWeights::get_weight(const std::string& name) const {
    auto it = weight_map_.find(name);
    return (it != weight_map_.end()) ? it->second : nullptr;
}

size_t GPTWeights::total_size_in_bytes() const {
    size_t total_size = 0;
    for (const auto& [name, tensor] : weight_map_) {
        total_size += tensor->size_in_bytes();
    }
    return total_size;
}

void GPTWeights::to_device(hal::Device* device) {
    if (!device) {
        throw std::runtime_error("Device cannot be null");
    }
    
    if (device == device_) {
        return;
    }
    
    // Move all weights to new device
    for (auto& [name, tensor] : weight_map_) {
        // TODO: Implement tensor device transfer
    }
    
    device_ = device;
}

void GPTWeights::to_host() {
    if (!device_ || device_->type() == hal::DeviceType::CPU) {
        return;
    }
    
    // Create CPU device
    auto cpu_device = hal::Device::create(hal::DeviceType::CPU);
    
    // Transfer all weights to CPU
    for (auto& [name, tensor] : weight_map_) {
        if (tensor) {
            // Create new tensor on CPU
            std::vector<int64_t> shape = tensor->shape();
            auto new_tensor = std::make_unique<hal::Tensor>(
                shape, tensor->dtype(), cpu_device.get());
            
            // Copy data to CPU
            cpu_device->copy_tensor(tensor, new_tensor.get());
            
            // Update the tensor in the weight map
            auto* old_tensor = tensor;
            tensor = new_tensor.release();
            delete old_tensor;
        }
    }
    
    device_ = cpu_device.release();
}

void GPTWeights::init_weights(const ModelConfig& config) {
    clear_weights();
    
    const auto& gpt_config = static_cast<const GPTConfig&>(config);
    
    // Create embedding layer weights
    std::vector<int64_t> embedding_shape = {
        static_cast<int64_t>(gpt_config.vocab_size),
        static_cast<int64_t>(gpt_config.hidden_size)
    };
    
    std::vector<int64_t> position_shape = {
        static_cast<int64_t>(gpt_config.max_sequence_length),
        static_cast<int64_t>(gpt_config.hidden_size)
    };
    
    token_embedding = std::make_unique<hal::Tensor>(
        embedding_shape, hal::DataType::FLOAT32, device_);
    position_embedding = std::make_unique<hal::Tensor>(
        position_shape, hal::DataType::FLOAT32, device_);
    
    // Create weights for each layer
    layer_weights.resize(config.num_layers);
    for (auto& layer : layer_weights) {
        // Attention weights shapes
        std::vector<int64_t> qkv_shape = {
            static_cast<int64_t>(gpt_config.hidden_size),
            static_cast<int64_t>(gpt_config.hidden_size)
        };
        
        std::vector<int64_t> qkv_bias_shape = {
            static_cast<int64_t>(gpt_config.hidden_size)
        };
        
        // Create attention weights
        layer.q_weight = std::make_unique<hal::Tensor>(
            qkv_shape, hal::DataType::FLOAT32, device_);
        layer.k_weight = std::make_unique<hal::Tensor>(
            qkv_shape, hal::DataType::FLOAT32, device_);
        layer.v_weight = std::make_unique<hal::Tensor>(
            qkv_shape, hal::DataType::FLOAT32, device_);
        layer.o_weight = std::make_unique<hal::Tensor>(
            qkv_shape, hal::DataType::FLOAT32, device_);
            
        layer.q_bias = std::make_unique<hal::Tensor>(
            qkv_bias_shape, hal::DataType::FLOAT32, device_);
        layer.k_bias = std::make_unique<hal::Tensor>(
            qkv_bias_shape, hal::DataType::FLOAT32, device_);
        layer.v_bias = std::make_unique<hal::Tensor>(
            qkv_bias_shape, hal::DataType::FLOAT32, device_);
        layer.o_bias = std::make_unique<hal::Tensor>(
            qkv_bias_shape, hal::DataType::FLOAT32, device_);
            
        // FFN weights shapes
        std::vector<int64_t> ffn_inter_shape = {
            static_cast<int64_t>(gpt_config.hidden_size),
            static_cast<int64_t>(gpt_config.intermediate_size)
        };
        
        std::vector<int64_t> ffn_out_shape = {
            static_cast<int64_t>(gpt_config.intermediate_size),
            static_cast<int64_t>(gpt_config.hidden_size)
        };
        
        std::vector<int64_t> ffn_inter_bias_shape = {
            static_cast<int64_t>(gpt_config.intermediate_size)
        };
        
        std::vector<int64_t> ffn_out_bias_shape = {
            static_cast<int64_t>(gpt_config.hidden_size)
        };
        
        // Create FFN weights
        layer.ffn_inter_weight = std::make_unique<hal::Tensor>(
            ffn_inter_shape, hal::DataType::FLOAT32, device_);
        layer.ffn_inter_bias = std::make_unique<hal::Tensor>(
            ffn_inter_bias_shape, hal::DataType::FLOAT32, device_);
        layer.ffn_out_weight = std::make_unique<hal::Tensor>(
            ffn_out_shape, hal::DataType::FLOAT32, device_);
        layer.ffn_out_bias = std::make_unique<hal::Tensor>(
            ffn_out_bias_shape, hal::DataType::FLOAT32, device_);
            
        // Layer norm weights shapes
        std::vector<int64_t> ln_shape = {
            static_cast<int64_t>(gpt_config.hidden_size)
        };
        
        // Create layer norm weights
        layer.attn_ln_weight = std::make_unique<hal::Tensor>(
            ln_shape, hal::DataType::FLOAT32, device_);
        layer.attn_ln_bias = std::make_unique<hal::Tensor>(
            ln_shape, hal::DataType::FLOAT32, device_);
        layer.ffn_ln_weight = std::make_unique<hal::Tensor>(
            ln_shape, hal::DataType::FLOAT32, device_);
        layer.ffn_ln_bias = std::make_unique<hal::Tensor>(
            ln_shape, hal::DataType::FLOAT32, device_);
    }
    
    // Create output layer weights
    std::vector<int64_t> ln_shape = {
        static_cast<int64_t>(gpt_config.hidden_size)
    };
    
    std::vector<int64_t> lm_head_shape = {
        static_cast<int64_t>(gpt_config.hidden_size),
        static_cast<int64_t>(gpt_config.vocab_size)
    };
    
    std::vector<int64_t> lm_head_bias_shape = {
        static_cast<int64_t>(gpt_config.vocab_size)
    };
    
    final_ln_weight = std::make_unique<hal::Tensor>(
        ln_shape, hal::DataType::FLOAT32, device_);
    final_ln_bias = std::make_unique<hal::Tensor>(
        ln_shape, hal::DataType::FLOAT32, device_);
    lm_head_weight = std::make_unique<hal::Tensor>(
        lm_head_shape, hal::DataType::FLOAT32, device_);
    lm_head_bias = std::make_unique<hal::Tensor>(
        lm_head_bias_shape, hal::DataType::FLOAT32, device_);
    
    // Build weight map
    build_weight_map();
}

void GPTWeights::build_weight_map() {
    weight_map_.clear();
    
    // Add embedding layer weights
    weight_map_["token_embedding"] = token_embedding.get();
    weight_map_["position_embedding"] = position_embedding.get();
    
    // Add weights for each layer
    for (size_t i = 0; i < layer_weights.size(); ++i) {
        const auto& layer = layer_weights[i];
        std::string prefix = "layer_" + std::to_string(i) + "_";
        
        weight_map_[prefix + "q_weight"] = layer.q_weight.get();
        weight_map_[prefix + "k_weight"] = layer.k_weight.get();
        weight_map_[prefix + "v_weight"] = layer.v_weight.get();
        weight_map_[prefix + "o_weight"] = layer.o_weight.get();
        
        weight_map_[prefix + "ffn_inter_weight"] = layer.ffn_inter_weight.get();
        weight_map_[prefix + "ffn_out_weight"] = layer.ffn_out_weight.get();
        
        weight_map_[prefix + "attn_ln_weight"] = layer.attn_ln_weight.get();
        weight_map_[prefix + "ffn_ln_weight"] = layer.ffn_ln_weight.get();
    }
    
    // Add output layer weights
    weight_map_["final_ln_weight"] = final_ln_weight.get();
    weight_map_["lm_head_weight"] = lm_head_weight.get();
}

void GPTWeights::clear_weights() {
    // Clear all weights
    token_embedding.reset();
    position_embedding.reset();
    layer_weights.clear();
    final_ln_weight.reset();
    lm_head_weight.reset();
    weight_map_.clear();
}

void GPTWeights::convert_to_fp16() {
    // Helper function to convert a tensor to FP16
    auto convert_tensor = [this](std::unique_ptr<hal::Tensor>& tensor) {
        if (tensor && tensor->dtype() != hal::DataType::FLOAT16) {
            auto shape = tensor->shape();
            auto new_tensor = std::make_unique<hal::Tensor>(
                shape, hal::DataType::FLOAT16, device_);
            device_->convert_precision(tensor.get(), new_tensor.get());
            tensor = std::move(new_tensor);
        }
    };
    
    // Convert embedding weights
    convert_tensor(token_embedding);
    convert_tensor(position_embedding);
    
    // Convert layer weights
    for (auto& layer : layer_weights) {
        // Convert attention weights
        convert_tensor(layer.q_weight);
        convert_tensor(layer.k_weight);
        convert_tensor(layer.v_weight);
        convert_tensor(layer.o_weight);
        convert_tensor(layer.q_bias);
        convert_tensor(layer.k_bias);
        convert_tensor(layer.v_bias);
        convert_tensor(layer.o_bias);
        
        // Convert FFN weights
        convert_tensor(layer.ffn_inter_weight);
        convert_tensor(layer.ffn_inter_bias);
        convert_tensor(layer.ffn_out_weight);
        convert_tensor(layer.ffn_out_bias);
        
        // Convert layer norm weights
        convert_tensor(layer.attn_ln_weight);
        convert_tensor(layer.attn_ln_bias);
        convert_tensor(layer.ffn_ln_weight);
        convert_tensor(layer.ffn_ln_bias);
    }
    
    // Convert output layer weights
    convert_tensor(final_ln_weight);
    convert_tensor(final_ln_bias);
    convert_tensor(lm_head_weight);
    convert_tensor(lm_head_bias);
    
    // Rebuild weight map with converted tensors
    build_weight_map();
}

void GPTWeights::convert_to_fp32() {
    // Helper function to convert a tensor to FP32
    auto convert_tensor = [this](std::unique_ptr<hal::Tensor>& tensor) {
        if (tensor && tensor->dtype() != hal::DataType::FLOAT32) {
            auto shape = tensor->shape();
            auto new_tensor = std::make_unique<hal::Tensor>(
                shape, hal::DataType::FLOAT32, device_);
            device_->convert_precision(tensor.get(), new_tensor.get());
            tensor = std::move(new_tensor);
        }
    };
    
    // Convert embedding weights
    convert_tensor(token_embedding);
    convert_tensor(position_embedding);
    
    // Convert layer weights
    for (auto& layer : layer_weights) {
        // Convert attention weights
        convert_tensor(layer.q_weight);
        convert_tensor(layer.k_weight);
        convert_tensor(layer.v_weight);
        convert_tensor(layer.o_weight);
        convert_tensor(layer.q_bias);
        convert_tensor(layer.k_bias);
        convert_tensor(layer.v_bias);
        convert_tensor(layer.o_bias);
        
        // Convert FFN weights
        convert_tensor(layer.ffn_inter_weight);
        convert_tensor(layer.ffn_inter_bias);
        convert_tensor(layer.ffn_out_weight);
        convert_tensor(layer.ffn_out_bias);
        
        // Convert layer norm weights
        convert_tensor(layer.attn_ln_weight);
        convert_tensor(layer.attn_ln_bias);
        convert_tensor(layer.ffn_ln_weight);
        convert_tensor(layer.ffn_ln_bias);
    }
    
    // Convert output layer weights
    convert_tensor(final_ln_weight);
    convert_tensor(final_ln_bias);
    convert_tensor(lm_head_weight);
    convert_tensor(lm_head_bias);
    
    // Rebuild weight map with converted tensors
    build_weight_map();
}

} // namespace deeppowers