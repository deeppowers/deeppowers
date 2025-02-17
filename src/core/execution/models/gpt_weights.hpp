#pragma once

#include "../model.hpp"
#include <unordered_map>
#include <string>

namespace deeppowers {

// GPT model weights class
class GPTWeights : public ModelWeights {
public:
    explicit GPTWeights(hal::Device* device);
    ~GPTWeights() override;

    // ModelWeights interface implementation
    void load_from_file(const std::string& path) override;
    void save_to_file(const std::string& path) const override;
    hal::Tensor* get_weight(const std::string& name) override;
    const hal::Tensor* get_weight(const std::string& name) const override;
    size_t total_size_in_bytes() const override;
    void to_device(hal::Device* device) override;
    void to_host() override;

    // Precision conversion methods
    void convert_to_fp16();
    void convert_to_fp32();

private:
    // Weight storage
    struct LayerWeights {
        // Attention layer weights
        std::unique_ptr<hal::Tensor> q_weight;     // Query weight
        std::unique_ptr<hal::Tensor> k_weight;     // Key weight
        std::unique_ptr<hal::Tensor> v_weight;     // Value weight
        std::unique_ptr<hal::Tensor> o_weight;     // Output weight
        std::unique_ptr<hal::Tensor> q_bias;       // Query bias
        std::unique_ptr<hal::Tensor> k_bias;       // Key bias
        std::unique_ptr<hal::Tensor> v_bias;       // Value bias
        std::unique_ptr<hal::Tensor> o_bias;       // Output bias
        
        // FFN layer weights
        std::unique_ptr<hal::Tensor> ffn_inter_weight;  // FFN intermediate weight
        std::unique_ptr<hal::Tensor> ffn_inter_bias;    // FFN intermediate bias
        std::unique_ptr<hal::Tensor> ffn_out_weight;    // FFN output weight
        std::unique_ptr<hal::Tensor> ffn_out_bias;      // FFN output bias
        
        // Layer normalization weights
        std::unique_ptr<hal::Tensor> attn_ln_weight;    // Pre-attention LN weight
        std::unique_ptr<hal::Tensor> attn_ln_bias;      // Pre-attention LN bias
        std::unique_ptr<hal::Tensor> ffn_ln_weight;     // Pre-FFN LN weight
        std::unique_ptr<hal::Tensor> ffn_ln_bias;       // Pre-FFN LN bias
    };

    // Embedding layer weights
    std::unique_ptr<hal::Tensor> token_embedding;      // Token embedding
    std::unique_ptr<hal::Tensor> position_embedding;   // Position encoding
    
    // Output layer weights
    std::unique_ptr<hal::Tensor> final_ln_weight;     // Final LN weight
    std::unique_ptr<hal::Tensor> final_ln_bias;       // Final LN bias
    std::unique_ptr<hal::Tensor> lm_head_weight;      // Language model head weight
    std::unique_ptr<hal::Tensor> lm_head_bias;        // Language model head bias
    
    // Transformer layer weights
    std::vector<LayerWeights> layer_weights;          // Weights for each layer
    
    // Device and memory management
    hal::Device* device_;
    std::unordered_map<std::string, hal::Tensor*> weight_map_;  // Weight name to pointer mapping
    
    // Helper methods
    void init_weights(const ModelConfig& config);
    void build_weight_map();
    void clear_weights();
};

} // namespace deeppowers 