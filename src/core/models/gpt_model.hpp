#pragma once

#include "../hal/hal.hpp"
#include "../model/weight_manager.hpp"
#include "../inference/inference_engine.hpp"
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace deeppowers {

/**
 * @brief Configuration for GPT model
 */
struct GPTConfig {
    size_t vocab_size = 50257;              // Vocabulary size
    size_t hidden_size = 768;               // Hidden layer size
    size_t num_layers = 12;                 // Number of transformer layers
    size_t num_heads = 12;                  // Number of attention heads
    size_t max_position_embeddings = 1024;  // Maximum position embeddings
    float attention_dropout = 0.1f;         // Attention dropout rate
    float hidden_dropout = 0.1f;            // Hidden layer dropout rate
    bool use_bias = true;                   // Whether to use bias terms
    bool use_rotary_embeddings = true;      // Whether to use rotary embeddings
    bool use_flash_attention = true;        // Whether to use flash attention
    bool use_alibi = false;                 // Whether to use ALiBi position encoding
};

/**
 * @brief GPT model implementation
 */
class GPTModel {
public:
    /**
     * @brief Constructor
     * @param device Device to run model on
     * @param config Model configuration
     */
    explicit GPTModel(hal::Device* device, const GPTConfig& config = GPTConfig());
    ~GPTModel();

    /**
     * @brief Load model weights
     * @param path Path to model weights
     */
    void load_weights(const std::string& path);

    /**
     * @brief Generate text from prompt
     * @param prompt Input prompt
     * @param max_tokens Maximum tokens to generate
     * @param temperature Sampling temperature
     * @param top_p Nucleus sampling threshold
     * @param top_k Top-k sampling threshold
     * @return Generated text
     */
    std::string generate(const std::string& prompt,
                        size_t max_tokens = 100,
                        float temperature = 0.7f,
                        float top_p = 0.9f,
                        float top_k = 0.0f);

    /**
     * @brief Generate text with streaming output
     * @param prompt Input prompt
     * @param callback Callback function for streaming output
     * @param max_tokens Maximum tokens to generate
     * @param temperature Sampling temperature
     * @param top_p Nucleus sampling threshold
     * @param top_k Top-k sampling threshold
     */
    void generate_stream(const std::string& prompt,
                        std::function<bool(const std::string&)> callback,
                        size_t max_tokens = 100,
                        float temperature = 0.7f,
                        float top_p = 0.9f,
                        float top_k = 0.0f);

    /**
     * @brief Move model to device
     * @param device Target device
     */
    void to_device(hal::Device* device);

private:
    // Internal helper methods
    void initialize();
    void create_attention_mask(const std::vector<int32_t>& input_ids);
    void create_position_ids(const std::vector<int32_t>& input_ids);
    void apply_rotary_embedding(hal::Tensor* query, hal::Tensor* key);
    void compute_attention_scores(hal::Tensor* query, hal::Tensor* key, hal::Tensor* mask);
    void apply_attention(hal::Tensor* scores, hal::Tensor* value, hal::Tensor* output);
    void compute_ffn(hal::Tensor* input, hal::Tensor* output);
    int32_t sample_token(const std::vector<float>& logits,
                        float temperature,
                        float top_p,
                        float top_k);

    // Forward pass methods
    void forward(const std::vector<int32_t>& input_ids,
                std::vector<float>& logits);
    void forward_layer(size_t layer_idx,
                      hal::Tensor* hidden_states,
                      hal::Tensor* attention_mask);

    // KV cache management
    void init_kv_cache(size_t batch_size, size_t max_length);
    void update_kv_cache(size_t layer_idx, size_t position);
    void clear_kv_cache();

    // Member variables
    hal::Device* device_;
    GPTConfig config_;
    std::unique_ptr<WeightManager> weights_;
    std::unique_ptr<InferenceEngine> inference_engine_;
    bool initialized_;

    // Compute buffers
    struct ComputeBuffers {
        std::unique_ptr<hal::Tensor> attention_mask;
        std::unique_ptr<hal::Tensor> position_ids;
        std::unique_ptr<hal::Tensor> hidden_states;
        std::unique_ptr<hal::Tensor> query;
        std::unique_ptr<hal::Tensor> key;
        std::unique_ptr<hal::Tensor> value;
        std::unique_ptr<hal::Tensor> attention_output;
        std::unique_ptr<hal::Tensor> ffn_intermediate;
        std::unique_ptr<hal::Tensor> ffn_output;
        std::unique_ptr<hal::Tensor> logits;
    };
    ComputeBuffers buffers_;

    // KV cache
    struct KVCache {
        std::vector<std::unique_ptr<hal::Tensor>> key_cache;
        std::vector<std::unique_ptr<hal::Tensor>> value_cache;
        size_t current_length;
        size_t max_length;
    };
    KVCache kv_cache_;
}; 
} // namespace deeppowers