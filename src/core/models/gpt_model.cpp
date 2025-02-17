#include "gpt_model.hpp"
#include <random>
#include <algorithm>
#include <cmath>

namespace deeppowers {

GPTModel::GPTModel(hal::Device* device, const GPTConfig& config)
    : device_(device)
    , config_(config)
    , initialized_(false) {
    
    if (!device_) {
        throw std::runtime_error("Device cannot be null");
    }

    // Create weight manager
    weights_ = std::make_unique<WeightManager>(device_);

    // Create inference engine
    InferenceConfig engine_config;
    engine_config.enable_tensor_cores = true;
    engine_config.enable_kernel_fusion = true;
    inference_engine_ = std::make_unique<InferenceEngine>(device_, engine_config);
}

GPTModel::~GPTModel() {
    clear_kv_cache();
}

void GPTModel::load_weights(const std::string& path) {
    // Load model weights
    weights_->load_weights(path);

    // Initialize model after loading weights
    initialize();
}

void GPTModel::initialize() {
    if (initialized_) return;

    // Initialize inference engine
    inference_engine_->initialize();

    // Allocate compute buffers
    size_t batch_size = 1;
    size_t seq_length = config_.max_position_embeddings;
    size_t hidden_size = config_.hidden_size;
    size_t head_dim = hidden_size / config_.num_heads;

    buffers_.attention_mask = std::make_unique<hal::Tensor>(
        std::vector<int64_t>{batch_size, 1, seq_length, seq_length},
        hal::DataType::FLOAT32,
        device_);

    buffers_.position_ids = std::make_unique<hal::Tensor>(
        std::vector<int64_t>{batch_size, seq_length},
        hal::DataType::INT32,
        device_);

    buffers_.hidden_states = std::make_unique<hal::Tensor>(
        std::vector<int64_t>{batch_size, seq_length, hidden_size},
        hal::DataType::FLOAT32,
        device_);

    buffers_.query = std::make_unique<hal::Tensor>(
        std::vector<int64_t>{batch_size, config_.num_heads, seq_length, head_dim},
        hal::DataType::FLOAT32,
        device_);

    buffers_.key = std::make_unique<hal::Tensor>(
        std::vector<int64_t>{batch_size, config_.num_heads, seq_length, head_dim},
        hal::DataType::FLOAT32,
        device_);

    buffers_.value = std::make_unique<hal::Tensor>(
        std::vector<int64_t>{batch_size, config_.num_heads, seq_length, head_dim},
        hal::DataType::FLOAT32,
        device_);

    buffers_.attention_output = std::make_unique<hal::Tensor>(
        std::vector<int64_t>{batch_size, seq_length, hidden_size},
        hal::DataType::FLOAT32,
        device_);

    buffers_.ffn_intermediate = std::make_unique<hal::Tensor>(
        std::vector<int64_t>{batch_size, seq_length, hidden_size * 4},
        hal::DataType::FLOAT32,
        device_);

    buffers_.ffn_output = std::make_unique<hal::Tensor>(
        std::vector<int64_t>{batch_size, seq_length, hidden_size},
        hal::DataType::FLOAT32,
        device_);

    buffers_.logits = std::make_unique<hal::Tensor>(
        std::vector<int64_t>{batch_size, seq_length, config_.vocab_size},
        hal::DataType::FLOAT32,
        device_);

    initialized_ = true;
}

std::string GPTModel::generate(const std::string& prompt,
                             size_t max_tokens,
                             float temperature,
                             float top_p,
                             float top_k) {
    // Initialize KV cache
    init_kv_cache(1, max_tokens);

    // Convert prompt to token IDs
    std::vector<int32_t> input_ids;  // TODO: Implement tokenization
    
    // Generate tokens
    std::vector<int32_t> output_ids = input_ids;
    std::vector<float> logits;

    for (size_t i = 0; i < max_tokens; i++) {
        // Forward pass
        forward(output_ids, logits);

        // Sample next token
        int32_t next_token = sample_token(logits, temperature, top_p, top_k);
        
        // Check for end of generation
        if (next_token == 0) break;  // Assuming 0 is the EOS token
        
        // Append token
        output_ids.push_back(next_token);
        
        // Update KV cache
        update_kv_cache(output_ids.size() - 1, i);
    }

    // Convert tokens to text
    std::string output;  // TODO: Implement detokenization
    return output;
}

void GPTModel::generate_stream(const std::string& prompt,
                             std::function<bool(const std::string&)> callback,
                             size_t max_tokens,
                             float temperature,
                             float top_p,
                             float top_k) {
    // Initialize KV cache
    init_kv_cache(1, max_tokens);

    // Convert prompt to token IDs
    std::vector<int32_t> input_ids;  // TODO: Implement tokenization
    
    // Generate tokens
    std::vector<int32_t> output_ids = input_ids;
    std::vector<float> logits;
    std::string current_text;

    for (size_t i = 0; i < max_tokens; i++) {
        // Forward pass
        forward(output_ids, logits);

        // Sample next token
        int32_t next_token = sample_token(logits, temperature, top_p, top_k);
        
        // Check for end of generation
        if (next_token == 0) break;  // Assuming 0 is the EOS token
        
        // Append token
        output_ids.push_back(next_token);
        
        // Convert new token to text
        std::string new_text;  // TODO: Implement detokenization
        current_text += new_text;
        
        // Call callback with new text
        if (!callback(new_text)) break;
        
        // Update KV cache
        update_kv_cache(output_ids.size() - 1, i);
    }
}

void GPTModel::to_device(hal::Device* device) {
    if (!device) {
        throw std::runtime_error("Device cannot be null");
    }

    device_ = device;
    weights_->to_device(device);
    inference_engine_->update_config(InferenceConfig());  // Reset engine config
    
    // Re-initialize on new device
    initialized_ = false;
    initialize();
}

void GPTModel::create_attention_mask(const std::vector<int32_t>& input_ids) {
    // Create causal attention mask
    size_t seq_length = input_ids.size();
    std::vector<float> mask_data(seq_length * seq_length, -std::numeric_limits<float>::infinity());
    
    for (size_t i = 0; i < seq_length; i++) {
        for (size_t j = 0; j <= i; j++) {
            mask_data[i * seq_length + j] = 0.0f;
        }
    }
    
    device_->memcpy_to_device(buffers_.attention_mask->data(),
                             mask_data.data(),
                             mask_data.size() * sizeof(float));
}

void GPTModel::create_position_ids(const std::vector<int32_t>& input_ids) {
    // Create position IDs
    size_t seq_length = input_ids.size();
    std::vector<int32_t> position_data(seq_length);
    std::iota(position_data.begin(), position_data.end(), 0);
    
    device_->memcpy_to_device(buffers_.position_ids->data(),
                             position_data.data(),
                             position_data.size() * sizeof(int32_t));
}

void GPTModel::apply_rotary_embedding(hal::Tensor* query, hal::Tensor* key) {
    if (!config_.use_rotary_embeddings) return;

    // TODO: Implement rotary embeddings
}

void GPTModel::compute_attention_scores(hal::Tensor* query,
                                      hal::Tensor* key,
                                      hal::Tensor* mask) {
    // TODO: Implement attention score computation
    // If using flash attention, use optimized kernel
}

void GPTModel::apply_attention(hal::Tensor* scores,
                             hal::Tensor* value,
                             hal::Tensor* output) {
    // TODO: Implement attention application
}

void GPTModel::compute_ffn(hal::Tensor* input, hal::Tensor* output) {
    // TODO: Implement feed-forward network computation
}

int32_t GPTModel::sample_token(const std::vector<float>& logits,
                              float temperature,
                              float top_p,
                              float top_k) {
    // Apply temperature
    std::vector<float> probs = logits;
    if (temperature > 0.0f) {
        for (float& p : probs) {
            p /= temperature;
        }
    }

    // Apply softmax
    float max_logit = *std::max_element(probs.begin(), probs.end());
    float sum = 0.0f;
    for (float& p : probs) {
        p = std::exp(p - max_logit);
        sum += p;
    }
    for (float& p : probs) {
        p /= sum;
    }

    // Apply top-k sampling
    if (top_k > 0.0f) {
        std::vector<std::pair<float, size_t>> prob_idx;
        prob_idx.reserve(probs.size());
        for (size_t i = 0; i < probs.size(); i++) {
            prob_idx.emplace_back(probs[i], i);
        }
        
        std::partial_sort(prob_idx.begin(),
                         prob_idx.begin() + static_cast<size_t>(top_k),
                         prob_idx.end(),
                         std::greater<>());
        
        std::fill(probs.begin(), probs.end(), 0.0f);
        for (size_t i = 0; i < static_cast<size_t>(top_k); i++) {
            probs[prob_idx[i].second] = prob_idx[i].first;
        }
    }

    // Apply nucleus (top-p) sampling
    if (top_p < 1.0f) {
        std::vector<std::pair<float, size_t>> prob_idx;
        prob_idx.reserve(probs.size());
        for (size_t i = 0; i < probs.size(); i++) {
            prob_idx.emplace_back(probs[i], i);
        }
        
        std::sort(prob_idx.begin(), prob_idx.end(), std::greater<>());
        
        float cumsum = 0.0f;
        size_t cutoff_idx = prob_idx.size();
        for (size_t i = 0; i < prob_idx.size(); i++) {
            cumsum += prob_idx[i].first;
            if (cumsum > top_p) {
                cutoff_idx = i + 1;
                break;
            }
        }
        
        std::fill(probs.begin(), probs.end(), 0.0f);
        for (size_t i = 0; i < cutoff_idx; i++) {
            probs[prob_idx[i].second] = prob_idx[i].first;
        }
    }

    // Sample from distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    return static_cast<int32_t>(dist(gen));
}

void GPTModel::forward(const std::vector<int32_t>& input_ids,
                      std::vector<float>& logits) {
    // Create attention mask and position IDs
    create_attention_mask(input_ids);
    create_position_ids(input_ids);

    // Get embedding
    auto embedding = weights_->get_tensor("embedding");
    // TODO: Implement embedding lookup

    // Forward through layers
    for (size_t i = 0; i < config_.num_layers; i++) {
        forward_layer(i, buffers_.hidden_states.get(), buffers_.attention_mask.get());
    }

    // Final layer norm
    auto final_ln_weight = weights_->get_tensor("final_ln_weight");
    auto final_ln_bias = weights_->get_tensor("final_ln_bias");
    // TODO: Implement layer normalization

    // Compute logits
    auto lm_head_weight = weights_->get_tensor("lm_head_weight");
    auto lm_head_bias = weights_->get_tensor("lm_head_bias");
    // TODO: Implement logits computation

    // Copy logits to host
    logits.resize(config_.vocab_size);
    device_->memcpy_to_host(logits.data(),
                           buffers_.logits->data(),
                           logits.size() * sizeof(float));
}

void GPTModel::forward_layer(size_t layer_idx,
                           hal::Tensor* hidden_states,
                           hal::Tensor* attention_mask) {
    // Get layer weights
    auto qkv_weight = weights_->get_tensor("layers." + std::to_string(layer_idx) + ".qkv_weight");
    auto qkv_bias = weights_->get_tensor("layers." + std::to_string(layer_idx) + ".qkv_bias");
    auto o_weight = weights_->get_tensor("layers." + std::to_string(layer_idx) + ".o_weight");
    auto o_bias = weights_->get_tensor("layers." + std::to_string(layer_idx) + ".o_bias");
    auto ln1_weight = weights_->get_tensor("layers." + std::to_string(layer_idx) + ".ln1_weight");
    auto ln1_bias = weights_->get_tensor("layers." + std::to_string(layer_idx) + ".ln1_bias");
    auto ffn_weight1 = weights_->get_tensor("layers." + std::to_string(layer_idx) + ".ffn_weight1");
    auto ffn_bias1 = weights_->get_tensor("layers." + std::to_string(layer_idx) + ".ffn_bias1");
    auto ffn_weight2 = weights_->get_tensor("layers." + std::to_string(layer_idx) + ".ffn_weight2");
    auto ffn_bias2 = weights_->get_tensor("layers." + std::to_string(layer_idx) + ".ffn_bias2");
    auto ln2_weight = weights_->get_tensor("layers." + std::to_string(layer_idx) + ".ln2_weight");
    auto ln2_bias = weights_->get_tensor("layers." + std::to_string(layer_idx) + ".ln2_bias");

    // Layer norm 1
    // TODO: Implement layer normalization

    // QKV projection
    // TODO: Implement QKV projection

    // Apply rotary embeddings
    apply_rotary_embedding(buffers_.query.get(), buffers_.key.get());

    // Compute attention
    compute_attention_scores(buffers_.query.get(),
                           buffers_.key.get(),
                           attention_mask);
    apply_attention(buffers_.attention_output.get(),
                   buffers_.value.get(),
                   hidden_states);

    // Layer norm 2
    // TODO: Implement layer normalization

    // Feed-forward network
    compute_ffn(hidden_states, buffers_.ffn_output.get());
}

void GPTModel::init_kv_cache(size_t batch_size, size_t max_length) {
    kv_cache_.current_length = 0;
    kv_cache_.max_length = max_length;

    size_t num_heads = config_.num_heads;
    size_t head_dim = config_.hidden_size / num_heads;

    // Initialize key cache
    kv_cache_.key_cache.resize(config_.num_layers);
    for (auto& cache : kv_cache_.key_cache) {
        cache = std::make_unique<hal::Tensor>(
            std::vector<int64_t>{batch_size, num_heads, max_length, head_dim},
            hal::DataType::FLOAT32,
            device_);
    }

    // Initialize value cache
    kv_cache_.value_cache.resize(config_.num_layers);
    for (auto& cache : kv_cache_.value_cache) {
        cache = std::make_unique<hal::Tensor>(
            std::vector<int64_t>{batch_size, num_heads, max_length, head_dim},
            hal::DataType::FLOAT32,
            device_);
    }
}

void GPTModel::update_kv_cache(size_t layer_idx, size_t position) {
    if (position >= kv_cache_.max_length) {
        throw std::runtime_error("KV cache position out of bounds");
    }

    // Copy current key and value to cache
    size_t offset = position * config_.hidden_size;
    device_->memcpy_to_device(
        static_cast<char*>(kv_cache_.key_cache[layer_idx]->data()) + offset,
        buffers_.key->data(),
        config_.hidden_size * sizeof(float));
    device_->memcpy_to_device(
        static_cast<char*>(kv_cache_.value_cache[layer_idx]->data()) + offset,
        buffers_.value->data(),
        config_.hidden_size * sizeof(float));

    kv_cache_.current_length = std::max(kv_cache_.current_length, position + 1);
}

void GPTModel::clear_kv_cache() {
    kv_cache_.key_cache.clear();
    kv_cache_.value_cache.clear();
    kv_cache_.current_length = 0;
    kv_cache_.max_length = 0;
}

} // namespace deeppowers 