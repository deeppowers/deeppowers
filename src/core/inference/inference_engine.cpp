#include "inference_engine.hpp"
#include <algorithm>
#include <chrono>
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace deeppowers {

InferenceEngine::InferenceEngine(std::shared_ptr<Model> model)
    : model_(std::move(model))
    , is_prepared_(false) {
    if (!model_) {
        throw std::runtime_error("InferenceEngine: Model cannot be null");
    }
}

InferenceEngine::~InferenceEngine() {
    reset();
}

InferenceResult InferenceEngine::generate(
    const std::vector<int>& input_ids,
    const std::vector<int>& attention_mask,
    const InferenceConfig& config
) {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Prepare the model if not already prepared
    if (!is_prepared_) {
        prepare(config);
    }
    
    // Create result structure
    InferenceResult result;
    result.token_ids.resize(config.num_return_sequences);
    result.logprobs.resize(config.num_return_sequences);
    result.stop_reasons.resize(config.num_return_sequences, "");
    
    // Validate input
    if (input_ids.empty()) {
        throw std::runtime_error("InferenceEngine: Input token IDs cannot be empty");
    }
    
    // Get model configuration
    auto model_config = model_->config();
    int eos_token_id = std::stoi(model_config.value("eos_token_id", "0"));
    int pad_token_id = std::stoi(model_config.value("pad_token_id", "0"));
    int vocab_size = std::stoi(model_config.value("vocab_size", "0"));
    
    // For each requested sequence
    for (int seq_idx = 0; seq_idx < config.num_return_sequences; ++seq_idx) {
        // Initialize output with input
        std::vector<int> output_ids = input_ids;
        std::vector<float> output_logprobs;
        
        // Main generation loop
        while (output_ids.size() < static_cast<size_t>(config.max_length) && 
              !should_stop(output_ids, output_ids.size(), config)) {
            
            // Prepare input tensor
            Tensor input_tensor = prepare_inputs(output_ids, attention_mask);
            
            // Run forward pass
            Tensor logits = model_->forward(input_tensor);
            
            // Extract logits for the last token
            // For simplicity, we're assuming logits shape is [batch_size, sequence_length, vocab_size]
            // and we're interested in the last token's logits
            
            // Sample next token
            int next_token = sample_token(logits, output_ids, config);
            float logprob = 0.0f; // Calculate actual logprob from logits
            
            // Add to output
            output_ids.push_back(next_token);
            output_logprobs.push_back(logprob);
            
            // Check for early stopping
            if (next_token == eos_token_id) {
                result.stop_reasons[seq_idx] = "eos_token";
                break;
            }
            
            // Check for minimum length
            if (output_ids.size() >= static_cast<size_t>(config.min_length) && 
                config.early_stopping) {
                // Implement early stopping logic if needed
            }
        }
        
        // Add results for this sequence
        result.token_ids[seq_idx] = output_ids;
        result.logprobs[seq_idx] = output_logprobs;
        
        // Set stop reason if not already set
        if (result.stop_reasons[seq_idx].empty()) {
            if (output_ids.size() >= static_cast<size_t>(config.max_length)) {
                result.stop_reasons[seq_idx] = "max_length";
            } else {
                result.stop_reasons[seq_idx] = "finished";
            }
        }
    }
    
    // Calculate generation time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = end_time - start_time;
    result.generation_time = elapsed.count();
    
    return result;
}

std::vector<InferenceResult> InferenceEngine::generate_batch(
    const std::vector<std::vector<int>>& batch_input_ids,
    const std::vector<std::vector<int>>& batch_attention_mask,
    const InferenceConfig& config
) {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Prepare the model if not already prepared
    if (!is_prepared_) {
        prepare(config);
    }
    
    // Create results vector
    std::vector<InferenceResult> results(batch_input_ids.size());
    
    // Initialize each result
    for (size_t i = 0; i < results.size(); ++i) {
        results[i].token_ids.resize(config.num_return_sequences);
        results[i].logprobs.resize(config.num_return_sequences);
        results[i].stop_reasons.resize(config.num_return_sequences, "");
    }
    
    // TODO: Implement actual batch generation
    // For now, we'll generate sequentially
    
    for (size_t batch_idx = 0; batch_idx < batch_input_ids.size(); ++batch_idx) {
        // For each sample in the batch
        const auto& input_ids = batch_input_ids[batch_idx];
        
        // Create attention mask if empty
        std::vector<int> attention_mask;
        if (batch_attention_mask.size() > batch_idx) {
            attention_mask = batch_attention_mask[batch_idx];
        } else {
            attention_mask.resize(input_ids.size(), 1);
        }
        
        // Generate for this sample
        InferenceResult result = generate(input_ids, attention_mask, config);
        
        // Store the result
        results[batch_idx] = result;
    }
    
    // Update generation time to total batch time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = end_time - start_time;
    float total_time = elapsed.count();
    
    for (auto& result : results) {
        result.generation_time = total_time / results.size();
    }
    
    return results;
}

void InferenceEngine::generate_stream(
    const std::vector<int>& input_ids,
    StreamingCallback callback,
    const InferenceConfig& config
) {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Prepare the model if not already prepared
    if (!is_prepared_) {
        prepare(config);
    }
    
    // Create result structure
    InferenceResult result;
    result.token_ids.resize(1);
    result.logprobs.resize(1);
    result.stop_reasons.resize(1, "");
    
    // Initialize output with input
    std::vector<int> output_ids = input_ids;
    std::vector<float> output_logprobs;
    
    // Create dummy attention mask if not provided
    std::vector<int> attention_mask(input_ids.size(), 1);
    
    // Get model configuration
    auto model_config = model_->config();
    int eos_token_id = std::stoi(model_config.value("eos_token_id", "0"));
    
    // Main generation loop
    bool should_continue = true;
    while (output_ids.size() < static_cast<size_t>(config.max_length) && 
          !should_stop(output_ids, output_ids.size(), config) && 
          should_continue) {
        
        // Prepare input tensor
        Tensor input_tensor = prepare_inputs(output_ids, attention_mask);
        
        // Run forward pass
        Tensor logits = model_->forward(input_tensor);
        
        // Sample next token
        int next_token = sample_token(logits, output_ids, config);
        float logprob = 0.0f; // Calculate actual logprob from logits
        
        // Add to output
        output_ids.push_back(next_token);
        output_logprobs.push_back(logprob);
        
        // Update result for callback
        result.token_ids[0] = output_ids;
        result.logprobs[0] = output_logprobs;
        
        // Calculate current generation time
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = current_time - start_time;
        result.generation_time = elapsed.count();
        
        // Check for EOS
        if (next_token == eos_token_id) {
            result.stop_reasons[0] = "eos_token";
            should_continue = callback(result);
            break;
        }
        
        // Call the callback with current state
        should_continue = callback(result);
    }
    
    // Final update for result if not already finished
    if (result.stop_reasons[0].empty()) {
        if (output_ids.size() >= static_cast<size_t>(config.max_length)) {
            result.stop_reasons[0] = "max_length";
        } else if (!should_continue) {
            result.stop_reasons[0] = "user_cancelled";
        } else {
            result.stop_reasons[0] = "finished";
        }
        
        // Final callback
        callback(result);
    }
}

void InferenceEngine::prepare(const InferenceConfig& config) {
    // Set device
    model_->to(config.device);
    
    // Set precision based on config
    if (config.use_mixed_precision) {
        model_->set_precision(PrecisionMode::MIXED);
    } else {
        model_->set_precision(PrecisionMode::FULL);
    }
    
    // Initialize KV cache if needed
    if (config.use_kv_cache) {
        // Get hidden size from model config
        auto model_config = model_->config();
        int hidden_size = std::stoi(model_config.value("hidden_size", "768"));
        
        // Initialize caches
        init_caches(config.batch_size, config.max_length, hidden_size);
    }
    
    is_prepared_ = true;
}

void InferenceEngine::reset() {
    // Clear caches
    key_cache_.clear();
    value_cache_.clear();
    
    is_prepared_ = false;
}

std::shared_ptr<Model> InferenceEngine::model() const {
    return model_;
}

void InferenceEngine::set_model(std::shared_ptr<Model> model) {
    if (!model) {
        throw std::runtime_error("InferenceEngine: Model cannot be null");
    }
    
    model_ = std::move(model);
    reset();
}

Tensor InferenceEngine::prepare_inputs(
    const std::vector<int>& input_ids,
    const std::vector<int>& attention_mask
) {
    // Create input tensor with shape [1, sequence_length]
    std::vector<size_t> shape = {1, input_ids.size()};
    Tensor input_tensor(shape, DataType::INT32);
    
    // Copy input_ids to tensor
    int* data = input_tensor.data<int>();
    for (size_t i = 0; i < input_ids.size(); ++i) {
        data[i] = input_ids[i];
    }
    
    // In a real implementation, we would also handle attention_mask
    // and potentially position_ids here
    
    return input_tensor;
}

int InferenceEngine::sample_token(
    const Tensor& logits,
    const std::vector<int>& prev_tokens,
    const InferenceConfig& config
) {
    // Get the logits for the last token
    // Assuming logits shape is [batch_size, sequence_length, vocab_size]
    const float* logits_data = logits.data<float>();
    
    // Get vocab size from the last dimension of logits
    int vocab_size = logits.shape()[2];
    
    // Extract logits for the last token
    std::vector<float> token_logits(vocab_size);
    size_t offset = logits.shape()[1] - 1;
    for (int i = 0; i < vocab_size; ++i) {
        token_logits[i] = logits_data[offset * vocab_size + i];
    }
    
    // Apply temperature
    if (config.temperature != 1.0f) {
        for (auto& logit : token_logits) {
            logit /= config.temperature;
        }
    }
    
    // Apply repetition penalty
    if (config.repetition_penalty != 1.0f) {
        for (int token_id : prev_tokens) {
            if (token_id < 0 || token_id >= vocab_size) continue;
            token_logits[token_id] /= config.repetition_penalty;
        }
    }
    
    // Apply top-k filtering
    if (config.top_k > 0 && config.top_k < vocab_size) {
        // Find the top_k largest values
        std::vector<int> top_indices(vocab_size);
        for (int i = 0; i < vocab_size; ++i) top_indices[i] = i;
        
        std::partial_sort(top_indices.begin(), top_indices.begin() + config.top_k, top_indices.end(),
            [&token_logits](int a, int b) { return token_logits[a] > token_logits[b]; });
        
        // Zero out values not in top-k
        std::vector<float> filtered_logits(vocab_size, -INFINITY);
        for (int i = 0; i < config.top_k; ++i) {
            filtered_logits[top_indices[i]] = token_logits[top_indices[i]];
        }
        token_logits = filtered_logits;
    }
    
    // Apply top-p (nucleus) filtering
    if (config.top_p < 1.0f) {
        // Convert logits to probabilities
        std::vector<float> probs(vocab_size);
        
        // Compute softmax
        float max_logit = *std::max_element(token_logits.begin(), token_logits.end());
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            probs[i] = std::exp(token_logits[i] - max_logit);
            sum_exp += probs[i];
        }
        for (int i = 0; i < vocab_size; ++i) {
            probs[i] /= sum_exp;
        }
        
        // Sort indices by probability
        std::vector<int> sorted_indices(vocab_size);
        for (int i = 0; i < vocab_size; ++i) sorted_indices[i] = i;
        std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&probs](int a, int b) { return probs[a] > probs[b]; });
        
        // Compute cumulative probabilities
        float cumsum = 0.0f;
        std::vector<float> filtered_logits(vocab_size, -INFINITY);
        
        for (int idx : sorted_indices) {
            cumsum += probs[idx];
            filtered_logits[idx] = token_logits[idx];
            if (cumsum >= config.top_p) break;
        }
        
        token_logits = filtered_logits;
    }
    
    // Sample from the distribution or take the argmax
    int next_token;
    if (config.do_sample) {
        // Convert to probabilities
        std::vector<float> probs(vocab_size);
        
        // Compute softmax
        float max_logit = *std::max_element(token_logits.begin(), token_logits.end());
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            probs[i] = std::exp(token_logits[i] - max_logit);
            sum_exp += probs[i];
        }
        for (int i = 0; i < vocab_size; ++i) {
            probs[i] /= sum_exp;
        }
        
        // Sample from the distribution
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<int> distribution(probs.begin(), probs.end());
        next_token = distribution(gen);
    } else {
        // Greedy selection (argmax)
        next_token = std::max_element(token_logits.begin(), token_logits.end()) - token_logits.begin();
    }
    
    return next_token;
}

bool InferenceEngine::should_stop(
    const std::vector<int>& output_ids,
    int current_length,
    const InferenceConfig& config
) {
    // Get model configuration
    auto model_config = model_->config();
    int eos_token_id = std::stoi(model_config.value("eos_token_id", "0"));
    
    // Check for max length
    if (current_length >= config.max_length) {
        return true;
    }
    
    // Check for EOS token
    if (!output_ids.empty() && output_ids.back() == eos_token_id) {
        return true;
    }
    
    // Additional stopping criteria can be added here
    
    return false;
}

void InferenceEngine::init_caches(int batch_size, int max_length, int hidden_size) {
    // Clear existing caches
    key_cache_.clear();
    value_cache_.clear();
    
    // Get model configuration
    auto model_config = model_->config();
    int num_layers = std::stoi(model_config.value("num_layers", "12"));
    int num_heads = std::stoi(model_config.value("num_attention_heads", "12"));
    int head_size = hidden_size / num_heads;
    
    // Create key and value caches for each layer
    for (int i = 0; i < num_layers; ++i) {
        // Key cache with shape [batch_size, num_heads, max_length, head_size]
        std::vector<size_t> key_shape = {
            static_cast<size_t>(batch_size),
            static_cast<size_t>(num_heads),
            static_cast<size_t>(max_length),
            static_cast<size_t>(head_size)
        };
        key_cache_.emplace_back(key_shape, DataType::FLOAT32);
        
        // Value cache with shape [batch_size, num_heads, max_length, head_size]
        std::vector<size_t> value_shape = {
            static_cast<size_t>(batch_size),
            static_cast<size_t>(num_heads),
            static_cast<size_t>(max_length),
            static_cast<size_t>(head_size)
        };
        value_cache_.emplace_back(value_shape, DataType::FLOAT32);
    }
}

} // namespace deeppowers 