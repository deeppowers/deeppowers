#include "gpt_model.hpp"
#include "gpt_kernels.hpp"
#include "quantization_manager.hpp"
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>

namespace deeppowers {

GPTModel::GPTModel(const GPTConfig& config, hal::Device* device)
    : Model(config, device)
    , config_(config)
    , weights_(std::make_unique<GPTWeights>(device))
    , kernels_({nullptr, nullptr, nullptr, nullptr, nullptr}) {
    
    // Initialize compute kernels based on precision
    hal::DataType compute_type = config.use_fp16 ? hal::DataType::FLOAT16 : hal::DataType::FLOAT32;
    
    // Create kernels with appropriate precision
    kernels_.attention_kernel = device->create_kernel("attention", compute_type);
    kernels_.ffn_kernel = device->create_kernel("ffn", compute_type);
    kernels_.layernorm_kernel = device->create_kernel("layernorm", compute_type);
    kernels_.softmax_kernel = device->create_kernel("softmax", compute_type);
    kernels_.sampling_kernel = device->create_kernel("sampling", compute_type);
    
    // Initialize weights with appropriate precision
    init_weights(config);
    
    // Initialize quantization manager
    init_quantization_manager();
}

void GPTModel::load_weights(const std::string& path) {
    weights_->load_from_file(path);
}

void GPTModel::save_weights(const std::string& path) const {
    weights_->save_to_file(path);
}

void GPTModel::to_device(hal::Device* device) {
    if (device == device_) {
        return;
    }
    
    weights_->to_device(device);
    device_ = device;
    
    // TODO: Re-initialize compute kernels
}

void GPTModel::to_host() {
    weights_->to_host();
}

std::vector<int32_t> GPTModel::generate(
    const std::vector<int32_t>& input_ids,
    size_t max_length,
    float temperature,
    float top_p,
    float top_k,
    const std::vector<int32_t>& stop_ids) {
    
    // Wrap single input into batch processing
    std::vector<std::vector<int32_t>> batch_input = {input_ids};
    auto batch_output = generate_batch(batch_input, max_length,
                                     temperature, top_p, top_k, stop_ids);
    return batch_output[0];
}

std::vector<std::vector<int32_t>> GPTModel::generate_batch(
    const std::vector<std::vector<int32_t>>& batch_input_ids,
    size_t max_length,
    float temperature,
    float top_p,
    float top_k,
    const std::vector<int32_t>& stop_ids) {
    
    if (batch_input_ids.empty()) {
        return {};
    }
    
    size_t batch_size = batch_input_ids.size();
    if (batch_size > config_.max_batch_size) {
        throw std::runtime_error("Batch size exceeds maximum allowed");
    }
    
    // Initialize KV cache
    init_kv_cache(batch_size, max_length);
    
    // Prepare outputs
    std::vector<std::vector<int32_t>> outputs = batch_input_ids;
    std::vector<bool> finished(batch_size, false);
    
    // Generate tokens
    for (size_t pos = 0; pos < max_length; ++pos) {
        // Create attention mask and position IDs
        create_attention_mask(outputs[0]);  // Use length of first sequence
        create_position_ids(outputs[0]);
        
        // Perform forward propagation for each position
        // TODO: Implement actual forward propagation logic
        
        // Sample for each sequence
        for (size_t b = 0; b < batch_size; ++b) {
            if (finished[b]) {
                continue;
            }
            
            // TODO: Get logits and sample
            std::vector<float> logits;  // Get from last layer output
            int32_t next_token = sample_token(logits, temperature, top_p, top_k);
            
            // Add generated token
            outputs[b].push_back(next_token);
            
            // Check if stop token is generated
            if (std::find(stop_ids.begin(), stop_ids.end(), next_token) != stop_ids.end()) {
                finished[b] = true;
            }
        }
        
        // Check if all sequences are finished
        if (std::all_of(finished.begin(), finished.end(), [](bool f) { return f; })) {
            break;
        }
        
        // Update KV cache
        update_kv_cache(pos, pos + 1);
    }
    
    // Clean up cache
    clear_kv_cache();
    
    return outputs;
}

void GPTModel::create_attention_mask(const std::vector<int32_t>& input_ids) {
    size_t seq_length = input_ids.size();
    if (!cache_.attention_mask || 
        cache_.attention_mask->shape()[1] < seq_length) {
        
        // Create new attention mask
        std::vector<int64_t> mask_shape = {1, seq_length, seq_length};
        cache_.attention_mask = std::make_unique<hal::Tensor>(
            mask_shape, hal::DataType::FLOAT32, device_);
        
        // Initialize to causal mask
        std::vector<float> mask_data(seq_length * seq_length, -INFINITY);
        for (size_t i = 0; i < seq_length; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                mask_data[i * seq_length + j] = 0.0f;
            }
        }
        
        cache_.attention_mask->copy_from_host(mask_data.data());
    }
}

void GPTModel::create_position_ids(const std::vector<int32_t>& input_ids) {
    size_t seq_length = input_ids.size();
    if (!cache_.position_ids || 
        cache_.position_ids->shape()[1] < seq_length) {
        
        // Create new position IDs
        std::vector<int64_t> pos_shape = {1, seq_length};
        cache_.position_ids = std::make_unique<hal::Tensor>(
            pos_shape, hal::DataType::INT32, device_);
        
        // Initialize to sequential position IDs
        std::vector<int32_t> pos_data(seq_length);
        for (size_t i = 0; i < seq_length; ++i) {
            pos_data[i] = static_cast<int32_t>(i);
        }
        
        cache_.position_ids->copy_from_host(pos_data.data());
    }
}

void GPTModel::apply_rotary_embedding(hal::Tensor* query, hal::Tensor* key) {
    if (!config_.use_rotary_embedding) {
        return;
    }
    
    const auto& shape = query->shape();
    int batch_size = shape[0];
    int num_heads = shape[1];
    int seq_length = shape[2];
    int head_dim = shape[3];
    
    dim3 grid(seq_length, num_heads, batch_size);
    dim3 block(head_dim / 2);
    
    kernels::rotary_embedding_kernel<float><<<grid, block>>>(
        query->data<float>(),
        key->data<float>(),
        batch_size,
        num_heads,
        seq_length,
        head_dim,
        config_.rotary_embedding_base);
}

void GPTModel::compute_attention_scores(
    hal::Tensor* query, hal::Tensor* key, hal::Tensor* mask) {
    
    const auto& shape = query->shape();
    int batch_size = shape[0];
    int num_heads = shape[1];
    int seq_length = shape[2];
    int head_dim = shape[3];
    
    dim3 grid(
        (seq_length + BLOCK_SIZE - 1) / BLOCK_SIZE,
        num_heads,
        batch_size);
    dim3 block(BLOCK_SIZE);
    
    if (config_.use_flash_attention) {
        kernels::flash_attention_kernel<float><<<grid, block>>>(
            query->data<float>(),
            key->data<float>(),
            value->data<float>(),
            mask ? mask->data<float>() : nullptr,
            output->data<float>(),
            batch_size,
            num_heads,
            seq_length,
            head_dim);
    } else {
        // TODO: Implement standard attention calculation
    }
}

void GPTModel::apply_attention(
    hal::Tensor* scores, hal::Tensor* value, hal::Tensor* output) {
    // TODO: Implement attention application
}

void GPTModel::compute_ffn(hal::Tensor* input, hal::Tensor* output) {
    const auto& shape = input->shape();
    int batch_size = shape[0];
    int seq_length = shape[1];
    int hidden_size = shape[2];
    
    dim3 grid(1, seq_length, batch_size);
    dim3 block(BLOCK_SIZE);
    
    kernels::ffn_kernel<float><<<grid, block>>>(
        input->data<float>(),
        weights_->get_weight("ffn_inter_weight")->data<float>(),
        weights_->get_weight("ffn_inter_bias")->data<float>(),
        weights_->get_weight("ffn_out_weight")->data<float>(),
        weights_->get_weight("ffn_out_bias")->data<float>(),
        output->data<float>(),
        batch_size,
        seq_length,
        hidden_size,
        config_.intermediate_size);
}

int32_t GPTModel::sample_token(
    const std::vector<float>& logits,
    float temperature,
    float top_p,
    float top_k) {
    
    if (logits.empty()) {
        return 0;
    }
    
    // Apply temperature
    std::vector<float> probs = logits;
    if (temperature > 0) {
        for (float& p : probs) {
            p /= temperature;
        }
    }
    
    // Apply softmax
    float max_logit = *std::max_element(probs.begin(), probs.end());
    float sum_exp = 0.0f;
    for (float& p : probs) {
        p = std::exp(p - max_logit);
        sum_exp += p;
    }
    for (float& p : probs) {
        p /= sum_exp;
    }
    
    // Apply top-k sampling
    if (top_k > 0 && top_k < probs.size()) {
        std::vector<std::pair<float, int32_t>> prob_idx;
        for (size_t i = 0; i < probs.size(); ++i) {
            prob_idx.emplace_back(probs[i], i);
        }
        std::partial_sort(prob_idx.begin(),
                         prob_idx.begin() + static_cast<int32_t>(top_k),
                         prob_idx.end(),
                         std::greater<>());
        
        std::fill(probs.begin(), probs.end(), 0.0f);
        float new_sum = 0.0f;
        for (size_t i = 0; i < top_k; ++i) {
            probs[prob_idx[i].second] = prob_idx[i].first;
            new_sum += prob_idx[i].first;
        }
        for (float& p : probs) {
            p /= new_sum;
        }
    }
    
    // Apply top-p sampling
    if (top_p > 0 && top_p < 1.0f) {
        std::vector<std::pair<float, int32_t>> prob_idx;
        for (size_t i = 0; i < probs.size(); ++i) {
            if (probs[i] > 0) {
                prob_idx.emplace_back(probs[i], i);
            }
        }
        std::sort(prob_idx.begin(), prob_idx.end(), std::greater<>());
        
        float cumsum = 0.0f;
        size_t cutoff_idx = prob_idx.size();
        for (size_t i = 0; i < prob_idx.size(); ++i) {
            cumsum += prob_idx[i].first;
            if (cumsum > top_p) {
                cutoff_idx = i + 1;
                break;
            }
        }
        
        std::fill(probs.begin(), probs.end(), 0.0f);
        float new_sum = 0.0f;
        for (size_t i = 0; i < cutoff_idx; ++i) {
            probs[prob_idx[i].second] = prob_idx[i].first;
            new_sum += prob_idx[i].first;
        }
        for (float& p : probs) {
            p /= new_sum;
        }
    }
    
    // Sample
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    float r = dis(gen);
    
    float cumsum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumsum += probs[i];
        if (r <= cumsum) {
            return static_cast<int32_t>(i);
        }
    }
    
    return static_cast<int32_t>(probs.size() - 1);
}

void GPTModel::init_kv_cache(size_t batch_size, size_t max_length) {
    // Clear old caches
    clear_kv_cache();
    
    // Create new caches
    size_t num_layers = config_.num_layers;
    size_t num_heads = config_.num_attention_heads;
    size_t head_dim = config_.hidden_size / num_heads;
    
    cache_.key_cache.resize(num_layers);
    cache_.value_cache.resize(num_layers);
    
    for (size_t i = 0; i < num_layers; ++i) {
        // Create key and value cache tensors
        std::vector<int64_t> cache_shape = {
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(num_heads),
            static_cast<int64_t>(max_length),
            static_cast<int64_t>(head_dim)
        };
        
        cache_.key_cache[i] = std::make_unique<hal::Tensor>(
            cache_shape, hal::DataType::FLOAT32, device_);
        cache_.value_cache[i] = std::make_unique<hal::Tensor>(
            cache_shape, hal::DataType::FLOAT32, device_);
            
        // Initialize caches to zero
        cache_.key_cache[i]->fill(0.0f);
        cache_.value_cache[i]->fill(0.0f);
    }
}

void GPTModel::update_kv_cache(size_t layer_idx, size_t position) {
    if (layer_idx >= cache_.key_cache.size() || 
        layer_idx >= cache_.value_cache.size()) {
        throw std::runtime_error("Layer index out of range for KV cache");
    }
    
    auto* key_cache = cache_.key_cache[layer_idx].get();
    auto* value_cache = cache_.value_cache[layer_idx].get();
    
    if (!key_cache || !value_cache) {
        throw std::runtime_error("KV cache not initialized");
    }
    
    const auto& shape = key_cache->shape();
    int batch_size = shape[0];
    int num_heads = shape[1];
    int head_dim = shape[3];
    
    // Update key cache
    dim3 grid(1, num_heads, batch_size);
    dim3 block(head_dim);
    
    kernels::update_kv_cache_kernel<float><<<grid, block>>>(
        cache_.key->data<float>(),
        key_cache->data<float>(),
        cache_.value->data<float>(),
        value_cache->data<float>(),
        batch_size,
        num_heads,
        position,
        head_dim);
}

void GPTModel::clear_kv_cache() {
    cache_.key_cache.clear();
    cache_.value_cache.clear();
}

void GPTModel::compress_tensors() {
    // Compress attention cache
    optimize_tensor_memory();
    
    // Clean up temporary buffers
    cache_.temp_buffer.reset();
    cache_.attention_scores.reset();
    
    // Compress other compute caches
    if (cache_.hidden_states) {
        auto shape = cache_.hidden_states->shape();
        if (shape[1] > config_.max_sequence_length) {
            resize_cache(shape[0], config_.max_sequence_length);
        }
    }
}

size_t GPTModel::get_cache_size() const {
    size_t total_size = 0;
    
    // Calculate KV cache size
    for (const auto& k : cache_.key_cache) {
        if (k) total_size += k->size_in_bytes();
    }
    for (const auto& v : cache_.value_cache) {
        if (v) total_size += v->size_in_bytes();
    }
    
    // Calculate other cache sizes
    if (cache_.hidden_states) total_size += cache_.hidden_states->size_in_bytes();
    if (cache_.attention_output) total_size += cache_.attention_output->size_in_bytes();
    if (cache_.mlp_output) total_size += cache_.mlp_output->size_in_bytes();
    if (cache_.norm_output) total_size += cache_.norm_output->size_in_bytes();
    if (cache_.logits) total_size += cache_.logits->size_in_bytes();
    if (cache_.query) total_size += cache_.query->size_in_bytes();
    if (cache_.key) total_size += cache_.key->size_in_bytes();
    if (cache_.value) total_size += cache_.value->size_in_bytes();
    if (cache_.attention_scores) total_size += cache_.attention_scores->size_in_bytes();
    if (cache_.temp_buffer) total_size += cache_.temp_buffer->size_in_bytes();
    
    return total_size;
}

void GPTModel::resize_cache(size_t batch_size, size_t seq_length) {
    // Reallocate compute caches
    std::vector<int64_t> hidden_shape = {
        static_cast<int64_t>(batch_size),
        static_cast<int64_t>(seq_length),
        static_cast<int64_t>(config_.hidden_size)
    };
    
    cache_.hidden_states = std::make_unique<hal::Tensor>(
        hidden_shape, hal::DataType::FLOAT32, device_);
    cache_.attention_output = std::make_unique<hal::Tensor>(
        hidden_shape, hal::DataType::FLOAT32, device_);
    cache_.mlp_output = std::make_unique<hal::Tensor>(
        hidden_shape, hal::DataType::FLOAT32, device_);
    cache_.norm_output = std::make_unique<hal::Tensor>(
        hidden_shape, hal::DataType::FLOAT32, device_);
    
    // Reallocate attention-related caches
    std::vector<int64_t> attention_shape = {
        static_cast<int64_t>(batch_size),
        static_cast<int64_t>(config_.num_attention_heads),
        static_cast<int64_t>(seq_length),
        static_cast<int64_t>(config_.hidden_size / config_.num_attention_heads)
    };
    
    cache_.query = std::make_unique<hal::Tensor>(
        attention_shape, hal::DataType::FLOAT32, device_);
    cache_.key = std::make_unique<hal::Tensor>(
        attention_shape, hal::DataType::FLOAT32, device_);
    cache_.value = std::make_unique<hal::Tensor>(
        attention_shape, hal::DataType::FLOAT32, device_);
}

void GPTModel::optimize_tensor_memory() {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    
    // Track current memory usage
    size_t current_usage = get_cache_size();
    size_t device_free, device_total;
    
    if (auto* cuda_device = dynamic_cast<hal::CUDADevice*>(device_)) {
        CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
        
        // If memory usage is too high, apply optimization strategies
        if (device_free < device_total * 0.2) {  // Less than 20% free memory
            // Strategy 1: Compress KV cache
            compress_kv_cache();
            
            // Strategy 2: Release unused tensors
            release_unused_tensors();
            
            // Strategy 3: Defragment memory
            defragment_memory();
        }
    }
}

void GPTModel::compress_kv_cache() {
    // Compress KV cache using quantization
    for (size_t i = 0; i < cache_.key_cache.size(); ++i) {
        if (cache_.key_cache[i] && cache_.value_cache[i]) {
            // Quantize to INT8 for storage
            auto compressed_key = quantize_tensor(cache_.key_cache[i].get(), hal::DataType::INT8);
            auto compressed_value = quantize_tensor(cache_.value_cache[i].get(), hal::DataType::INT8);
            
            // Replace original tensors with compressed versions
            cache_.key_cache[i] = std::move(compressed_key);
            cache_.value_cache[i] = std::move(compressed_value);
        }
    }
}

void GPTModel::release_unused_tensors() {
    // Release temporary tensors that are not needed
    cache_.attention_scores.reset();
    cache_.query.reset();
    cache_.key.reset();
    cache_.value.reset();
    
    // Clear historical metrics if too many
    while (historical_metrics_.size() > 50) {
        historical_metrics_.pop_front();
    }
}

void GPTModel::defragment_memory() {
    std::vector<hal::Tensor*> active_tensors;
    
    // Collect all active tensors
    for (const auto& tensor : cache_.key_cache) {
        if (tensor) active_tensors.push_back(tensor.get());
    }
    for (const auto& tensor : cache_.value_cache) {
        if (tensor) active_tensors.push_back(tensor.get());
    }
    
    // Sort tensors by address
    std::sort(active_tensors.begin(), active_tensors.end(),
              [](const hal::Tensor* a, const hal::Tensor* b) {
                  return a->data() < b->data();
              });
    
    // Reallocate and copy tensors sequentially
    for (auto* tensor : active_tensors) {
        auto new_tensor = std::make_unique<hal::Tensor>(
            tensor->shape(), tensor->dtype(), device_);
        device_->memcpy_to_device(new_tensor->data(), tensor->data(), tensor->size_in_bytes());
        
        // Replace old tensor with new one
        // Note: This requires finding the tensor in our cache structures
        replace_tensor_in_cache(tensor, new_tensor.release());
    }
}

std::unique_ptr<hal::Tensor> GPTModel::quantize_tensor(
    const hal::Tensor* tensor,
    hal::DataType target_dtype) {
    
    if (!tensor) return nullptr;
    
    const auto& shape = tensor->shape();
    auto quantized = std::make_unique<hal::Tensor>(shape, target_dtype, device_);
    
    // Implement quantization based on target dtype
    if (target_dtype == hal::DataType::INT8) {
        // INT8 quantization
        const float* src_data = static_cast<const float*>(tensor->data());
        int8_t* dst_data = static_cast<int8_t*>(quantized->data());
        
        // Find scale factor
        float max_abs = 0.0f;
        size_t num_elements = tensor->size_in_bytes() / sizeof(float);
        
        for (size_t i = 0; i < num_elements; ++i) {
            max_abs = std::max(max_abs, std::abs(src_data[i]));
        }
        
        float scale = max_abs / 127.0f;
        
        // Store scale factor in tensor metadata
        quantized->set_scale(scale);
        
        // Quantize values
        for (size_t i = 0; i < num_elements; ++i) {
            dst_data[i] = static_cast<int8_t>(src_data[i] / scale);
        }
    }
    
    return quantized;
}

void GPTModel::forward(const std::vector<int32_t>& input_ids,
                      hal::Tensor* output,
                      bool use_cache) {
    // Create attention mask and position IDs
    create_attention_mask(input_ids);
    create_position_ids(input_ids);
    
    // Embed input tokens
    embed_inputs(input_ids, cache_.hidden_states.get());
    
    // Collect statistics or apply quantization
    collect_calibration_statistics(cache_.hidden_states.get(), "embedding_output");
    apply_activation_quantization(cache_.hidden_states.get(), "embedding_output");
    
    // Forward through transformer layers
    for (size_t i = 0; i < config_.num_layers; ++i) {
        forward_transformer_layer(i, cache_.hidden_states.get(), use_cache);
    }
    
    // Apply final layer norm
    apply_layer_norm(cache_.hidden_states.get(),
                    weights_->get_weight("final_ln_weight"),
                    weights_->get_weight("final_ln_bias"),
                    cache_.norm_output.get());
    
    // Collect statistics or apply quantization
    collect_calibration_statistics(cache_.norm_output.get(), "final_output");
    apply_activation_quantization(cache_.norm_output.get(), "final_output");
    
    // Compute logits
    compute_logits(cache_.norm_output.get(), output);
    
    // If calibration stage and enough samples collected, complete calibration
    if (config_.quant_config.method == QuantizationMethod::POST_TRAINING &&
        !quant_manager_->is_calibrated()) {
        size_t total_samples = 0;
        for (const auto& [name, params] : quant_params_) {
            total_samples += params.num_samples;
        }
        
        if (total_samples >= config_.quant_config.calibration_samples) {
            quant_manager_->finalize_calibration();
            apply_weight_quantization();
            activations_quantized_ = true;
        }
    }
}

void GPTModel::forward_batch(const std::vector<std::vector<int32_t>>& batch_input_ids,
                           hal::Tensor* output,
                           bool use_cache) {
    size_t batch_size = batch_input_ids.size();
    size_t max_seq_length = 0;
    
    // Find maximum sequence length
    for (const auto& input_ids : batch_input_ids) {
        max_seq_length = std::max(max_seq_length, input_ids.size());
    }
    
    // Resize caches if needed
    resize_cache(batch_size, max_seq_length);
    
    // Create padded input tensor
    std::vector<int32_t> padded_input_ids;
    padded_input_ids.reserve(batch_size * max_seq_length);
    
    for (const auto& input_ids : batch_input_ids) {
        padded_input_ids.insert(padded_input_ids.end(), input_ids.begin(), input_ids.end());
        padded_input_ids.insert(padded_input_ids.end(), 
                               max_seq_length - input_ids.size(), 
                               0);  // Padding token ID
    }
    
    // Forward pass with padded inputs
    forward(padded_input_ids, output, use_cache);
}

void GPTModel::embed_inputs(const std::vector<int32_t>& input_ids,
                          hal::Tensor* embedded) {
    size_t batch_size = 1;  // For single sequence
    size_t seq_length = input_ids.size();
    
    // Launch embedding lookup kernel
    dim3 grid(1, seq_length, batch_size);
    dim3 block(config_.hidden_size);
    
    kernels::embedding_lookup_kernel<float><<<grid, block>>>(
        reinterpret_cast<const int32_t*>(input_ids.data()),
        weights_->get_weight("token_embedding")->data<float>(),
        weights_->get_weight("position_embedding")->data<float>(),
        embedded->data<float>(),
        batch_size,
        seq_length,
        config_.hidden_size);
}

void GPTModel::forward_transformer_layer(size_t layer_idx,
                                       hal::Tensor* hidden_states,
                                       bool use_cache) {
    std::string prefix = "layer_" + std::to_string(layer_idx) + "_";
    
    // Collect statistics or apply quantization
    collect_calibration_statistics(hidden_states, prefix + "input");
    apply_activation_quantization(hidden_states, prefix + "input");
    
    // Apply pre-attention layer norm
    apply_layer_norm(hidden_states,
                    weights_->get_weight(prefix + "attn_ln_weight"),
                    weights_->get_weight(prefix + "attn_ln_bias"),
                    cache_.norm_output.get());
    
    // Collect statistics or apply quantization
    collect_calibration_statistics(cache_.norm_output.get(), prefix + "attn_ln_output");
    apply_activation_quantization(cache_.norm_output.get(), prefix + "attn_ln_output");
    
    // Self-attention
    forward_attention(layer_idx, cache_.norm_output.get(), 
                     cache_.attention_output.get(), use_cache);
    
    // Collect statistics or apply quantization
    collect_calibration_statistics(cache_.attention_output.get(), prefix + "attention_output");
    apply_activation_quantization(cache_.attention_output.get(), prefix + "attention_output");
    
    // Residual connection
    kernels::residual_add_kernel<float><<<grid, block>>>(
        cache_.attention_output->data<float>(),
        hidden_states->data<float>(),
        hidden_states->data<float>(),
        batch_size_,
        seq_length_,
        config_.hidden_size);
    
    // Apply pre-FFN layer norm
    apply_layer_norm(hidden_states,
                    weights_->get_weight(prefix + "ffn_ln_weight"),
                    weights_->get_weight(prefix + "ffn_ln_bias"),
                    cache_.norm_output.get());
    
    // Collect statistics or apply quantization
    collect_calibration_statistics(cache_.norm_output.get(), prefix + "ffn_ln_output");
    apply_activation_quantization(cache_.norm_output.get(), prefix + "ffn_ln_output");
    
    // Feed-forward network
    forward_mlp(layer_idx, cache_.norm_output.get(), 
                cache_.mlp_output.get());
    
    // Collect statistics or apply quantization
    collect_calibration_statistics(cache_.mlp_output.get(), prefix + "ffn_output");
    apply_activation_quantization(cache_.mlp_output.get(), prefix + "ffn_output");
    
    // Residual connection
    kernels::residual_add_kernel<float><<<grid, block>>>(
        cache_.mlp_output->data<float>(),
        hidden_states->data<float>(),
        hidden_states->data<float>(),
        batch_size_,
        seq_length_,
        config_.hidden_size);
}

void GPTModel::forward_attention(size_t layer_idx,
                               hal::Tensor* hidden_states,
                               hal::Tensor* attention_output,
                               bool use_cache) {
    std::string prefix = "layer_" + std::to_string(layer_idx) + "_";
    
    // Transform hidden states to Q, K, V
    kernels::qkv_transform_kernel<float><<<grid, block>>>(
        hidden_states->data<float>(),
        weights_->get_weight(prefix + "q_weight")->data<float>(),
        weights_->get_weight(prefix + "k_weight")->data<float>(),
        weights_->get_weight(prefix + "v_weight")->data<float>(),
        weights_->get_weight(prefix + "q_bias")->data<float>(),
        weights_->get_weight(prefix + "k_bias")->data<float>(),
        weights_->get_weight(prefix + "v_bias")->data<float>(),
        cache_.query->data<float>(),
        cache_.key->data<float>(),
        cache_.value->data<float>(),
        batch_size_,
        seq_length_,
        config_.hidden_size,
        config_.num_attention_heads,
        config_.hidden_size / config_.num_attention_heads);
    
    // Apply rotary position embeddings if enabled
    if (config_.use_rotary_embedding) {
        apply_rotary_embedding(cache_.query.get(), cache_.key.get());
    }
    
    // Update KV cache if using caching
    if (use_cache) {
        update_kv_cache(layer_idx, hidden_states->shape()[1] - 1);
    }
    
    // Compute attention scores and apply attention
    if (config_.use_flash_attention) {
        kernels::flash_attention_kernel<float><<<grid, block>>>(
            cache_.query->data<float>(),
            cache_.key->data<float>(),
            cache_.value->data<float>(),
            cache_.attention_mask->data<float>(),
            attention_output->data<float>(),
            batch_size_,
            config_.num_attention_heads,
            seq_length_,
            config_.hidden_size / config_.num_attention_heads);
    } else {
        compute_attention_scores(cache_.query.get(), 
                               cache_.key.get(),
                               cache_.attention_mask.get());
        apply_attention(cache_.attention_scores.get(),
                       cache_.value.get(),
                       attention_output);
    }
}

void GPTModel::forward_mlp(size_t layer_idx,
                          hal::Tensor* hidden_states,
                          hal::Tensor* mlp_output) {
    std::string prefix = "layer_" + std::to_string(layer_idx) + "_";
    
    // Launch FFN kernel
    kernels::ffn_kernel<float><<<grid, block>>>(
        hidden_states->data<float>(),
        weights_->get_weight(prefix + "ffn_inter_weight")->data<float>(),
        weights_->get_weight(prefix + "ffn_inter_bias")->data<float>(),
        weights_->get_weight(prefix + "ffn_out_weight")->data<float>(),
        weights_->get_weight(prefix + "ffn_out_bias")->data<float>(),
        mlp_output->data<float>(),
        batch_size_,
        seq_length_,
        config_.hidden_size,
        config_.intermediate_size);
}

void GPTModel::apply_layer_norm(hal::Tensor* input,
                              hal::Tensor* weight,
                              hal::Tensor* bias,
                              hal::Tensor* output) {
    dim3 grid(1);  // One block per sequence
    dim3 block(256);  // Threads per block
    
    kernels::layer_norm_kernel<float><<<grid, block>>>(
        input->data<float>(),
        weight->data<float>(),
        bias->data<float>(),
        output->data<float>(),
        1,  // batch_size
        config_.hidden_size);
}

void GPTModel::compute_logits(hal::Tensor* hidden_states,
                            hal::Tensor* logits) {
    size_t vocab_chunks = (config_.vocab_size + 255) / 256;
    dim3 grid(vocab_chunks, hidden_states->shape()[1], 1);
    dim3 block(256);
    
    kernels::compute_logits_kernel<float><<<grid, block>>>(
        hidden_states->data<float>(),
        weights_->get_weight("lm_head_weight")->data<float>(),
        weights_->get_weight("lm_head_bias")->data<float>(),
        logits->data<float>(),
        1,  // batch_size
        hidden_states->shape()[1],  // seq_length
        config_.hidden_size,
        config_.vocab_size);
}

// Execution engine related methods
void GPTModel::execute_compute_graph() {
    if (!compute_graph_.is_optimized) {
        optimize_compute_graph();
    }

    // Create execution schedule
    std::vector<ComputeGraphNode*> schedule = create_execution_schedule();
    
    // Allocate execution buffers
    allocate_execution_buffers();
    
    // Execute nodes in schedule
    for (auto* node : schedule) {
        if (node->is_fused) continue;  // Skip fused nodes
        
        // Set up kernel parameters
        KernelParams params;
        params.batch_size = batch_size_;
        params.seq_length = seq_length_;
        params.hidden_size = config_.hidden_size;
        params.num_heads = config_.num_attention_heads;
        params.head_dim = config_.hidden_size / config_.num_attention_heads;
        
        // Launch kernel based on node type
        execute_node(node, params);
    }
    
    // Synchronize device
    device_->synchronize();
}

std::vector<ComputeGraphNode*> GPTModel::create_execution_schedule() {
    std::vector<ComputeGraphNode*> schedule;
    std::unordered_set<ComputeGraphNode*> visited;
    
    // Perform topological sort with priority
    std::function<void(ComputeGraphNode*)> schedule_node = 
        [&](ComputeGraphNode* node) {
            if (visited.count(node)) return;
            visited.insert(node);
            
            // Schedule inputs first
            for (auto* input : node->inputs) {
                schedule_node(input);
            }
            
            // Add node to schedule if not fused
            if (!node->is_fused) {
                schedule.push_back(node);
            }
        };
    
    // Start from output nodes
    for (const auto& node : compute_graph_.nodes) {
        if (node->outputs.empty()) {
            schedule_node(node.get());
        }
    }
    
    return schedule;
}

void GPTModel::allocate_execution_buffers() {
    // Calculate total memory requirements
    size_t total_memory = 0;
    for (const auto& node : compute_graph_.nodes) {
        if (node->is_fused) continue;
        
        size_t node_memory = calculate_node_memory(node.get());
        total_memory = std::max(total_memory, node->memory_offset + node_memory);
    }
    
    // Allocate workspace buffer
    if (cache_.temp_buffer) {
        if (cache_.temp_buffer->size_in_bytes() < total_memory) {
            cache_.temp_buffer.reset();
        }
    }
    
    if (!cache_.temp_buffer) {
        cache_.temp_buffer = std::make_unique<hal::Tensor>(
            std::vector<int64_t>{static_cast<int64_t>(total_memory)},
            config_.use_fp16 ? hal::DataType::FLOAT16 : hal::DataType::FLOAT32,
            device_);
    }
}

size_t GPTModel::calculate_node_memory(const ComputeGraphNode* node) {
    size_t memory_size = 0;
    
    switch (node->op_type) {
        case ComputeGraphNode::OpType::EMBEDDING:
            memory_size = batch_size_ * seq_length_ * config_.hidden_size;
            break;
            
        case ComputeGraphNode::OpType::ATTENTION: {
            // Memory for Q, K, V projections
            size_t qkv_size = batch_size_ * seq_length_ * config_.hidden_size;
            // Memory for attention scores
            size_t score_size = batch_size_ * config_.num_attention_heads * 
                              seq_length_ * seq_length_;
            memory_size = 3 * qkv_size + score_size;
            break;
        }
        
        case ComputeGraphNode::OpType::MLP: {
            // Memory for intermediate activations
            size_t hidden_size = batch_size_ * seq_length_ * config_.hidden_size;
            size_t intermediate_size = batch_size_ * seq_length_ * config_.intermediate_size;
            memory_size = hidden_size + intermediate_size;
            break;
        }
        
        case ComputeGraphNode::OpType::LAYER_NORM:
            memory_size = batch_size_ * seq_length_ * config_.hidden_size;
            break;
            
        case ComputeGraphNode::OpType::RESIDUAL:
            memory_size = batch_size_ * seq_length_ * config_.hidden_size;
            break;
            
        case ComputeGraphNode::OpType::LOGITS:
            memory_size = batch_size_ * seq_length_ * config_.vocab_size;
            break;
    }
    
    // Account for FP16 vs FP32
    memory_size *= config_.use_fp16 ? 2 : 4;
    return memory_size;
}

void GPTModel::execute_node(ComputeGraphNode* node, const KernelParams& params) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Original execute_node implementation
    void* workspace = static_cast<char*>(cache_.temp_buffer->data()) + 
                     node->memory_offset;
    
    dim3 grid, block;
    size_t shared_memory = 0;
    
    switch (node->op_type) {
        case ComputeGraphNode::OpType::ATTENTION:
            configure_attention_kernel(node, params, grid, block, shared_memory);
            break;
        case ComputeGraphNode::OpType::MLP:
            configure_mlp_kernel(node, params, grid, block, shared_memory);
            break;
        case ComputeGraphNode::OpType::LAYER_NORM:
            configure_layer_norm_kernel(node, params, grid, block, shared_memory);
            break;
        default:
            configure_default_kernel(node, params, grid, block, shared_memory);
            break;
    }
    
    hal::Kernel::LaunchConfig config;
    config.grid_dim = {grid.x, grid.y, grid.z};
    config.block_dim = {block.x, block.y, block.z};
    config.shared_memory_bytes = shared_memory;
    
    hal::Kernel* kernel = get_kernel_for_node(node);
    if (!kernel) {
        throw std::runtime_error("No kernel available for node: " + node->kernel_type);
    }
    
    set_kernel_arguments(node, kernel, workspace, params);
    kernel->launch(config);
    
    // Performance monitoring
    auto end_time = std::chrono::high_resolution_clock::now();
    auto execution_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    
    std::string kernel_name;
    switch (node->op_type) {
        case ComputeGraphNode::OpType::ATTENTION:
            kernel_name = "attention";
            break;
        case ComputeGraphNode::OpType::MLP:
            kernel_name = "ffn";
            break;
        default:
            kernel_name = "other";
            break;
    }
    
    update_performance_stats(kernel_name, execution_time);
    track_memory_usage();
    monitor_hardware_utilization();
}

void GPTModel::configure_attention_kernel(
    ComputeGraphNode* node,
    const KernelParams& params,
    dim3& grid,
    dim3& block,
    size_t& shared_memory) {
    
    if (node->kernel_type == "flash_attention") {
        // Flash Attention configuration
        block = dim3(256);
        grid = dim3(
            (params.seq_length + 16 - 1) / 16,
            params.num_heads,
            params.batch_size
        );
        shared_memory = 48 * 1024;  // 48 KB shared memory
    } else {
        // Standard attention configuration
        block = dim3(256);
        grid = dim3(
            (params.seq_length + block.x - 1) / block.x,
            params.num_heads,
            params.batch_size
        );
        shared_memory = 0;
    }
}

void GPTModel::configure_mlp_kernel(
    ComputeGraphNode* node,
    const KernelParams& params,
    dim3& grid,
    dim3& block,
    size_t& shared_memory) {
    
    block = dim3(256);
    grid = dim3(
        (params.hidden_size + block.x - 1) / block.x,
        params.seq_length,
        params.batch_size
    );
    shared_memory = 0;
}

void GPTModel::configure_layer_norm_kernel(
    ComputeGraphNode* node,
    const KernelParams& params,
    dim3& grid,
    dim3& block,
    size_t& shared_memory) {
    
    block = dim3(node->block_size);
    grid = dim3(params.batch_size * params.seq_length);
    shared_memory = block.x * sizeof(float) * 2;  // For mean and variance
}

void GPTModel::configure_default_kernel(
    ComputeGraphNode* node,
    const KernelParams& params,
    dim3& grid,
    dim3& block,
    size_t& shared_memory) {
    
    block = dim3(256);
    grid = dim3(
        (params.hidden_size + block.x - 1) / block.x,
        params.seq_length,
        params.batch_size
    );
    shared_memory = 0;
}

hal::Kernel* GPTModel::get_kernel_for_node(ComputeGraphNode* node) {
    switch (node->op_type) {
        case ComputeGraphNode::OpType::ATTENTION:
            return kernels_.attention_kernel;
        case ComputeGraphNode::OpType::MLP:
            return kernels_.ffn_kernel;
        case ComputeGraphNode::OpType::LAYER_NORM:
            return kernels_.layernorm_kernel;
        case ComputeGraphNode::OpType::LOGITS:
            return kernels_.sampling_kernel;
        default:
            return nullptr;
    }
}

void GPTModel::set_kernel_arguments(
    ComputeGraphNode* node,
    hal::Kernel* kernel,
    void* workspace,
    const KernelParams& params) {
    
    // Common parameters
    kernel->set_arg(0, sizeof(void*), &workspace);
    kernel->set_arg(1, sizeof(int), &params.batch_size);
    kernel->set_arg(2, sizeof(int), &params.seq_length);
    kernel->set_arg(3, sizeof(int), &params.hidden_size);
    
    // Node-specific parameters
    switch (node->op_type) {
        case ComputeGraphNode::OpType::ATTENTION: {
            kernel->set_arg(4, sizeof(int), &params.num_heads);
            kernel->set_arg(5, sizeof(int), &params.head_dim);
            // Set attention mask if available
            void* mask_ptr = cache_.attention_mask ? 
                           cache_.attention_mask->data() : nullptr;
            kernel->set_arg(6, sizeof(void*), &mask_ptr);
            break;
        }
        
        case ComputeGraphNode::OpType::MLP: {
            int intermediate_size = config_.intermediate_size;
            kernel->set_arg(4, sizeof(int), &intermediate_size);
            break;
        }
        
        case ComputeGraphNode::OpType::LAYER_NORM: {
            float epsilon = 1e-5f;
            kernel->set_arg(4, sizeof(float), &epsilon);
            break;
        }
        
        default:
            break;
    }
}

// Compute graph optimization methods
struct ComputeGraphNode {
    enum class OpType {
        EMBEDDING,
        ATTENTION,
        MLP,
        LAYER_NORM,
        RESIDUAL,
        LOGITS
    };
    
    OpType op_type;
    std::vector<ComputeGraphNode*> inputs;
    std::vector<ComputeGraphNode*> outputs;
    hal::Tensor* output_tensor;
    bool is_cached;
    bool is_fused;
    std::string kernel_type;
    size_t block_size;
    size_t memory_offset;
};

struct ComputeGraph {
    std::vector<std::unique_ptr<ComputeGraphNode>> nodes;
    std::unordered_map<std::string, ComputeGraphNode*> node_map;
    bool is_optimized;
};

void GPTModel::optimize_compute_graph() {
    // Skip if already optimized
    if (compute_graph_.is_optimized) {
        return;
    }

    // Create nodes for each operation
    auto* embedding_node = create_compute_node("embedding", ComputeGraphNode::OpType::EMBEDDING);
    
    std::vector<ComputeGraphNode*> layer_nodes;
    for (size_t i = 0; i < config_.num_layers; ++i) {
        // Create attention nodes
        auto* attn_ln_node = create_compute_node(
            "layer_" + std::to_string(i) + "_attn_ln",
            ComputeGraphNode::OpType::LAYER_NORM);
        auto* attn_node = create_compute_node(
            "layer_" + std::to_string(i) + "_attention",
            ComputeGraphNode::OpType::ATTENTION);
        auto* attn_residual_node = create_compute_node(
            "layer_" + std::to_string(i) + "_attn_residual",
            ComputeGraphNode::OpType::RESIDUAL);
            
        // Create FFN nodes
        auto* ffn_ln_node = create_compute_node(
            "layer_" + std::to_string(i) + "_ffn_ln",
            ComputeGraphNode::OpType::LAYER_NORM);
        auto* ffn_node = create_compute_node(
            "layer_" + std::to_string(i) + "_ffn",
            ComputeGraphNode::OpType::MLP);
        auto* ffn_residual_node = create_compute_node(
            "layer_" + std::to_string(i) + "_ffn_residual",
            ComputeGraphNode::OpType::RESIDUAL);
            
        // Connect layer nodes
        attn_ln_node->inputs = {i == 0 ? embedding_node : layer_nodes.back()};
        attn_node->inputs = {attn_ln_node};
        attn_residual_node->inputs = {attn_node, attn_ln_node};
        
        ffn_ln_node->inputs = {attn_residual_node};
        ffn_node->inputs = {ffn_ln_node};
        ffn_residual_node->inputs = {ffn_node, ffn_ln_node};
        
        layer_nodes.push_back(ffn_residual_node);
    }
    
    // Create output nodes
    auto* final_ln_node = create_compute_node("final_ln", ComputeGraphNode::OpType::LAYER_NORM);
    auto* logits_node = create_compute_node("logits", ComputeGraphNode::OpType::LOGITS);
    
    final_ln_node->inputs = {layer_nodes.back()};
    logits_node->inputs = {final_ln_node};
    
    // Apply optimizations
    
    // 1. Operator fusion
    fuse_operators();
    
    // 2. Memory planning
    plan_memory_allocation();
    
    // 3. Kernel selection and configuration
    select_optimal_kernels();
    
    compute_graph_.is_optimized = true;
}

void GPTModel::fuse_operators() {
    // Identify fusion opportunities
    for (auto& node : compute_graph_.nodes) {
        if (node->op_type == ComputeGraphNode::OpType::LAYER_NORM) {
            // Try to fuse LayerNorm with subsequent attention or FFN
            if (!node->outputs.empty() && 
                (node->outputs[0]->op_type == ComputeGraphNode::OpType::ATTENTION ||
                 node->outputs[0]->op_type == ComputeGraphNode::OpType::MLP)) {
                node->is_fused = true;
                node->outputs[0]->is_fused = true;
            }
        }
        else if (node->op_type == ComputeGraphNode::OpType::ATTENTION) {
            // Try to fuse attention with residual connection
            if (!node->outputs.empty() && 
                node->outputs[0]->op_type == ComputeGraphNode::OpType::RESIDUAL) {
                node->is_fused = true;
                node->outputs[0]->is_fused = true;
            }
        }
    }
}

void GPTModel::plan_memory_allocation() {
    // Track tensor lifetimes
    std::unordered_map<ComputeGraphNode*, size_t> node_start_times;
    std::unordered_map<ComputeGraphNode*, size_t> node_end_times;
    size_t current_time = 0;
    
    // Perform topological sort
    std::vector<ComputeGraphNode*> sorted_nodes;
    std::unordered_set<ComputeGraphNode*> visited;
    
    std::function<void(ComputeGraphNode*)> topo_sort = 
        [&](ComputeGraphNode* node) {
            if (visited.count(node)) return;
            visited.insert(node);
            
            for (auto* input : node->inputs) {
                topo_sort(input);
            }
            
            node_start_times[node] = current_time++;
            sorted_nodes.push_back(node);
            
            // Estimate tensor lifetime
            size_t max_output_time = current_time;
            for (auto* output : node->outputs) {
                max_output_time = std::max(max_output_time, 
                                         node_start_times[output] + 1);
            }
            node_end_times[node] = max_output_time;
        };
    
    // Start from output node
    topo_sort(compute_graph_.nodes.back().get());
    
    // Allocate memory regions
    size_t total_memory = 0;
    for (auto* node : sorted_nodes) {
        // Skip fused nodes as they share memory with their fusion partners
        if (node->is_fused) continue;
        
        // Calculate required memory
        size_t memory_size = 0;
        switch (node->op_type) {
            case ComputeGraphNode::OpType::EMBEDDING:
                memory_size = batch_size_ * seq_length_ * config_.hidden_size * 
                            (config_.use_fp16 ? 2 : 4);
                break;
            case ComputeGraphNode::OpType::ATTENTION:
                memory_size = batch_size_ * config_.num_attention_heads * 
                            seq_length_ * (seq_length_ / 32 + 1) * 
                            (config_.use_fp16 ? 2 : 4);
                break;
            // Add cases for other op types
            default:
                break;
        }
        
        node->memory_offset = total_memory;
        total_memory += memory_size;
    }
}

void GPTModel::select_optimal_kernels() {
    // Configure kernel parameters based on hardware capabilities
    const auto* cuda_device = dynamic_cast<hal::CUDADevice*>(device_);
    if (!cuda_device) return;
    
    // Get device properties
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, cuda_device->device_id()));
    
    // Select optimal kernel configurations
    for (auto& node : compute_graph_.nodes) {
        switch (node->op_type) {
            case ComputeGraphNode::OpType::ATTENTION:
                if (config_.use_flash_attention && 
                    props.sharedMemPerBlock >= 48 * 1024) {
                    node->kernel_type = "flash_attention";
                } else {
                    node->kernel_type = "standard_attention";
                }
                break;
            case ComputeGraphNode::OpType::LAYER_NORM:
                node->kernel_type = "layer_norm";
                // Configure block size based on hidden size
                node->block_size = std::min(256, 
                    next_power_of_2(config_.hidden_size));
                break;
            // Add cases for other op types
            default:
                break;
        }
    }
}

// Add new methods for FP16 support
void GPTModel::convert_to_fp16() {
    if (config_.use_fp16) {
        return;  // Already in FP16
    }
    
    // Update config
    config_.use_fp16 = true;
    
    // Convert all weights to FP16
    weights_->convert_to_fp16();
    
    // Reinitialize compute kernels for FP16
    to_device(device_);  // This will recreate kernels with FP16
    
    // Convert compute cache tensors
    convert_cache_to_fp16();
}

void GPTModel::convert_to_fp32() {
    if (!config_.use_fp16) {
        return;  // Already in FP32
    }
    
    // Update config
    config_.use_fp16 = false;
    
    // Convert all weights to FP32
    weights_->convert_to_fp32();
    
    // Reinitialize compute kernels for FP32
    to_device(device_);  // This will recreate kernels with FP32
    
    // Convert compute cache tensors
    convert_cache_to_fp32();
}

void GPTModel::convert_cache_to_fp16() {
    // Helper function to convert a tensor to FP16
    auto convert_tensor = [this](std::unique_ptr<hal::Tensor>& tensor) {
        if (tensor) {
            auto shape = tensor->shape();
            auto new_tensor = std::make_unique<hal::Tensor>(
                shape, hal::DataType::FLOAT16, device_);
            device_->convert_precision(tensor.get(), new_tensor.get());
            tensor = std::move(new_tensor);
        }
    };
    
    // Convert attention-related tensors
    convert_tensor(cache_.attention_mask);
    convert_tensor(cache_.query);
    convert_tensor(cache_.key);
    convert_tensor(cache_.value);
    convert_tensor(cache_.attention_scores);
    
    // Convert hidden state tensors
    convert_tensor(cache_.hidden_states);
    convert_tensor(cache_.attention_output);
    convert_tensor(cache_.mlp_output);
    convert_tensor(cache_.norm_output);
    convert_tensor(cache_.logits);
    
    // Convert KV cache tensors
    for (auto& tensor : cache_.key_cache) {
        convert_tensor(tensor);
    }
    for (auto& tensor : cache_.value_cache) {
        convert_tensor(tensor);
    }
}

void GPTModel::convert_cache_to_fp32() {
    // Helper function to convert a tensor to FP32
    auto convert_tensor = [this](std::unique_ptr<hal::Tensor>& tensor) {
        if (tensor) {
            auto shape = tensor->shape();
            auto new_tensor = std::make_unique<hal::Tensor>(
                shape, hal::DataType::FLOAT32, device_);
            device_->convert_precision(tensor.get(), new_tensor.get());
            tensor = std::move(new_tensor);
        }
    };
    
    // Convert attention-related tensors
    convert_tensor(cache_.attention_mask);
    convert_tensor(cache_.query);
    convert_tensor(cache_.key);
    convert_tensor(cache_.value);
    convert_tensor(cache_.attention_scores);
    
    // Convert hidden state tensors
    convert_tensor(cache_.hidden_states);
    convert_tensor(cache_.attention_output);
    convert_tensor(cache_.mlp_output);
    convert_tensor(cache_.norm_output);
    convert_tensor(cache_.logits);
    
    // Convert KV cache tensors
    for (auto& tensor : cache_.key_cache) {
        convert_tensor(tensor);
    }
    for (auto& tensor : cache_.value_cache) {
        convert_tensor(tensor);
    }
}

// Performance monitoring implementation
void GPTModel::update_performance_stats(
    const std::string& kernel_name,
    std::chrono::microseconds execution_time) {
    
    double time_ms = execution_time.count() / 1000.0;
    
    if (kernel_name == "attention") {
        perf_stats_.avg_attention_time_ms = 
            0.9 * perf_stats_.avg_attention_time_ms + 0.1 * time_ms;
    } else if (kernel_name == "ffn") {
        perf_stats_.avg_ffn_time_ms = 
            0.9 * perf_stats_.avg_ffn_time_ms + 0.1 * time_ms;
    }
    
    perf_stats_.avg_total_time_ms = 
        perf_stats_.avg_attention_time_ms + perf_stats_.avg_ffn_time_ms;
}

void GPTModel::track_memory_usage() {
    size_t current_usage = get_cache_size();
    perf_stats_.current_memory_usage = current_usage;
    perf_stats_.peak_memory_usage = 
        std::max(perf_stats_.peak_memory_usage, current_usage);
    
    // Monitor device memory
    if (auto* cuda_device = dynamic_cast<hal::CUDADevice*>(device_)) {
        size_t free_memory, total_memory;
        CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
        size_t used_memory = total_memory - free_memory;
        perf_stats_.current_memory_usage = used_memory;
        perf_stats_.peak_memory_usage = 
            std::max(perf_stats_.peak_memory_usage, used_memory);
    }
}

void GPTModel::calculate_throughput(const BatchMetrics& metrics) {
    auto now = std::chrono::high_resolution_clock::now();
    auto time_since_last_update = 
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_stats_update_);
    
    if (time_since_last_update.count() > 1000) {  // Update every second
        total_processed_tokens_ += metrics.total_tokens;
        total_processed_sequences_ += metrics.num_sequences;
        
        double seconds = time_since_last_update.count() / 1000.0;
        perf_stats_.tokens_per_second = 
            total_processed_tokens_ / seconds;
        perf_stats_.sequences_per_second = 
            total_processed_sequences_ / seconds;
        
        // Reset counters
        total_processed_tokens_ = 0;
        total_processed_sequences_ = 0;
        last_stats_update_ = now;
    }
}

void GPTModel::monitor_hardware_utilization() {
    if (auto* cuda_device = dynamic_cast<hal::CUDADevice*>(device_)) {
        // Get GPU utilization
        nvmlDevice_t nvml_device;
        nvmlUtilization_t utilization;
        if (nvmlDeviceGetHandleByIndex(cuda_device->device_id(), &nvml_device) == NVML_SUCCESS &&
            nvmlDeviceGetUtilizationRates(nvml_device, &utilization) == NVML_SUCCESS) {
            perf_stats_.gpu_utilization = utilization.gpu / 100.0f;
            perf_stats_.memory_bandwidth_utilization = utilization.memory / 100.0f;
        }
    }
}

void GPTModel::update_cache_statistics(bool cache_hit) {
    if (cache_hit) {
        perf_stats_.cache_hits++;
    } else {
        perf_stats_.cache_misses++;
    }
    
    size_t total_accesses = perf_stats_.cache_hits + perf_stats_.cache_misses;
    if (total_accesses > 0) {
        perf_stats_.cache_hit_rate = 
            static_cast<float>(perf_stats_.cache_hits) / total_accesses;
    }
}

// Dynamic batching implementation
void GPTModel::set_batching_config(const BatchingConfig& config) {
    std::lock_guard<std::mutex> lock(batch_mutex_);
    batch_config_ = config;
}

bool GPTModel::can_add_to_batch(
    const std::vector<int32_t>& sequence,
    const BatchMetrics& current_metrics) {
    
    if (current_metrics.num_sequences >= batch_config_.max_batch_size) {
        return false;
    }
    
    size_t new_max_length = std::max(
        current_metrics.max_sequence_length,
        sequence.size());
        
    // Check if adding this sequence would exceed padding threshold
    size_t total_tokens = current_metrics.total_tokens + sequence.size();
    size_t max_tokens = (current_metrics.num_sequences + 1) * new_max_length;
    float new_padding_ratio = 1.0f - (static_cast<float>(total_tokens) / max_tokens);
    
    return new_padding_ratio <= batch_config_.padding_threshold;
}

void GPTModel::optimize_batch_size(const BatchMetrics& metrics) {
    if (!batch_config_.adaptive_batch_size) {
        return;
    }
    
    // Store historical metrics
    historical_metrics_.push_back(metrics);
    if (historical_metrics_.size() > 100) {
        historical_metrics_.pop_front();
    }
    
    // Analyze recent performance
    if (historical_metrics_.size() >= 10) {
        float avg_efficiency = 0.0f;
        for (const auto& m : historical_metrics_) {
            avg_efficiency += m.efficiency;
        }
        avg_efficiency /= historical_metrics_.size();
        
        // Adjust target batch size based on efficiency
        if (avg_efficiency > 0.9f && 
            batch_config_.target_batch_size < batch_config_.max_batch_size) {
            batch_config_.target_batch_size = 
                std::min(batch_config_.target_batch_size + 1,
                        batch_config_.max_batch_size);
        } else if (avg_efficiency < 0.7f && 
                   batch_config_.target_batch_size > batch_config_.min_batch_size) {
            batch_config_.target_batch_size = 
                std::max(batch_config_.target_batch_size - 1,
                        batch_config_.min_batch_size);
        }
    }
}

BatchMetrics GPTModel::calculate_batch_metrics(
    const std::vector<std::vector<int32_t>>& sequences) {
    
    BatchMetrics metrics;
    metrics.num_sequences = sequences.size();
    metrics.max_sequence_length = 0;
    metrics.total_tokens = 0;
    
    for (const auto& seq : sequences) {
        metrics.max_sequence_length = 
            std::max(metrics.max_sequence_length, seq.size());
        metrics.total_tokens += seq.size();
    }
    
    size_t max_tokens = metrics.num_sequences * metrics.max_sequence_length;
    metrics.padding_ratio = 1.0f - (static_cast<float>(metrics.total_tokens) / max_tokens);
    
    // Calculate efficiency based on padding ratio and batch utilization
    float batch_utilization = 
        static_cast<float>(metrics.num_sequences) / batch_config_.target_batch_size;
    metrics.efficiency = (1.0f - metrics.padding_ratio) * batch_utilization;
    
    return metrics;
}

void GPTModel::adjust_batch_parameters(const BatchMetrics& metrics) {
    // Adjust max sequence length ratio based on observed lengths
    if (metrics.max_sequence_length > 0) {
        float current_ratio = batch_config_.max_sequence_length_ratio;
        float observed_ratio = static_cast<float>(metrics.max_sequence_length) / 
                             config_.max_sequence_length;
        
        // Smooth adjustment
        batch_config_.max_sequence_length_ratio = 
            0.9f * current_ratio + 0.1f * observed_ratio;
    }
    
    // Adjust padding threshold based on efficiency
    if (metrics.efficiency < 0.5f && batch_config_.padding_threshold > 0.1f) {
        batch_config_.padding_threshold *= 0.9f;
    } else if (metrics.efficiency > 0.9f && batch_config_.padding_threshold < 0.3f) {
        batch_config_.padding_threshold *= 1.1f;
    }
}

float GPTModel::get_current_batch_efficiency() const {
    if (historical_metrics_.empty()) {
        return 1.0f;
    }
    
    float avg_efficiency = 0.0f;
    size_t count = std::min(historical_metrics_.size(), size_t(10));
    
    auto it = historical_metrics_.end();
    for (size_t i = 0; i < count; ++i) {
        --it;
        avg_efficiency += it->efficiency;
    }
    
    return avg_efficiency / count;
}

void GPTModel::quantize(const QuantizationConfig& config) {
    if (is_quantized_) {
        return;
    }

    // Update model config
    config_.quant_config = config;
    
    // Quantize weights based on configuration
    switch (config.type) {
        case QuantizationType::INT8:
            quantize_weights(QuantizationType::INT8);
            break;
        case QuantizationType::INT4:
            quantize_weights(QuantizationType::INT4);
            break;
        case QuantizationType::FP16:
            convert_to_fp16();
            break;
        default:
            return;
    }
    
    // Set up runtime quantization for activations if using dynamic quantization
    if (config.method == QuantizationMethod::DYNAMIC) {
        quantize_activations(config.type);
    }
    
    is_quantized_ = true;
}

void GPTModel::calibrate(const std::vector<std::vector<int32_t>>& calibration_data) {
    if (calibration_data.empty() || is_calibrated_) {
        return;
    }
    
    // Collect statistics on activations using calibration data
    collect_activation_statistics(calibration_data);
    
    // Calculate quantization parameters based on collected statistics
    for (auto& [op_name, stats] : activation_stats_) {
        float scale = (stats.max_val - stats.min_val) / 255.0f;
        int8_t zero_point = static_cast<int8_t>(-stats.min_val / scale);
        quant_params_[op_name] = {scale, zero_point};
    }
    
    is_calibrated_ = true;
}

void GPTModel::quantize_weights(QuantizationType type) {
    // Helper function to quantize a tensor
    auto quantize_tensor_wrapper = [this, type](hal::Tensor* tensor, bool per_channel) {
        if (!tensor) return;
        
        std::unique_ptr<hal::Tensor> quantized;
        if (type == QuantizationType::INT8) {
            quantized = quantize_tensor_int8(tensor, per_channel);
        } else if (type == QuantizationType::INT4) {
            quantized = quantize_tensor_int4(tensor, per_channel);
        }
        
        if (quantized) {
            tensor = quantized.release();
        }
    };
    
    // Quantize all model weights
    for (size_t i = 0; i < config_.num_layers; ++i) {
        std::string prefix = "layer_" + std::to_string(i) + "_";
        
        // Quantize attention weights
        quantize_tensor_wrapper(weights_->get_weight(prefix + "q_weight"), true);
        quantize_tensor_wrapper(weights_->get_weight(prefix + "k_weight"), true);
        quantize_tensor_wrapper(weights_->get_weight(prefix + "v_weight"), true);
        quantize_tensor_wrapper(weights_->get_weight(prefix + "o_weight"), true);
        
        // Quantize FFN weights
        quantize_tensor_wrapper(weights_->get_weight(prefix + "ffn_inter_weight"), true);
        quantize_tensor_wrapper(weights_->get_weight(prefix + "ffn_out_weight"), true);
    }
}

void GPTModel::quantize_activations(QuantizationType type) {
    // Set up activation quantization parameters for dynamic quantization
    std::vector<std::string> activation_ops = {
        "attention_output",
        "ffn_intermediate",
        "ffn_output"
    };
    
    for (const auto& op : activation_ops) {
        if (std::find(config_.quant_config.excluded_ops.begin(),
                     config_.quant_config.excluded_ops.end(),
                     op) == config_.quant_config.excluded_ops.end()) {
            // Initialize with dummy values, will be updated during runtime
            quant_params_[op] = {1.0f, 0};
        }
    }
}

std::unique_ptr<hal::Tensor> GPTModel::quantize_tensor_int8(
    const hal::Tensor* tensor, bool per_channel) {
    
    if (!tensor) return nullptr;
    
    const auto& shape = tensor->shape();
    auto quantized = std::make_unique<hal::Tensor>(shape, hal::DataType::INT8, device_);
    
    std::vector<float> scales;
    std::vector<int8_t> zero_points;
    calculate_scaling_factors(tensor, scales, zero_points, per_channel);
    
    // Store quantization parameters in tensor metadata
    quantized->set_scales(scales);
    quantized->set_zero_points(zero_points);
    
    // Perform quantization
    const float* src_data = static_cast<const float*>(tensor->data());
    int8_t* dst_data = static_cast<int8_t*>(quantized->data());
    
    size_t num_elements = tensor->size_in_bytes() / sizeof(float);
    size_t channel_size = per_channel ? (num_elements / scales.size()) : num_elements;
    
    for (size_t i = 0; i < num_elements; ++i) {
        size_t scale_idx = per_channel ? (i / channel_size) : 0;
        float scaled_val = src_data[i] / scales[scale_idx];
        dst_data[i] = static_cast<int8_t>(std::round(scaled_val) + zero_points[scale_idx]);
    }
    
    return quantized;
}

std::unique_ptr<hal::Tensor> GPTModel::quantize_tensor_int4(
    const hal::Tensor* tensor, bool per_channel) {
    
    if (!tensor) return nullptr;
    
    const auto& shape = tensor->shape();
    // INT4 values are packed two per byte
    std::vector<int64_t> packed_shape = shape;
    packed_shape.back() = (shape.back() + 1) / 2;
    auto quantized = std::make_unique<hal::Tensor>(packed_shape, hal::DataType::INT4, device_);
    
    std::vector<float> scales;
    std::vector<int8_t> zero_points;
    calculate_scaling_factors(tensor, scales, zero_points, per_channel);
    
    // Store quantization parameters
    quantized->set_scales(scales);
    quantized->set_zero_points(zero_points);
    
    // Perform quantization
    const float* src_data = static_cast<const float*>(tensor->data());
    int8_t* dst_data = static_cast<int8_t*>(quantized->data());
    
    size_t num_elements = tensor->size_in_bytes() / sizeof(float);
    size_t channel_size = per_channel ? (num_elements / scales.size()) : num_elements;
    
    for (size_t i = 0; i < num_elements; i += 2) {
        size_t scale_idx = per_channel ? (i / channel_size) : 0;
        
        // Quantize two values and pack them into one byte
        float scaled_val1 = src_data[i] / scales[scale_idx];
        int8_t quant_val1 = static_cast<int8_t>(std::round(scaled_val1) + zero_points[scale_idx]);
        quant_val1 = std::max(std::min(quant_val1, int8_t(7)), int8_t(-8));
        
        int8_t quant_val2 = 0;
        if (i + 1 < num_elements) {
            float scaled_val2 = src_data[i + 1] / scales[scale_idx];
            quant_val2 = static_cast<int8_t>(std::round(scaled_val2) + zero_points[scale_idx]);
            quant_val2 = std::max(std::min(quant_val2, int8_t(7)), int8_t(-8));
        }
        
        // Pack two 4-bit values into one byte
        dst_data[i/2] = (quant_val1 << 4) | (quant_val2 & 0x0F);
    }
    
    return quantized;
}

void GPTModel::calculate_scaling_factors(
    const hal::Tensor* tensor,
    std::vector<float>& scales,
    std::vector<int8_t>& zero_points,
    bool per_channel) {
    
    if (!tensor) return;
    
    const float* data = static_cast<const float*>(tensor->data());
    size_t num_elements = tensor->size_in_bytes() / sizeof(float);
    
    if (per_channel) {
        // Assume last dimension is the channel dimension
        size_t num_channels = tensor->shape().back();
        size_t channel_size = num_elements / num_channels;
        
        scales.resize(num_channels);
        zero_points.resize(num_channels);
        
        for (size_t c = 0; c < num_channels; ++c) {
            float min_val = std::numeric_limits<float>::max();
            float max_val = std::numeric_limits<float>::lowest();
            
            // Find min/max values for this channel
            for (size_t i = 0; i < channel_size; ++i) {
                float val = data[c * channel_size + i];
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
            
            // Calculate scale and zero point for this channel
            if (config_.quant_config.symmetric) {
                float abs_max = std::max(std::abs(min_val), std::abs(max_val));
                scales[c] = abs_max / 127.0f;
                zero_points[c] = 0;
            } else {
                scales[c] = (max_val - min_val) / 255.0f;
                zero_points[c] = static_cast<int8_t>(-min_val / scales[c]);
            }
        }
    } else {
        // Per-tensor quantization
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        
        // Find global min/max values
        for (size_t i = 0; i < num_elements; ++i) {
            min_val = std::min(min_val, data[i]);
            max_val = std::max(max_val, data[i]);
        }
        
        // Calculate scale and zero point
        scales.resize(1);
        zero_points.resize(1);
        
        if (config_.quant_config.symmetric) {
            float abs_max = std::max(std::abs(min_val), std::abs(max_val));
            scales[0] = abs_max / 127.0f;
            zero_points[0] = 0;
        } else {
            scales[0] = (max_val - min_val) / 255.0f;
            zero_points[0] = static_cast<int8_t>(-min_val / scales[0]);
        }
    }
}

void GPTModel::collect_activation_statistics(
    const std::vector<std::vector<int32_t>>& calibration_data) {
    
    // Process each calibration sample
    size_t num_samples = std::min(
        calibration_data.size(),
        size_t(calibration_data.size() * config_.quant_config.calibration_ratio));
        
    for (size_t i = 0; i < num_samples; ++i) {
        // Forward pass with activation statistics collection
        std::vector<int32_t> input_ids = calibration_data[i];
        
        // Create attention mask and position IDs
        create_attention_mask(input_ids);
        create_position_ids(input_ids);
        
        // Embed input tokens
        embed_inputs(input_ids, cache_.hidden_states.get());
        
        // Forward through transformer layers with statistics collection
        for (size_t layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
            // Collect statistics for attention output
            forward_attention(layer_idx, cache_.hidden_states.get(),
                            cache_.attention_output.get(), false);
            update_activation_ranges(cache_.attention_output.get(), "attention_output");
            
            // Collect statistics for FFN intermediate and output
            forward_mlp(layer_idx, cache_.hidden_states.get(),
                       cache_.mlp_output.get());
            update_activation_ranges(cache_.mlp_output.get(), "ffn_output");
        }
    }
}

void GPTModel::update_activation_ranges(
    const hal::Tensor* activation,
    const std::string& op_name) {
    
    if (!activation || activation_stats_.count(op_name) == 0) {
        activation_stats_[op_name] = ActivationStats();
    }
    
    auto& stats = activation_stats_[op_name];
    const float* data = static_cast<const float*>(activation->data());
    size_t num_elements = activation->size_in_bytes() / sizeof(float);
    
    // Update min/max values
    for (size_t i = 0; i < num_elements; ++i) {
        float val = data[i];
        stats.min_val = std::min(stats.min_val, val);
        stats.max_val = std::max(stats.max_val, val);
        
        // Update running statistics
        stats.count++;
        float delta = val - stats.running_mean;
        stats.running_mean += delta / stats.count;
        float delta2 = val - stats.running_mean;
        stats.running_variance += delta * delta2;
    }
}

hal::Tensor* GPTModel::maybe_quantize_activation(
    hal::Tensor* tensor,
    const std::string& op_name) {
    
    if (!is_quantized_ || !tensor || 
        config_.quant_config.method != QuantizationMethod::DYNAMIC ||
        std::find(config_.quant_config.excluded_ops.begin(),
                 config_.quant_config.excluded_ops.end(),
                 op_name) != config_.quant_config.excluded_ops.end()) {
        return tensor;
    }
    
    // Dynamically quantize the activation
    auto quantized = (config_.quant_config.type == QuantizationType::INT8) ?
                    quantize_tensor_int8(tensor, false) :
                    quantize_tensor_int4(tensor, false);
                    
    if (quantized) {
        return quantized.release();
    }
    
    return tensor;
}

hal::Tensor* GPTModel::maybe_dequantize_activation(
    hal::Tensor* tensor,
    const std::string& op_name) {
    
    if (!is_quantized_ || !tensor || 
        config_.quant_config.method != QuantizationMethod::DYNAMIC ||
        tensor->dtype() == hal::DataType::FLOAT32) {
        return tensor;
    }
    
    // Create output tensor
    auto dequantized = std::make_unique<hal::Tensor>(
        tensor->shape(), hal::DataType::FLOAT32, device_);
        
    // Get quantization parameters
    const auto& scales = tensor->scales();
    const auto& zero_points = tensor->zero_points();
    
    // Dequantize based on data type
    if (tensor->dtype() == hal::DataType::INT8) {
        const int8_t* src_data = static_cast<const int8_t*>(tensor->data());
        float* dst_data = static_cast<float*>(dequantized->data());
        
        size_t num_elements = tensor->size_in_bytes();
        float scale = scales[0];
        int8_t zero_point = zero_points[0];
        
        for (size_t i = 0; i < num_elements; ++i) {
            dst_data[i] = scale * (src_data[i] - zero_point);
        }
    } else if (tensor->dtype() == hal::DataType::INT4) {
        const int8_t* src_data = static_cast<const int8_t*>(tensor->data());
        float* dst_data = static_cast<float*>(dequantized->data());
        
        size_t num_elements = tensor->size_in_bytes() * 2;  // Two values per byte
        float scale = scales[0];
        int8_t zero_point = zero_points[0];
        
        for (size_t i = 0; i < num_elements; i += 2) {
            // Unpack two 4-bit values
            int8_t packed = src_data[i/2];
            int8_t val1 = (packed >> 4) & 0x0F;
            int8_t val2 = packed & 0x0F;
            
            // Dequantize
            dst_data[i] = scale * (val1 - zero_point);
            if (i + 1 < num_elements) {
                dst_data[i + 1] = scale * (val2 - zero_point);
            }
        }
    }
    
    return dequantized.release();
}

void GPTModel::init_quantization_manager() {
    quant_manager_ = std::make_unique<QuantizationManager>(device_);
    
    // Configure quantization parameters
    quant_manager_->set_quantization_type(config_.quant_config.type);
    quant_manager_->set_quantization_method(config_.quant_config.method);
    quant_manager_->set_per_channel(config_.quant_config.per_channel);
    quant_manager_->set_symmetric(config_.quant_config.symmetric);
}

void GPTModel::apply_weight_quantization() {
    if (weights_quantized_) return;
    
    // Quantize each layer's weights
    for (size_t i = 0; i < config_.num_layers; ++i) {
        std::string prefix = "layer_" + std::to_string(i) + "_";
        
        // Quantize attention weights
        auto* q_weight = weights_->get_weight(prefix + "q_weight");
        auto* k_weight = weights_->get_weight(prefix + "k_weight");
        auto* v_weight = weights_->get_weight(prefix + "v_weight");
        auto* o_weight = weights_->get_weight(prefix + "o_weight");
        
        if (q_weight) weights_->set_weight(prefix + "q_weight", quant_manager_->quantize_int8(q_weight, true));
        if (k_weight) weights_->set_weight(prefix + "k_weight", quant_manager_->quantize_int8(k_weight, true));
        if (v_weight) weights_->set_weight(prefix + "v_weight", quant_manager_->quantize_int8(v_weight, true));
        if (o_weight) weights_->set_weight(prefix + "o_weight", quant_manager_->quantize_int8(o_weight, true));
        
        // Quantize FFN weights
        auto* ffn_inter_weight = weights_->get_weight(prefix + "ffn_inter_weight");
        auto* ffn_out_weight = weights_->get_weight(prefix + "ffn_out_weight");
        
        if (ffn_inter_weight) weights_->set_weight(prefix + "ffn_inter_weight", 
            quant_manager_->quantize_int8(ffn_inter_weight, true));
        if (ffn_out_weight) weights_->set_weight(prefix + "ffn_out_weight",
            quant_manager_->quantize_int8(ffn_out_weight, true));
    }
    
    // Quantize token embedding and final layer weights
    auto* token_embedding = weights_->get_weight("token_embedding");
    auto* position_embedding = weights_->get_weight("position_embedding");
    auto* lm_head_weight = weights_->get_weight("lm_head_weight");
    
    if (token_embedding) weights_->set_weight("token_embedding",
        quant_manager_->quantize_int8(token_embedding, false));
    if (position_embedding) weights_->set_weight("position_embedding",
        quant_manager_->quantize_int8(position_embedding, false));
    if (lm_head_weight) weights_->set_weight("lm_head_weight",
        quant_manager_->quantize_int8(lm_head_weight, true));
    
    weights_quantized_ = true;
}

void GPTModel::apply_activation_quantization(hal::Tensor* tensor, const std::string& name) {
    if (!activations_quantized_ || !tensor) return;
    
    // Quantize the activation
    auto quantized = quant_manager_->maybe_quantize(tensor, name);
    if (quantized) {
        // Copy quantized data back to the original tensor
        device_->memcpy_to_device(tensor->data(), quantized->data(), quantized->size_in_bytes());
        tensor->set_scales(quantized->scales());
        tensor->set_zero_points(quantized->zero_points());
    }
}

void GPTModel::collect_calibration_statistics(hal::Tensor* tensor, const std::string& name) {
    if (!tensor || config_.quant_config.method != QuantizationMethod::POST_TRAINING) return;
    
    quant_manager_->calibrate(tensor, name);
}

} // namespace deeppowers 