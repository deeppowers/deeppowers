#pragma once

#include "gpt_weights.hpp"
#include "../model.hpp"
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <deque>
#include <mutex>
#include <unordered_map>

namespace deeppowers {

// GPT model configuration
struct GPTConfig : public ModelConfig {
    // GPT specific configuration
    size_t intermediate_size = 3072;        // FFN intermediate layer size
    size_t max_batch_size = 32;            // Maximum batch size
    size_t max_sequence_length = 2048;      // Maximum sequence length
    bool use_rotary_embedding = true;       // Whether to use rotary position encoding
    bool use_flash_attention = true;        // Whether to use Flash Attention
    bool use_parallel_attention = true;     // Whether to use parallel attention calculation
    float rotary_embedding_base = 10000.0f; // Rotary position encoding base
    
    GPTConfig() {
        type = ModelType::DECODER_ONLY;
    }
};

// Performance monitoring structures
struct PerformanceStats {
    // Execution time statistics
    double avg_attention_time_ms = 0.0;
    double avg_ffn_time_ms = 0.0;
    double avg_total_time_ms = 0.0;
    
    // Memory statistics
    size_t peak_memory_usage = 0;
    size_t current_memory_usage = 0;
    
    // Throughput statistics
    double tokens_per_second = 0.0;
    double sequences_per_second = 0.0;
    
    // Hardware utilization
    float gpu_utilization = 0.0;
    float memory_bandwidth_utilization = 0.0;
    
    // Cache statistics
    size_t cache_hits = 0;
    size_t cache_misses = 0;
    float cache_hit_rate = 0.0;
    
    // Reset statistics
    void reset() {
        avg_attention_time_ms = 0.0;
        avg_ffn_time_ms = 0.0;
        avg_total_time_ms = 0.0;
        peak_memory_usage = 0;
        current_memory_usage = 0;
        tokens_per_second = 0.0;
        sequences_per_second = 0.0;
        gpu_utilization = 0.0;
        memory_bandwidth_utilization = 0.0;
        cache_hits = 0;
        cache_misses = 0;
        cache_hit_rate = 0.0;
    }
};

// Dynamic batching structures
struct BatchingConfig {
    size_t min_batch_size = 1;
    size_t max_batch_size = 32;
    size_t target_batch_size = 16;
    float max_sequence_length_ratio = 1.5f;
    float padding_threshold = 0.2f;
    bool adaptive_batch_size = true;
    std::chrono::milliseconds max_wait_time{100};
};

struct BatchMetrics {
    size_t num_sequences;
    size_t max_sequence_length;
    size_t total_tokens;
    float padding_ratio;
    float efficiency;
    std::chrono::microseconds processing_time;
};

// GPT model class
class GPTModel : public Model {
public:
    GPTModel(const GPTConfig& config, hal::Device* device);
    ~GPTModel() override = default;

    // Model interface implementation
    void load_weights(const std::string& path) override;
    void save_weights(const std::string& path) const override;
    void to_device(hal::Device* device) override;
    void to_host() override;

    // Precision conversion methods
    void convert_to_fp16();
    void convert_to_fp32();

    // GPT specific methods
    std::vector<int32_t> generate(
        const std::vector<int32_t>& input_ids,
        size_t max_length,
        float temperature = 1.0f,
        float top_p = 1.0f,
        float top_k = 0.0f,
        const std::vector<int32_t>& stop_ids = {});
        
    std::vector<std::vector<int32_t>> generate_batch(
        const std::vector<std::vector<int32_t>>& batch_input_ids,
        size_t max_length,
        float temperature = 1.0f,
        float top_p = 1.0f,
        float top_k = 0.0f,
        const std::vector<int32_t>& stop_ids = {});

    // Memory management methods
    void clear_kv_cache();
    void compress_tensors();
    size_t get_cache_size() const;

    // Performance monitoring interface
    const PerformanceStats& get_performance_stats() const { return perf_stats_; }
    void reset_performance_stats() { perf_stats_.reset(); }
    
    // Dynamic batching interface
    void set_batching_config(const BatchingConfig& config);
    const BatchingConfig& get_batching_config() const { return batch_config_; }
    float get_current_batch_efficiency() const;

    // Quantization methods
    void quantize(const QuantizationConfig& config);
    void calibrate(const std::vector<std::vector<int32_t>>& calibration_data);

private:
    // Internal compute methods
    void create_attention_mask(const std::vector<int32_t>& input_ids);
    void create_position_ids(const std::vector<int32_t>& input_ids);
    void apply_rotary_embedding(hal::Tensor* query, hal::Tensor* key);
    void compute_attention_scores(hal::Tensor* query, hal::Tensor* key, hal::Tensor* mask);
    void apply_attention(hal::Tensor* scores, hal::Tensor* value, hal::Tensor* output);
    void compute_ffn(hal::Tensor* input, hal::Tensor* output);
    
    // Sampling methods
    int32_t sample_token(const std::vector<float>& logits,
                        float temperature,
                        float top_p,
                        float top_k);
    
    // Cache management
    void init_kv_cache(size_t batch_size, size_t max_length);
    void update_kv_cache(size_t layer_idx, size_t position);
    
    // Forward propagation methods
    void forward(const std::vector<int32_t>& input_ids,
                hal::Tensor* output,
                bool use_cache = false);
                
    void forward_batch(const std::vector<std::vector<int32_t>>& batch_input_ids,
                      hal::Tensor* output,
                      bool use_cache = false);
                      
    void embed_inputs(const std::vector<int32_t>& input_ids,
                     hal::Tensor* embedded);
                     
    void forward_transformer_layer(size_t layer_idx,
                                 hal::Tensor* hidden_states,
                                 bool use_cache = false);
                                 
    void forward_attention(size_t layer_idx,
                         hal::Tensor* hidden_states,
                         hal::Tensor* attention_output,
                         bool use_cache = false);
                         
    void forward_mlp(size_t layer_idx,
                    hal::Tensor* hidden_states,
                    hal::Tensor* mlp_output);
                    
    void apply_layer_norm(hal::Tensor* input,
                         hal::Tensor* weight,
                         hal::Tensor* bias,
                         hal::Tensor* output);
                         
    void compute_logits(hal::Tensor* hidden_states,
                       hal::Tensor* logits);
    
    // Memory management helper methods
    void resize_cache(size_t batch_size, size_t seq_length);
    void optimize_tensor_memory();
    
    // Precision conversion helpers
    void convert_cache_to_fp16();
    void convert_cache_to_fp32();
    
    // Performance monitoring methods
    void update_performance_stats(const std::string& kernel_name,
                                std::chrono::microseconds execution_time);
    void track_memory_usage();
    void calculate_throughput(const BatchMetrics& metrics);
    void monitor_hardware_utilization();
    void update_cache_statistics(bool cache_hit);
    
    // Dynamic batching methods
    bool can_add_to_batch(const std::vector<int32_t>& sequence,
                         const BatchMetrics& current_metrics);
    void optimize_batch_size(const BatchMetrics& metrics);
    BatchMetrics calculate_batch_metrics(
        const std::vector<std::vector<int32_t>>& sequences);
    void adjust_batch_parameters(const BatchMetrics& metrics);
    
    // Quantization helper methods
    void quantize_weights(QuantizationType type);
    void quantize_activations(QuantizationType type);
    std::unique_ptr<hal::Tensor> quantize_tensor_int8(const hal::Tensor* tensor, bool per_channel = false);
    std::unique_ptr<hal::Tensor> quantize_tensor_int4(const hal::Tensor* tensor, bool per_channel = false);
    void calculate_scaling_factors(const hal::Tensor* tensor, std::vector<float>& scales, 
                                 std::vector<int8_t>& zero_points, bool per_channel = false);
    
    // Calibration helper methods
    void collect_activation_statistics(const std::vector<std::vector<int32_t>>& calibration_data);
    void update_activation_ranges(const hal::Tensor* activation, const std::string& op_name);
    
    // Runtime quantization
    hal::Tensor* maybe_quantize_activation(hal::Tensor* tensor, const std::string& op_name);
    hal::Tensor* maybe_dequantize_activation(hal::Tensor* tensor, const std::string& op_name);
    
    // Quantization manager
    std::unique_ptr<QuantizationManager> quant_manager_;
    
    // Quantization related helper methods
    void init_quantization_manager();
    void apply_weight_quantization();
    void apply_activation_quantization(hal::Tensor* tensor, const std::string& name);
    void collect_calibration_statistics(hal::Tensor* tensor, const std::string& name);
    
    // Quantization status
    bool weights_quantized_ = false;
    bool activations_quantized_ = false;
    
    // Internal state
    GPTConfig config_;
    std::unique_ptr<GPTWeights> weights_;
    
    // Compute cache
    struct ComputeCache {
        std::unique_ptr<hal::Tensor> attention_mask;
        std::unique_ptr<hal::Tensor> position_ids;
        std::vector<std::unique_ptr<hal::Tensor>> key_cache;
        std::vector<std::unique_ptr<hal::Tensor>> value_cache;
        std::unique_ptr<hal::Tensor> temp_buffer;
        
        // Forward propagation cache
        std::unique_ptr<hal::Tensor> hidden_states;      // Hidden states
        std::unique_ptr<hal::Tensor> attention_output;   // Attention output
        std::unique_ptr<hal::Tensor> mlp_output;        // MLP output
        std::unique_ptr<hal::Tensor> norm_output;       // Normalization output
        std::unique_ptr<hal::Tensor> logits;            // Output logits
        
        // Attention calculation cache
        std::unique_ptr<hal::Tensor> query;             // Query vector
        std::unique_ptr<hal::Tensor> key;               // Key vector
        std::unique_ptr<hal::Tensor> value;             // Value vector
        std::unique_ptr<hal::Tensor> attention_scores;  // Attention scores
    };
    ComputeCache cache_;
    
    // Compute kernels
    struct ComputeKernels {
        hal::Kernel* attention_kernel;
        hal::Kernel* ffn_kernel;
        hal::Kernel* layernorm_kernel;
        hal::Kernel* softmax_kernel;
        hal::Kernel* sampling_kernel;
    };
    ComputeKernels kernels_;
    
    // Performance monitoring state
    PerformanceStats perf_stats_;
    std::chrono::high_resolution_clock::time_point last_stats_update_;
    size_t total_processed_tokens_;
    size_t total_processed_sequences_;
    
    // Dynamic batching state
    BatchingConfig batch_config_;
    std::deque<BatchMetrics> historical_metrics_;
    std::mutex batch_mutex_;

    // Additional member variables
    struct ActivationStats {
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        float running_mean = 0.0f;
        float running_variance = 0.0f;
        size_t count = 0;
    };
    
    std::unordered_map<std::string, ActivationStats> activation_stats_;
    std::unordered_map<std::string, std::pair<float, int8_t>> quant_params_;  // scale and zero point
    bool is_quantized_ = false;
    bool is_calibrated_ = false;
};

} // namespace deeppowers 