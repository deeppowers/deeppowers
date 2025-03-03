#pragma once

#include "../hal/hal.hpp"
#include "../memory/memory_pool.hpp"
#include <memory>
#include <vector>
#include <queue>
#include <future>
#include <thread>
#include <mutex>
#include <string>
#include <function>

namespace deeppowers {

/**
 * Configuration for inference execution
 */
struct InferenceConfig {
    // Generation parameters
    int max_length = 100;
    int min_length = 0;
    float temperature = 1.0f;
    int top_k = 50;
    float top_p = 0.9f;
    float repetition_penalty = 1.0f;
    int num_return_sequences = 1;
    bool do_sample = true;
    bool early_stopping = false;
    
    // Execution parameters
    bool use_cuda = true;
    bool use_mixed_precision = true;
    int batch_size = 1;
    std::string device = "cuda";
    
    // Performance parameters
    bool use_memory_cache = true;
    bool use_kv_cache = true;
    int prefill_chunk_size = 512;
};

/**
 * Result of inference operation
 */
struct InferenceResult {
    std::vector<std::vector<int>> token_ids;  // Generated token IDs
    std::vector<std::vector<float>> logprobs; // Log probabilities
    std::vector<std::string> stop_reasons;    // Reasons for stopping
    float generation_time = 0.0f;            // Time taken for generation in seconds
};

/**
 * Callback function for streaming generation
 */
using StreamingCallback = std::function<bool(const InferenceResult&)>;

/**
 * @brief Statistics for inference engine
 */
struct InferenceStats {
    double avg_latency_ms = 0.0;           // Average inference latency
    double throughput = 0.0;               // Inference throughput
    size_t num_inferences = 0;             // Number of inferences performed
    size_t peak_memory_usage = 0;          // Peak memory usage
    double gpu_utilization = 0.0;          // GPU utilization
    double memory_utilization = 0.0;       // Memory utilization
};

/**
 * @brief Inference engine for text generation
 */
class InferenceEngine {
public:
    /**
     * Create inference engine with model
     * @param model Pretrained model
     */
    explicit InferenceEngine(std::shared_ptr<Model> model);
    
    /**
     * Destructor
     */
    ~InferenceEngine();
    
    /**
     * Generate text from input tokens
     * @param input_ids Input token IDs
     * @param attention_mask Attention mask (optional)
     * @param config Inference configuration
     * @return Generated token IDs and metrics
     */
    InferenceResult generate(
        const std::vector<int>& input_ids,
        const std::vector<int>& attention_mask,
        const InferenceConfig& config = InferenceConfig()
    );
    
    /**
     * Generate text from batch of input tokens
     * @param batch_input_ids Batch of input token IDs
     * @param batch_attention_mask Batch of attention masks (optional)
     * @param config Inference configuration
     * @return Generated token IDs and metrics for each input
     */
    std::vector<InferenceResult> generate_batch(
        const std::vector<std::vector<int>>& batch_input_ids,
        const std::vector<std::vector<int>>& batch_attention_mask,
        const InferenceConfig& config = InferenceConfig()
    );
    
    /**
     * Generate text with streaming output
     * @param input_ids Input token IDs
     * @param callback Callback function for streaming output
     * @param config Inference configuration
     */
    void generate_stream(
        const std::vector<int>& input_ids,
        StreamingCallback callback,
        const InferenceConfig& config = InferenceConfig()
    );
    
    /**
     * Prepare model for inference
     * @param config Inference configuration
     */
    void prepare(const InferenceConfig& config = InferenceConfig());
    
    /**
     * Reset inference state
     */
    void reset();
    
    /**
     * Get current model
     * @return Model pointer
     */
    std::shared_ptr<Model> model() const;
    
    /**
     * Set new model
     * @param model New model
     */
    void set_model(std::shared_ptr<Model> model);

    /**
     * @brief Get inference statistics
     * @return Current inference statistics
     */
    InferenceStats get_stats() const;

    /**
     * @brief Reset statistics
     */
    void reset_stats();

    /**
     * @brief Update configuration
     * @param config New configuration
     */
    void update_config(const InferenceConfig& config);

private:
    std::shared_ptr<Model> model_;
    bool is_prepared_ = false;
    
    // Internal state for KV cache
    std::vector<Tensor> key_cache_;
    std::vector<Tensor> value_cache_;
    
    // Internal methods
    Tensor prepare_inputs(const std::vector<int>& input_ids, 
                        const std::vector<int>& attention_mask);
    
    int sample_token(const Tensor& logits, 
                    const std::vector<int>& prev_tokens,
                    const InferenceConfig& config);
    
    bool should_stop(const std::vector<int>& output_ids, 
                    int current_length,
                    const InferenceConfig& config);
    
    void init_caches(int batch_size, int max_length, int hidden_size);

    // Device and configuration
    hal::Device* device_;
    InferenceConfig config_;
    bool initialized_;

    // Memory management
    std::unique_ptr<MemoryPool> memory_pool_;
    void* workspace_;
    size_t workspace_size_;

    // Compute graph
    std::unique_ptr<ComputeGraph> compute_graph_;
    bool graph_optimized_;

    // Worker threads
    std::vector<std::thread> worker_threads_;
    std::queue<InferenceTask> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    bool should_stop_;

    // Statistics
    InferenceStats stats_;
    mutable std::mutex stats_mutex_;
}; 