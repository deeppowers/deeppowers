#pragma once

#include "../hal/hal.hpp"
#include "../memory/memory_pool.hpp"
#include <memory>
#include <vector>
#include <queue>
#include <future>
#include <thread>
#include <mutex>

namespace deeppowers {

/**
 * @brief Configuration for inference engine
 */
struct InferenceConfig {
    size_t batch_size = 1;                  // Batch size for inference
    bool enable_tensor_cores = true;        // Whether to use tensor cores
    bool enable_graph_optimization = true;  // Whether to enable graph optimization
    bool enable_kernel_fusion = true;       // Whether to enable kernel fusion
    bool enable_dynamic_batching = true;    // Whether to enable dynamic batching
    size_t num_worker_threads = 4;          // Number of worker threads
    size_t workspace_size = 1024*1024*1024; // Workspace size in bytes (1GB)
};

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
 * @brief Inference engine for model execution
 */
class InferenceEngine {
public:
    /**
     * @brief Constructor
     * @param device Device to run inference on
     * @param config Inference configuration
     */
    explicit InferenceEngine(hal::Device* device,
                           const InferenceConfig& config = InferenceConfig());
    ~InferenceEngine();

    /**
     * @brief Initialize the engine
     */
    void initialize();

    /**
     * @brief Run inference on input tensors
     * @param inputs Input tensors
     * @param outputs Output tensors
     */
    void run(const std::vector<hal::Tensor*>& inputs,
            std::vector<hal::Tensor*>& outputs);

    /**
     * @brief Run inference asynchronously
     * @param inputs Input tensors
     * @param outputs Output tensors
     * @return Future for inference completion
     */
    std::future<void> run_async(const std::vector<hal::Tensor*>& inputs,
                               std::vector<hal::Tensor*>& outputs);

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
    // Internal structures
    struct InferenceTask {
        std::vector<hal::Tensor*> inputs;
        std::vector<hal::Tensor*> outputs;
        std::promise<void> promise;
    };

    struct ComputeGraph {
        struct Node {
            hal::Kernel* kernel;
            std::vector<Node*> inputs;
            std::vector<Node*> outputs;
            hal::Tensor* output_tensor;
            bool is_fused;
        };
        std::vector<std::unique_ptr<Node>> nodes;
    };

    // Internal helper methods
    void worker_thread_func();
    void optimize_graph();
    void fuse_kernels();
    void allocate_workspace();
    void update_statistics(double latency_ms);

    // Memory management
    void* allocate_workspace(size_t size);
    void free_workspace(void* ptr);
    void manage_memory_pressure();

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