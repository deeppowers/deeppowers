#pragma once

#include "model.hpp"
#include "../batching/batch.hpp"
#include <memory>
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

namespace deeppowers {

// Execution engine configuration
struct ExecutionEngineConfig {
    size_t num_worker_threads = 2;         // Number of worker threads
    size_t max_active_batches = 4;         // Maximum active batches
    size_t max_sequence_length = 2048;     // Maximum sequence length
    bool enable_tensor_cores = true;       // Whether to enable Tensor Cores
    bool enable_graph_optimization = true;  // Whether to enable graph optimization
    size_t workspace_size_mb = 1024;       // Workspace size (MB)
    std::string cache_dir = "./cache";     // Cache directory
};

// Execution statistics
struct ExecutionStats {
    size_t total_batches_processed = 0;    // Total batches processed
    size_t total_requests_processed = 0;   // Total requests processed
    size_t total_tokens_generated = 0;     // Total tokens generated
    double average_batch_time_ms = 0.0;    // Average batch processing time
    double average_tokens_per_second = 0.0; // Average token generation speed
    size_t peak_memory_usage_mb = 0;       // Peak memory usage
};

// Execution engine class
class ExecutionEngine {
public:
    ExecutionEngine(std::unique_ptr<Model> model,
                   hal::Device* device,
                   const ExecutionEngineConfig& config = ExecutionEngineConfig());
    ~ExecutionEngine();

    // Disable copying and moving
    ExecutionEngine(const ExecutionEngine&) = delete;
    ExecutionEngine& operator=(const ExecutionEngine&) = delete;
    ExecutionEngine(ExecutionEngine&&) = delete;
    ExecutionEngine& operator=(ExecutionEngine&&) = delete;

    // Start and stop
    void start();
    void stop();

    // Batch execution
    void execute_batch(BatchPtr batch);
    void execute_batch_async(BatchPtr batch);
    
    // Status query
    bool is_running() const { return running_; }
    const ExecutionStats& get_stats() const { return stats_; }
    size_t get_num_active_batches() const;
    
    // Configuration access and update
    const ExecutionEngineConfig& config() const { return config_; }
    void update_config(const ExecutionEngineConfig& config);
    
    // Model and device access
    Model* model() const { return model_.get(); }
    hal::Device* device() const { return device_; }

private:
    // Worker thread function
    void worker_thread();
    
    // Internal execution methods
    void process_batch(BatchPtr batch);
    void optimize_compute_graph();
    void manage_memory();
    void update_stats(const BatchPtr& batch, 
                     const std::chrono::microseconds& execution_time);
    
    // Internal state
    std::unique_ptr<Model> model_;
    hal::Device* device_;
    ExecutionEngineConfig config_;
    std::atomic<bool> running_;
    ExecutionStats stats_;
    
    // Batch queue
    std::queue<BatchPtr> batch_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<size_t> num_active_batches_;
    
    // Worker threads
    std::vector<std::thread> worker_threads_;
    
    // Memory management
    struct MemoryManager {
        size_t total_allocated;
        size_t peak_allocated;
        std::mutex mutex;
    };
    MemoryManager memory_manager_;
    
    // Compute graph optimization
    struct ComputeGraph {
        bool is_optimized;
        std::mutex mutex;
        // TODO: Add compute graph related data structures
    };
    ComputeGraph compute_graph_;
};

} // namespace deeppowers 