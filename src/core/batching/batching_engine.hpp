#pragma once

#include "batch.hpp"
#include "../request_queue/request_queue.hpp"
#include <memory>
#include <thread>
#include <atomic>
#include <functional>

namespace deeppowers {

// Batching engine configuration options
struct BatchingEngineConfig {
    size_t max_batch_size = 32;            // Maximum batch size
    size_t min_batch_size = 1;             // Minimum batch size
    float max_padding_ratio = 0.3f;        // Maximum padding ratio
    bool enable_dynamic_batching = true;    // Enable dynamic batching
    std::chrono::milliseconds batch_timeout{10};   // Batch timeout
    size_t max_sequence_length = 2048;     // Maximum sequence length
    size_t num_worker_threads = 2;         // Number of worker threads
};

// Batching engine class
class BatchingEngine {
public:
    explicit BatchingEngine(std::shared_ptr<RequestQueue> request_queue,
                          const BatchingEngineConfig& config = BatchingEngineConfig());
    ~BatchingEngine();

    // Disable copy and move
    BatchingEngine(const BatchingEngine&) = delete;
    BatchingEngine& operator=(const BatchingEngine&) = delete;
    BatchingEngine(BatchingEngine&&) = delete;
    BatchingEngine& operator=(BatchingEngine&&) = delete;

    // Start and stop
    void start();
    void stop();

    // Batch processing callback settings
    using BatchCallback = std::function<void(BatchPtr)>;
    void set_batch_ready_callback(BatchCallback callback);
    void set_batch_complete_callback(BatchCallback callback);

    // Status query
    bool is_running() const { return running_; }
    size_t get_num_processed_batches() const { return num_processed_batches_; }
    float get_average_batch_size() const;
    float get_average_padding_ratio() const;
    
    // Configuration access and update
    const BatchingEngineConfig& config() const { return config_; }
    void update_config(const BatchingEngineConfig& config);

private:
    // Worker thread function
    void worker_thread();
    
    // Batch creation and management
    BatchPtr create_batch(const std::vector<RequestPtr>& requests);
    bool can_add_to_batch(const BatchPtr& batch, const RequestPtr& request) const;
    void process_batch(BatchPtr batch);
    
    // Internal state
    std::shared_ptr<RequestQueue> request_queue_;
    BatchingEngineConfig config_;
    std::atomic<bool> running_;
    std::atomic<size_t> num_processed_batches_;
    
    // Statistics
    struct Stats {
        size_t total_batch_size;
        size_t total_padding_tokens;
        std::mutex mutex;
    };
    Stats stats_;
    
    // Callback functions
    BatchCallback batch_ready_callback_;
    BatchCallback batch_complete_callback_;
    std::mutex callback_mutex_;
    
    // Worker threads
    std::vector<std::thread> worker_threads_;
};

} // namespace deeppowers 