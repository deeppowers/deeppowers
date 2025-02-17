#pragma once

#include "../execution/models/gpt_model.hpp"
#include "../request_queue/request.hpp"
#include <memory>
#include <queue>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <condition_variable>

namespace deeppowers {

// Priority levels for scheduling
enum class SchedulePriority {
    LOW,
    NORMAL,
    HIGH,
    CRITICAL
};

// Resource constraints for scheduling
struct ResourceConstraints {
    size_t max_batch_size = 32;           // Maximum batch size
    size_t max_sequence_length = 2048;    // Maximum sequence length
    size_t min_free_memory_mb = 1024;     // Minimum free memory in MB
    float max_gpu_utilization = 0.9f;     // Maximum GPU utilization
    size_t max_active_requests = 100;      // Maximum concurrent requests
};

// Scheduling policy configuration
struct SchedulerConfig {
    bool enable_dynamic_batching = true;   // Enable dynamic batching
    bool enable_priority_queue = true;     // Enable priority-based scheduling
    bool enable_load_balancing = true;     // Enable load balancing
    bool enable_fault_recovery = true;     // Enable fault recovery
    size_t num_worker_threads = 4;         // Number of worker threads
    ResourceConstraints resource_limits;    // Resource constraints
};

// Scheduler statistics
struct SchedulerStats {
    size_t total_requests = 0;             // Total processed requests
    size_t active_requests = 0;            // Currently active requests
    size_t dropped_requests = 0;           // Dropped requests
    double avg_latency_ms = 0.0;           // Average request latency
    double avg_throughput = 0.0;           // Average throughput
    double gpu_utilization = 0.0;          // Current GPU utilization
    size_t memory_usage_mb = 0;            // Current memory usage
};

// Advanced scheduler class
class Scheduler {
public:
    explicit Scheduler(const SchedulerConfig& config);
    ~Scheduler();

    // Initialization and cleanup
    void initialize();
    void finalize();
    
    // Request handling
    void submit_request(RequestPtr request);
    void cancel_request(const std::string& request_id);
    RequestPtr get_request_status(const std::string& request_id);
    
    // Batch management
    void process_batch(const std::vector<RequestPtr>& batch);
    void handle_batch_completion(const std::vector<RequestPtr>& batch);
    
    // Resource management
    bool check_resources(const RequestPtr& request);
    void update_resource_usage();
    void handle_resource_pressure();
    
    // Load balancing
    void balance_load();
    void migrate_requests(size_t source_worker, size_t target_worker);
    
    // Fault recovery
    void handle_worker_failure(size_t worker_id);
    void recover_failed_requests();
    
    // Status and monitoring
    const SchedulerStats& get_stats() const { return stats_; }
    bool is_healthy() const;
    
    // Configuration
    const SchedulerConfig& config() const { return config_; }
    void update_config(const SchedulerConfig& config);

private:
    // Internal helper methods
    void init_worker_threads();
    void worker_thread_func(size_t worker_id);
    void monitor_thread_func();
    void cleanup_thread_func();
    
    // Request queue management
    void enqueue_request(RequestPtr request);
    RequestPtr dequeue_request();
    void requeue_request(RequestPtr request);
    
    // Batch formation
    bool can_batch_request(const RequestPtr& request, const std::vector<RequestPtr>& current_batch);
    std::vector<RequestPtr> form_batch();
    void optimize_batch_size();
    
    // Resource tracking
    struct ResourceUsage {
        size_t memory_used = 0;
        float gpu_util = 0.0f;
        size_t active_batches = 0;
    };
    
    // Worker state
    struct WorkerState {
        bool active = false;
        size_t processed_requests = 0;
        ResourceUsage resources;
        std::queue<RequestPtr> local_queue;
        std::mutex mutex;
    };
    
    // Member variables
    SchedulerConfig config_;
    SchedulerStats stats_;
    bool initialized_ = false;
    bool should_stop_ = false;
    
    // Thread management
    std::vector<std::thread> worker_threads_;
    std::thread monitor_thread_;
    std::thread cleanup_thread_;
    
    // Request management
    std::priority_queue<RequestPtr> priority_queue_;
    std::unordered_map<std::string, RequestPtr> request_map_;
    std::vector<WorkerState> worker_states_;
    
    // Synchronization
    std::mutex scheduler_mutex_;
    std::condition_variable scheduler_cv_;
    
    // Resource monitoring
    ResourceUsage current_usage_;
    std::mutex resource_mutex_;
};

} // namespace deeppowers 