#include "scheduler.hpp"
#include <chrono>
#include <algorithm>
#include <nvml.h>

namespace deeppowers {

Scheduler::Scheduler(const SchedulerConfig& config)
    : config_(config) {
}

Scheduler::~Scheduler() {
    if (initialized_) {
        finalize();
    }
}

void Scheduler::initialize() {
    if (initialized_) return;
    
    // Initialize worker states
    worker_states_.resize(config_.num_worker_threads);
    
    // Initialize worker threads
    init_worker_threads();
    
    // Start monitor thread
    monitor_thread_ = std::thread(&Scheduler::monitor_thread_func, this);
    
    // Start cleanup thread
    cleanup_thread_ = std::thread(&Scheduler::cleanup_thread_func, this);
    
    initialized_ = true;
}

void Scheduler::finalize() {
    if (!initialized_) return;
    
    // Stop all threads
    should_stop_ = true;
    scheduler_cv_.notify_all();
    
    // Wait for worker threads
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // Wait for monitor and cleanup threads
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
    if (cleanup_thread_.joinable()) {
        cleanup_thread_.join();
    }
    
    // Clear queues and states
    std::priority_queue<RequestPtr>().swap(priority_queue_);
    request_map_.clear();
    worker_states_.clear();
    
    initialized_ = false;
}

void Scheduler::submit_request(RequestPtr request) {
    if (!initialized_) throw std::runtime_error("Scheduler not initialized");
    
    // Check resource availability
    if (!check_resources(request)) {
        request->mark_failed("Resource limits exceeded");
        stats_.dropped_requests++;
        return;
    }
    
    // Enqueue request
    enqueue_request(request);
    
    // Update statistics
    {
        std::lock_guard<std::mutex> lock(scheduler_mutex_);
        stats_.total_requests++;
        stats_.active_requests++;
    }
    
    // Notify workers
    scheduler_cv_.notify_one();
}

void Scheduler::cancel_request(const std::string& request_id) {
    if (!initialized_) return;
    
    std::lock_guard<std::mutex> lock(scheduler_mutex_);
    
    // Find and remove request
    auto it = request_map_.find(request_id);
    if (it != request_map_.end()) {
        it->second->mark_failed("Request cancelled");
        request_map_.erase(it);
        stats_.active_requests--;
    }
}

RequestPtr Scheduler::get_request_status(const std::string& request_id) {
    if (!initialized_) return nullptr;
    
    std::lock_guard<std::mutex> lock(scheduler_mutex_);
    
    auto it = request_map_.find(request_id);
    return (it != request_map_.end()) ? it->second : nullptr;
}

void Scheduler::process_batch(const std::vector<RequestPtr>& batch) {
    if (batch.empty()) return;
    
    // Update resource usage
    {
        std::lock_guard<std::mutex> lock(resource_mutex_);
        current_usage_.active_batches++;
    }
    
    // Process each request in the batch
    for (const auto& request : batch) {
        // TODO: Implement actual request processing
    }
    
    // Handle batch completion
    handle_batch_completion(batch);
}

void Scheduler::handle_batch_completion(const std::vector<RequestPtr>& batch) {
    // Update statistics
    {
        std::lock_guard<std::mutex> lock(scheduler_mutex_);
        stats_.active_requests -= batch.size();
        
        // Update latency statistics
        double total_latency = 0.0;
        for (const auto& request : batch) {
            total_latency += request->processing_time().count() / 1000.0;  // Convert to ms
        }
        stats_.avg_latency_ms = (stats_.avg_latency_ms * (stats_.total_requests - batch.size()) +
                                total_latency) / stats_.total_requests;
    }
    
    // Update resource usage
    {
        std::lock_guard<std::mutex> lock(resource_mutex_);
        current_usage_.active_batches--;
    }
    
    // Remove completed requests from tracking
    {
        std::lock_guard<std::mutex> lock(scheduler_mutex_);
        for (const auto& request : batch) {
            request_map_.erase(request->id());
        }
    }
}

bool Scheduler::check_resources(const RequestPtr& request) {
    std::lock_guard<std::mutex> lock(resource_mutex_);
    
    // Check memory usage
    if (current_usage_.memory_used + request->estimated_memory_usage() >
        config_.resource_limits.min_free_memory_mb * 1024 * 1024) {
        return false;
    }
    
    // Check GPU utilization
    if (current_usage_.gpu_util > config_.resource_limits.max_gpu_utilization) {
        return false;
    }
    
    // Check active requests
    if (stats_.active_requests >= config_.resource_limits.max_active_requests) {
        return false;
    }
    
    return true;
}

void Scheduler::update_resource_usage() {
    std::lock_guard<std::mutex> lock(resource_mutex_);
    
    // Update GPU utilization using NVML
    nvmlDevice_t device;
    nvmlUtilization_t utilization;
    if (nvmlDeviceGetHandleByIndex(0, &device) == NVML_SUCCESS &&
        nvmlDeviceGetUtilizationRates(device, &utilization) == NVML_SUCCESS) {
        current_usage_.gpu_util = utilization.gpu / 100.0f;
    }
    
    // Update memory usage
    size_t free_memory, total_memory;
    if (nvmlDeviceGetMemoryInfo(device, &free_memory, &total_memory, nullptr) == NVML_SUCCESS) {
        current_usage_.memory_used = total_memory - free_memory;
    }
    
    // Update statistics
    stats_.gpu_utilization = current_usage_.gpu_util;
    stats_.memory_usage_mb = current_usage_.memory_used / (1024 * 1024);
}

void Scheduler::handle_resource_pressure() {
    std::lock_guard<std::mutex> lock(scheduler_mutex_);
    
    if (current_usage_.gpu_util > config_.resource_limits.max_gpu_utilization ||
        current_usage_.memory_used > config_.resource_limits.min_free_memory_mb * 1024 * 1024) {
        
        // Reduce batch size
        optimize_batch_size();
        
        // Pause accepting new requests if necessary
        if (stats_.active_requests > config_.resource_limits.max_active_requests) {
            // TODO: Implement request admission control
        }
    }
}

void Scheduler::balance_load() {
    std::vector<size_t> worker_loads(worker_states_.size());
    
    // Calculate load for each worker
    for (size_t i = 0; i < worker_states_.size(); ++i) {
        std::lock_guard<std::mutex> lock(worker_states_[i].mutex);
        worker_loads[i] = worker_states_[i].local_queue.size();
    }
    
    // Find workers with highest and lowest loads
    auto max_it = std::max_element(worker_loads.begin(), worker_loads.end());
    auto min_it = std::min_element(worker_loads.begin(), worker_loads.end());
    
    size_t max_worker = std::distance(worker_loads.begin(), max_it);
    size_t min_worker = std::distance(worker_loads.begin(), min_it);
    
    // Balance if load difference is significant
    if (*max_it > *min_it + 2) {
        migrate_requests(max_worker, min_worker);
    }
}

void Scheduler::migrate_requests(size_t source_worker, size_t target_worker) {
    std::lock_guard<std::mutex> source_lock(worker_states_[source_worker].mutex);
    std::lock_guard<std::mutex> target_lock(worker_states_[target_worker].mutex);
    
    auto& source_queue = worker_states_[source_worker].local_queue;
    auto& target_queue = worker_states_[target_worker].local_queue;
    
    // Calculate number of requests to migrate
    size_t num_to_migrate = (source_queue.size() - target_queue.size()) / 2;
    
    // Migrate requests
    for (size_t i = 0; i < num_to_migrate && !source_queue.empty(); ++i) {
        target_queue.push(source_queue.front());
        source_queue.pop();
    }
}

void Scheduler::handle_worker_failure(size_t worker_id) {
    std::lock_guard<std::mutex> lock(worker_states_[worker_id].mutex);
    
    // Mark worker as inactive
    worker_states_[worker_id].active = false;
    
    // Requeue failed requests
    auto& failed_queue = worker_states_[worker_id].local_queue;
    while (!failed_queue.empty()) {
        requeue_request(failed_queue.front());
        failed_queue.pop();
    }
    
    // Start recovery process
    recover_failed_requests();
}

void Scheduler::recover_failed_requests() {
    // TODO: Implement request recovery logic
}

bool Scheduler::is_healthy() const {
    // Check worker thread health
    for (size_t i = 0; i < worker_states_.size(); ++i) {
        if (!worker_states_[i].active) {
            return false;
        }
    }
    
    // Check resource usage
    if (current_usage_.gpu_util > config_.resource_limits.max_gpu_utilization ||
        current_usage_.memory_used > config_.resource_limits.min_free_memory_mb * 1024 * 1024) {
        return false;
    }
    
    return true;
}

void Scheduler::update_config(const SchedulerConfig& config) {
    if (initialized_) {
        throw std::runtime_error("Cannot update config while scheduler is running");
    }
    config_ = config;
}

void Scheduler::init_worker_threads() {
    worker_threads_.resize(config_.num_worker_threads);
    
    for (size_t i = 0; i < config_.num_worker_threads; ++i) {
        worker_states_[i].active = true;
        worker_threads_[i] = std::thread(&Scheduler::worker_thread_func, this, i);
    }
}

void Scheduler::worker_thread_func(size_t worker_id) {
    while (!should_stop_) {
        // Get next request batch
        std::vector<RequestPtr> batch = form_batch();
        
        if (!batch.empty()) {
            // Process batch
            process_batch(batch);
            
            // Update worker statistics
            {
                std::lock_guard<std::mutex> lock(worker_states_[worker_id].mutex);
                worker_states_[worker_id].processed_requests += batch.size();
            }
        } else {
            // Wait for new requests
            std::unique_lock<std::mutex> lock(scheduler_mutex_);
            scheduler_cv_.wait(lock, [this]() {
                return should_stop_ || !priority_queue_.empty();
            });
        }
    }
}

void Scheduler::monitor_thread_func() {
    while (!should_stop_) {
        // Update resource usage
        update_resource_usage();
        
        // Check system health
        if (!is_healthy()) {
            handle_resource_pressure();
        }
        
        // Balance load if needed
        if (config_.enable_load_balancing) {
            balance_load();
        }
        
        // Sleep for monitoring interval
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void Scheduler::cleanup_thread_func() {
    while (!should_stop_) {
        // Clean up completed requests
        {
            std::lock_guard<std::mutex> lock(scheduler_mutex_);
            for (auto it = request_map_.begin(); it != request_map_.end();) {
                if (it->second->status() == RequestStatus::COMPLETED ||
                    it->second->status() == RequestStatus::FAILED) {
                    it = request_map_.erase(it);
                } else {
                    ++it;
                }
            }
        }
        
        // Sleep for cleanup interval
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void Scheduler::enqueue_request(RequestPtr request) {
    std::lock_guard<std::mutex> lock(scheduler_mutex_);
    
    // Add to priority queue if enabled
    if (config_.enable_priority_queue) {
        priority_queue_.push(request);
    } else {
        // Round-robin to worker queues
        static size_t next_worker = 0;
        std::lock_guard<std::mutex> worker_lock(worker_states_[next_worker].mutex);
        worker_states_[next_worker].local_queue.push(request);
        next_worker = (next_worker + 1) % worker_states_.size();
    }
    
    // Track request
    request_map_[request->id()] = request;
}

RequestPtr Scheduler::dequeue_request() {
    std::lock_guard<std::mutex> lock(scheduler_mutex_);
    
    if (priority_queue_.empty()) {
        return nullptr;
    }
    
    RequestPtr request = priority_queue_.top();
    priority_queue_.pop();
    return request;
}

void Scheduler::requeue_request(RequestPtr request) {
    if (!request) return;
    
    // Reset request state
    request->set_status(RequestStatus::PENDING);
    
    // Re-enqueue
    enqueue_request(request);
}

bool Scheduler::can_batch_request(
    const RequestPtr& request,
    const std::vector<RequestPtr>& current_batch) {
    
    if (current_batch.empty()) {
        return true;
    }
    
    // Check batch size limit
    if (current_batch.size() >= config_.resource_limits.max_batch_size) {
        return false;
    }
    
    // Check sequence length compatibility
    size_t max_seq_len = 0;
    for (const auto& req : current_batch) {
        max_seq_len = std::max(max_seq_len, req->prompt().length());
    }
    
    size_t new_seq_len = request->prompt().length();
    if (new_seq_len > config_.resource_limits.max_sequence_length ||
        max_seq_len > config_.resource_limits.max_sequence_length) {
        return false;
    }
    
    // Check padding ratio
    size_t total_tokens = 0;
    for (const auto& req : current_batch) {
        total_tokens += req->prompt().length();
    }
    total_tokens += new_seq_len;
    
    float padding_ratio = 1.0f - (float)total_tokens / 
                         ((current_batch.size() + 1) * std::max(max_seq_len, new_seq_len));
    
    return padding_ratio <= 0.2f;  // Maximum 20% padding
}

std::vector<RequestPtr> Scheduler::form_batch() {
    std::vector<RequestPtr> batch;
    
    while (batch.size() < config_.resource_limits.max_batch_size) {
        RequestPtr request = dequeue_request();
        if (!request) break;
        
        if (can_batch_request(request, batch)) {
            batch.push_back(request);
        } else {
            requeue_request(request);
            break;
        }
    }
    
    return batch;
}

void Scheduler::optimize_batch_size() {
    // Dynamically adjust batch size based on resource usage
    float gpu_util = current_usage_.gpu_util;
    size_t memory_used = current_usage_.memory_used;
    
    if (gpu_util > 0.95f || 
        memory_used > config_.resource_limits.min_free_memory_mb * 1024 * 1024) {
        // Reduce batch size
        config_.resource_limits.max_batch_size = 
            std::max(size_t(1), config_.resource_limits.max_batch_size / 2);
    } else if (gpu_util < 0.7f && 
               memory_used < config_.resource_limits.min_free_memory_mb * 1024 * 1024 / 2) {
        // Increase batch size
        config_.resource_limits.max_batch_size = 
            std::min(size_t(128), config_.resource_limits.max_batch_size * 2);
    }
}

} // namespace deeppowers
 