#pragma once

#include "request.hpp"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <functional>

namespace deeppowers {

// Request queue configuration options
struct RequestQueueConfig {
    size_t max_queue_size = 1000;          // Maximum queue length
    size_t max_batch_size = 32;            // Maximum batch processing size
    bool enable_dynamic_batching = true;    // Whether to enable dynamic batch processing
    std::chrono::milliseconds max_wait_time{100};  // Maximum waiting time
    std::chrono::milliseconds batch_timeout{10};   // Batch processing timeout
};

// Request queue class
class RequestQueue {
public:
    explicit RequestQueue(const RequestQueueConfig& config = RequestQueueConfig());
    ~RequestQueue();

    // Disable copy and move
    RequestQueue(const RequestQueue&) = delete;
    RequestQueue& operator=(const RequestQueue&) = delete;
    RequestQueue(RequestQueue&&) = delete;
    RequestQueue& operator=(RequestQueue&&) = delete;

    // Request management
    bool enqueue(RequestPtr request);
    RequestPtr dequeue();
    std::vector<RequestPtr> dequeue_batch();

    // Status query
    size_t size() const;
    bool empty() const;
    bool full() const;
    
    // Request lookup
    RequestPtr find_request(const std::string& request_id) const;
    
    // Callback settings
    using RequestCallback = std::function<void(RequestPtr)>;
    void set_enqueue_callback(RequestCallback callback);
    void set_dequeue_callback(RequestCallback callback);
    
    // Configuration access and update
    const RequestQueueConfig& config() const { return config_; }
    void update_config(const RequestQueueConfig& config);

private:
    // Priority comparator
    struct RequestComparator {
        bool operator()(const RequestPtr& a, const RequestPtr& b) const {
            // Higher priority requests
            if (a->priority() != b->priority()) {
                return a->priority() < b->priority();
            }
            // Same priority sorted by creation time
            return a->created_time() > b->created_time();
        }
    };

    RequestQueueConfig config_;
    
    // Request storage
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::priority_queue<RequestPtr, std::vector<RequestPtr>, RequestComparator> queue_;
    std::unordered_map<std::string, RequestPtr> request_map_;  // For fast lookup
    
    // Callback functions
    RequestCallback enqueue_callback_;
    RequestCallback dequeue_callback_;
    
    // Internal helper methods
    bool should_create_new_batch() const;
    void notify_enqueue(RequestPtr request);
    void notify_dequeue(RequestPtr request);
};

} // namespace deeppowers 