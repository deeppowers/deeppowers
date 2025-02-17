#include "request_queue.hpp"

namespace deeppowers {

RequestQueue::RequestQueue(const RequestQueueConfig& config)
    : config_(config)
    , enqueue_callback_(nullptr)
    , dequeue_callback_(nullptr) {
}

RequestQueue::~RequestQueue() {
    // Clear queue
    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) {
        queue_.pop();
    }
    request_map_.clear();
}

bool RequestQueue::enqueue(RequestPtr request) {
    if (!request) {
        return false;
    }

    std::unique_lock<std::mutex> lock(mutex_);
    
    // Wait for queue not full
    if (!not_full_.wait_for(lock, 
                           config_.max_wait_time,
                           [this] { return !full(); })) {
        return false;  // Timeout
    }

    // Add to queue and map
    queue_.push(request);
    request_map_[request->id()] = request;

    // Notify waiting consumers
    lock.unlock();
    not_empty_.notify_one();

    // Trigger callback
    notify_enqueue(request);

    return true;
}

RequestPtr RequestQueue::dequeue() {
    std::unique_lock<std::mutex> lock(mutex_);
    
    // Wait for queue not empty
    if (!not_empty_.wait_for(lock,
                            config_.max_wait_time,
                            [this] { return !empty(); })) {
        return nullptr;  // Timeout
    }

    // Get request
    RequestPtr request = queue_.top();
    queue_.pop();
    request_map_.erase(request->id());

    // Notify waiting producers
    lock.unlock();
    not_full_.notify_one();

    // Trigger callback
    notify_dequeue(request);

    return request;
}

std::vector<RequestPtr> RequestQueue::dequeue_batch() {
    std::vector<RequestPtr> batch;
    std::unique_lock<std::mutex> lock(mutex_);

    // Wait for queue not empty or batch timeout
    if (!not_empty_.wait_for(lock,
                            config_.batch_timeout,
                            [this] { return !empty(); })) {
        return batch;  // Timeout return empty batch
    }

    // Collect batch
    while (!queue_.empty() && batch.size() < config_.max_batch_size) {
        if (batch.empty() || should_create_new_batch()) {
            RequestPtr request = queue_.top();
            queue_.pop();
            request_map_.erase(request->id());
            batch.push_back(request);
            
            // Trigger callback
            notify_dequeue(request);
        } else {
            break;
        }
    }

    // Notify waiting producers
    lock.unlock();
    not_full_.notify_one();

    return batch;
}

size_t RequestQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

bool RequestQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

bool RequestQueue::full() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size() >= config_.max_queue_size;
}

RequestPtr RequestQueue::find_request(const std::string& request_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = request_map_.find(request_id);
    return (it != request_map_.end()) ? it->second : nullptr;
}

void RequestQueue::set_enqueue_callback(RequestCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    enqueue_callback_ = callback;
}

void RequestQueue::set_dequeue_callback(RequestCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    dequeue_callback_ = callback;
}

void RequestQueue::update_config(const RequestQueueConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
}

bool RequestQueue::should_create_new_batch() const {
    if (!config_.enable_dynamic_batching) {
        return false;
    }
    
    // Here you can add more complex batching strategies
    // For example: check sequence length, model type, hardware utilization, etc.
    return true;
}

void RequestQueue::notify_enqueue(RequestPtr request) {
    if (enqueue_callback_) {
        try {
            enqueue_callback_(request);
        } catch (...) {
            // Callback exception should not affect queue operations
        }
    }
}

void RequestQueue::notify_dequeue(RequestPtr request) {
    if (dequeue_callback_) {
        try {
            dequeue_callback_(request);
        } catch (...) {
            // Callback exception should not affect queue operations
        }
    }
}

} // namespace deeppowers 