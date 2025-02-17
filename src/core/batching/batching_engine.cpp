#include "batching_engine.hpp"

namespace deeppowers {

BatchingEngine::BatchingEngine(std::shared_ptr<RequestQueue> request_queue,
                             const BatchingEngineConfig& config)
    : request_queue_(request_queue)
    , config_(config)
    , running_(false)
    , num_processed_batches_(0)
    , stats_({0, 0}) {
    
    if (!request_queue_) {
        throw std::runtime_error("Request queue cannot be null");
    }
}

BatchingEngine::~BatchingEngine() {
    stop();
}

void BatchingEngine::start() {
    if (running_) {
        return;
    }

    running_ = true;
    
    // Create worker threads
    for (size_t i = 0; i < config_.num_worker_threads; ++i) {
        worker_threads_.emplace_back(&BatchingEngine::worker_thread, this);
    }
}

void BatchingEngine::stop() {
    if (!running_) {
        return;
    }

    running_ = false;
    
    // Wait for all worker threads to finish
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
}

void BatchingEngine::set_batch_ready_callback(BatchCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    batch_ready_callback_ = callback;
}

void BatchingEngine::set_batch_complete_callback(BatchCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    batch_complete_callback_ = callback;
}

float BatchingEngine::get_average_batch_size() const {
    std::lock_guard<std::mutex> lock(stats_.mutex);
    if (num_processed_batches_ == 0) {
        return 0.0f;
    }
    return static_cast<float>(stats_.total_batch_size) / num_processed_batches_;
}

float BatchingEngine::get_average_padding_ratio() const {
    std::lock_guard<std::mutex> lock(stats_.mutex);
    if (stats_.total_batch_size == 0) {
        return 0.0f;
    }
    return static_cast<float>(stats_.total_padding_tokens) / 
           (stats_.total_batch_size * config_.max_sequence_length);
}

void BatchingEngine::update_config(const BatchingEngineConfig& config) {
    // Need to stop the engine to update the configuration
    bool was_running = running_;
    if (was_running) {
        stop();
    }

    config_ = config;

    if (was_running) {
        start();
    }
}

void BatchingEngine::worker_thread() {
    while (running_) {
        // Get a batch of requests from the request queue
        std::vector<RequestPtr> requests = request_queue_->dequeue_batch();
        if (requests.empty()) {
            continue;
        }

        // Create a batch
        BatchPtr batch = create_batch(requests);
        if (!batch) {
            continue;
        }

        // Notify the batch is ready
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (batch_ready_callback_) {
                batch_ready_callback_(batch);
            }
        }

        // Process the batch
        process_batch(batch);

        // Update statistics
        {
            std::lock_guard<std::mutex> lock(stats_.mutex);
            stats_.total_batch_size += batch->requests().size();
            stats_.total_padding_tokens += batch->stats().padding_tokens;
        }
        ++num_processed_batches_;

        // Notify the batch is complete
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (batch_complete_callback_) {
                batch_complete_callback_(batch);
            }
        }
    }
}

BatchPtr BatchingEngine::create_batch(const std::vector<RequestPtr>& requests) {
    if (requests.empty() || requests.size() > config_.max_batch_size) {
        return nullptr;
    }

    // Create a new batch
    auto batch = std::make_shared<Batch>(requests);

    // Check if the batch meets the requirements
    if (batch->stats().padding_ratio > config_.max_padding_ratio) {
        return nullptr;
    }

    return batch;
}

bool BatchingEngine::can_add_to_batch(const BatchPtr& batch, 
                                    const RequestPtr& request) const {
    if (!batch || !request) {
        return false;
    }

    // Check the batch size
    if (batch->requests().size() >= config_.max_batch_size) {
        return false;
    }

    // Check the sequence length
    size_t sequence_length = request->prompt().length(); // Temporary use character length instead
    if (sequence_length > config_.max_sequence_length) {
        return false;
    }

    // Check the padding ratio
    if (!config_.enable_dynamic_batching) {
        return false;
    }

    return batch->is_compatible(request);
}

void BatchingEngine::process_batch(BatchPtr batch) {
    if (!batch) {
        return;
    }

    try {
        // Mark the batch as being processed
        batch->mark_started();

        // TODO: Actual batch processing logic
        // This will be handled by the execution engine

        // Mark the batch as completed
        batch->mark_completed();
    } catch (const std::exception& e) {
        batch->mark_failed(e.what());
    }
}

} // namespace deeppowers