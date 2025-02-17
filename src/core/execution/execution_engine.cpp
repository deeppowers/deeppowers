#include "execution_engine.hpp"
#include <algorithm>
#include <stdexcept>

namespace deeppowers {

ExecutionEngine::ExecutionEngine(std::unique_ptr<Model> model,
                               hal::Device* device,
                               const ExecutionEngineConfig& config)
    : model_(std::move(model))
    , device_(device)
    , config_(config)
    , running_(false)
    , num_active_batches_(0)
    , memory_manager_({0, 0})
    , compute_graph_({false}) {
    
    if (!model_) {
        throw std::runtime_error("Model cannot be null");
    }
    if (!device_) {
        throw std::runtime_error("Device cannot be null");
    }
}

ExecutionEngine::~ExecutionEngine() {
    stop();
}

void ExecutionEngine::start() {
    if (running_) {
        return;
    }

    running_ = true;
    
    // Optimize compute graph
    if (config_.enable_graph_optimization) {
        optimize_compute_graph();
    }
    
    // Create worker threads
    for (size_t i = 0; i < config_.num_worker_threads; ++i) {
        worker_threads_.emplace_back(&ExecutionEngine::worker_thread, this);
    }
}

void ExecutionEngine::stop() {
    if (!running_) {
        return;
    }

    running_ = false;
    
    // Notify all waiting threads
    queue_cv_.notify_all();
    
    // Wait for all worker threads to finish
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
}

void ExecutionEngine::execute_batch(BatchPtr batch) {
    if (!batch) {
        return;
    }

    // Process batch synchronously
    process_batch(batch);
}

void ExecutionEngine::execute_batch_async(BatchPtr batch) {
    if (!batch) {
        return;
    }

    // Check active batch count
    if (num_active_batches_ >= config_.max_active_batches) {
        throw std::runtime_error("Maximum number of active batches reached");
    }

    // Add to queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        batch_queue_.push(batch);
    }
    
    // Notify worker threads
    queue_cv_.notify_one();
}

size_t ExecutionEngine::get_num_active_batches() const {
    return num_active_batches_;
}

void ExecutionEngine::update_config(const ExecutionEngineConfig& config) {
    // Need to stop engine to update config
    bool was_running = running_;
    if (was_running) {
        stop();
    }

    config_ = config;

    if (was_running) {
        start();
    }
}

void ExecutionEngine::worker_thread() {
    while (running_) {
        BatchPtr batch;
        
        // Get batch
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !running_ || !batch_queue_.empty();
            });
            
            if (!running_) {
                break;
            }
            
            batch = batch_queue_.front();
            batch_queue_.pop();
        }
        
        // Process batch
        if (batch) {
            ++num_active_batches_;
            process_batch(batch);
            --num_active_batches_;
        }
    }
}

void ExecutionEngine::process_batch(BatchPtr batch) {
    if (!batch) {
        return;
    }

    try {
        // Mark batch as being processed
        batch->mark_started();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Manage memory
        manage_memory();
        
        // Prepare input tensors
        const auto& requests = batch->requests();
        std::vector<std::vector<int32_t>> batch_input_ids;
        batch_input_ids.reserve(requests.size());
        
        for (const auto& request : requests) {
            batch_input_ids.push_back(request->input_ids());
        }
        
        // Create output tensor
        size_t batch_size = requests.size();
        size_t max_seq_length = config_.max_sequence_length;
        size_t vocab_size = model_->vocab_size();
        
        std::vector<int64_t> output_shape = {
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(max_seq_length),
            static_cast<int64_t>(vocab_size)
        };
        
        auto output_tensor = std::make_unique<hal::Tensor>(
            output_shape, hal::DataType::FLOAT32, device_);
        
        // Execute forward propagation
        model_->forward_batch(batch_input_ids, output_tensor.get());
        
        // Process output
        for (size_t i = 0; i < batch_size; ++i) {
            auto& request = requests[i];
            
            // Get logits for this batch
            std::vector<float> logits(vocab_size);
            output_tensor->copy_to_host(
                logits.data(),
                i * max_seq_length * vocab_size,
                vocab_size);
            
            // Sample next token
            int32_t next_token = model_->sample_token(
                logits,
                request->temperature(),
                request->top_p(),
                request->top_k());
            
            // Update request output
            request->add_output_token(next_token);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto execution_time = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
        
        // Update stats
        update_stats(batch, execution_time);
        
        // Mark batch completed
        batch->mark_completed();
    } catch (const std::exception& e) {
        batch->mark_failed(e.what());
    }
}

void ExecutionEngine::optimize_compute_graph() {
    std::lock_guard<std::mutex> lock(compute_graph_.mutex);
    if (compute_graph_.is_optimized) {
        return;
    }

    // TODO: Implement compute graph optimization
    // 1. Operator fusion
    // 2. Memory layout optimization
    // 3. Compute scheduling optimization
    
    compute_graph_.is_optimized = true;
}

void ExecutionEngine::manage_memory() {
    std::lock_guard<std::mutex> lock(memory_manager_.mutex);
    
    // Get current device memory usage
    size_t free_memory, total_memory;
    if (device_->type() == hal::DeviceType::CUDA) {
        free_memory = device_->available_memory();
        total_memory = device_->total_memory();
    } else {
        // For other device types, add corresponding memory query logic
        return;
    }
    
    // Calculate used memory
    size_t used_memory = total_memory - free_memory;
    memory_manager_.total_allocated = used_memory;
    memory_manager_.peak_allocated = std::max(memory_manager_.peak_allocated, used_memory);
    
    // Check if memory optimization is needed
    if (used_memory > config_.workspace_size_mb * 1024 * 1024) {
        // Execute memory optimization strategy
        
        // 1. Clean up unused caches
        if (model_) {
            // For GPT model, clean up unused KV caches
            auto* gpt_model = dynamic_cast<GPTModel*>(model_.get());
            if (gpt_model) {
                gpt_model->clear_kv_cache();
            }
        }
        
        // 2. Compress tensor memory
        // TODO: Implement tensor memory compression
        
        // 3. Force synchronization of device, ensure all operations are completed
        device_->synchronize();
    }
    
    // Update stats
    stats_.peak_memory_usage_mb = memory_manager_.peak_allocated / (1024 * 1024);
}

void ExecutionEngine::update_stats(const BatchPtr& batch,
                                 const std::chrono::microseconds& execution_time) {
    // Update stats
    ++stats_.total_batches_processed;
    stats_.total_requests_processed += batch->requests().size();
    
    // Update time stats
    double batch_time_ms = execution_time.count() / 1000.0;
    stats_.average_batch_time_ms = 
        (stats_.average_batch_time_ms * (stats_.total_batches_processed - 1) +
         batch_time_ms) / stats_.total_batches_processed;
    
    // Update memory stats
    std::lock_guard<std::mutex> lock(memory_manager_.mutex);
    stats_.peak_memory_usage_mb = 
        std::max(stats_.peak_memory_usage_mb,
                memory_manager_.peak_allocated / (1024 * 1024));
}

} // namespace deeppowers