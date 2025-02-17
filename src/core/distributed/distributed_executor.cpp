#include "distributed_executor.hpp"
#include <chrono>
#include <algorithm>

namespace deeppowers {

DistributedExecutor::DistributedExecutor(
    std::shared_ptr<DistributedContext> context,
    const DistributedExecutorConfig& config)
    : context_(context)
    , config_(config) {
}

DistributedExecutor::~DistributedExecutor() {
    if (initialized_) {
        finalize();
    }
}

void DistributedExecutor::initialize() {
    if (initialized_) return;
    
    // Initialize worker threads
    init_worker_threads();
    
    // If pipeline parallel is enabled, initialize pipeline buffers
    if (config_.enable_pipeline) {
        init_pipeline_buffers();
    }
    
    initialized_ = true;
}

void DistributedExecutor::finalize() {
    if (!initialized_) return;
    
    // Stop worker threads
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        should_stop_ = true;
    }
    task_cv_.notify_all();
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // Clean up pipeline buffers
    pipeline_buffers_.clear();
    
    initialized_ = false;
}

void DistributedExecutor::execute_forward(
    GPTModel* model,
    const std::vector<std::vector<int32_t>>& batch_input_ids) {
    
    if (!initialized_) throw std::runtime_error("DistributedExecutor not initialized");
    
    // If pipeline parallel is enabled, use pipeline execution
    if (config_.enable_pipeline) {
        execute_pipeline(model, batch_input_ids);
        return;
    }
    
    // Forward propagation in data parallel mode
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Split input data into micro batches
    size_t batch_size = batch_input_ids.size();
    size_t micro_batch_size = (batch_size + config_.num_micro_batches - 1) / config_.num_micro_batches;
    
    for (size_t i = 0; i < config_.num_micro_batches; ++i) {
        size_t start_idx = i * micro_batch_size;
        size_t end_idx = std::min(start_idx + micro_batch_size, batch_size);
        
        if (start_idx >= end_idx) break;
        
        // Extract current micro batch inputs
        std::vector<std::vector<int32_t>> micro_batch_inputs(
            batch_input_ids.begin() + start_idx,
            batch_input_ids.begin() + end_idx);
        
        // Process current micro batch
        process_micro_batch(i);
        
        // If communication and computation overlap is enabled, perform communication when processing the next micro batch
        if (config_.overlap_comm && i < config_.num_micro_batches - 1) {
            overlap_comm_compute();
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Update performance statistics
    perf_stats_.compute_time += duration.count() / 1000.0;  // Convert to milliseconds
    perf_stats_.num_micro_batches += config_.num_micro_batches;
}

void DistributedExecutor::execute_backward(GPTModel* model) {
    if (!initialized_) throw std::runtime_error("DistributedExecutor not initialized");
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Backward propagation
    for (size_t i = 0; i < config_.num_micro_batches; ++i) {
        // Process backward propagation of micro batch
        size_t micro_batch_idx = config_.num_micro_batches - 1 - i;
        
        // If pipeline parallel is enabled, schedule backward propagation tasks
        if (config_.enable_pipeline) {
            schedule_backward_task(context_->config().pipeline_stages - 1, micro_batch_idx);
        } else {
            // TODO: Implement ordinary backward propagation
        }
        
        // If communication and computation overlap is enabled, perform communication when processing the next micro batch
        if (config_.overlap_comm && i < config_.num_micro_batches - 1) {
            overlap_comm_compute();
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Update performance statistics
    perf_stats_.compute_time += duration.count() / 1000.0;
}

void DistributedExecutor::execute_pipeline(
    GPTModel* model,
    const std::vector<std::vector<int32_t>>& batch_input_ids) {
    
    if (!initialized_) throw std::runtime_error("DistributedExecutor not initialized");
    if (!config_.enable_pipeline) throw std::runtime_error("Pipeline execution not enabled");
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Clear task queue
    std::queue<PipelineTask>().swap(task_queue_);
    
    // Schedule pipeline tasks
    schedule_pipeline_tasks();
    
    // Wait for all tasks to complete
    wait_pipeline();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Update performance statistics
    perf_stats_.compute_time += duration.count() / 1000.0;
}

void DistributedExecutor::sync_gradients() {
    if (!initialized_) throw std::runtime_error("DistributedExecutor not initialized");
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Execute gradient synchronization
    // TODO: Implement gradient synchronization logic
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Update performance statistics
    perf_stats_.comm_time += duration.count() / 1000.0;
}

void DistributedExecutor::sync_parameters() {
    if (!initialized_) throw std::runtime_error("DistributedExecutor not initialized");
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Execute parameter synchronization
    // TODO: Implement parameter synchronization logic
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Update performance statistics
    perf_stats_.comm_time += duration.count() / 1000.0;
}

void DistributedExecutor::wait_pipeline() {
    if (!initialized_) return;
    
    std::unique_lock<std::mutex> lock(task_mutex_);
    task_cv_.wait(lock, [this]() { return task_queue_.empty(); });
}

void DistributedExecutor::init_worker_threads() {
    worker_threads_.resize(config_.num_worker_threads);
    
    for (size_t i = 0; i < config_.num_worker_threads; ++i) {
        worker_threads_[i] = std::thread([this]() {
            while (true) {
                PipelineTask task;
                
                {
                    std::unique_lock<std::mutex> lock(task_mutex_);
                    task_cv_.wait(lock, [this]() {
                        return should_stop_ || !task_queue_.empty();
                    });
                    
                    if (should_stop_ && task_queue_.empty()) {
                        break;
                    }
                    
                    if (!task_queue_.empty()) {
                        task = std::move(task_queue_.front());
                        task_queue_.pop();
                    }
                }
                
                if (task.task) {
                    task.task();
                }
                
                task_cv_.notify_all();
            }
        });
    }
}

void DistributedExecutor::init_pipeline_buffers() {
    size_t num_stages = context_->config().pipeline_stages;
    pipeline_buffers_.resize(num_stages);
    
    // Allocate buffers for each stage
    for (auto& buffer : pipeline_buffers_) {
        // TODO: Allocate appropriate buffer size based on model configuration
    }
}

void DistributedExecutor::schedule_pipeline_tasks() {
    size_t num_stages = context_->config().pipeline_stages;
    
    // Forward propagation stage
    for (size_t i = 0; i < config_.num_micro_batches; ++i) {
        for (size_t stage = 0; stage < num_stages; ++stage) {
            schedule_forward_task(stage, i);
        }
    }
    
    // Backward propagation stage
    for (size_t i = 0; i < config_.num_micro_batches; ++i) {
        for (size_t stage = num_stages - 1; stage != size_t(-1); --stage) {
            schedule_backward_task(stage, config_.num_micro_batches - 1 - i);
        }
    }
}

void DistributedExecutor::process_micro_batch(size_t micro_batch_idx) {
    // TODO: Implement micro batch processing logic
}

void DistributedExecutor::overlap_comm_compute() {
    // TODO: Implement communication and computation overlap logic
}

void DistributedExecutor::schedule_forward_task(size_t stage_id, size_t micro_batch_idx) {
    PipelineTask task;
    task.type = PipelineTask::Type::FORWARD;
    task.stage_id = stage_id;
    task.micro_batch_idx = micro_batch_idx;
    
    task.task = [this, stage_id, micro_batch_idx]() {
        // Receive activations from the previous stage
        if (stage_id > 0) {
            recv_activations(stage_id - 1, micro_batch_idx);
        }
        
        // Execute the calculation of the current stage
        // TODO: Implement forward calculation
        
        // Send activations to the next stage
        if (stage_id < context_->config().pipeline_stages - 1) {
            send_activations(stage_id + 1, micro_batch_idx);
        }
    };
    
    std::lock_guard<std::mutex> lock(task_mutex_);
    task_queue_.push(std::move(task));
    task_cv_.notify_one();
}

void DistributedExecutor::schedule_backward_task(size_t stage_id, size_t micro_batch_idx) {
    PipelineTask task;
    task.type = PipelineTask::Type::BACKWARD;
    task.stage_id = stage_id;
    task.micro_batch_idx = micro_batch_idx;
    
    task.task = [this, stage_id, micro_batch_idx]() {
        // Receive gradients from the next stage
        if (stage_id < context_->config().pipeline_stages - 1) {
            recv_gradients(stage_id + 1, micro_batch_idx);
        }
        
        // Execute the backward propagation of the current stage
        // TODO: Implement backward propagation
        
        // Send gradients to the previous stage
        if (stage_id > 0) {
            send_gradients(stage_id - 1, micro_batch_idx);
        }
    };
    
    std::lock_guard<std::mutex> lock(task_mutex_);
    task_queue_.push(std::move(task));
    task_cv_.notify_one();
}

void DistributedExecutor::handle_pipeline_bubble() {
    // TODO: Implement pipeline bubble processing logic
}

void DistributedExecutor::send_activations(size_t stage_id, size_t micro_batch_idx) {
    auto& buffer = pipeline_buffers_[stage_id];
    if (buffer.output_activations) {
        context_->send(
            buffer.output_activations->data(),
            buffer.output_activations->size_in_bytes(),
            stage_id + 1);
    }
}

void DistributedExecutor::recv_activations(size_t stage_id, size_t micro_batch_idx) {
    auto& buffer = pipeline_buffers_[stage_id];
    if (buffer.input_activations) {
        context_->recv(
            buffer.input_activations->data(),
            buffer.input_activations->size_in_bytes(),
            stage_id - 1);
    }
}

void DistributedExecutor::send_gradients(size_t stage_id, size_t micro_batch_idx) {
    auto& buffer = pipeline_buffers_[stage_id];
    if (buffer.output_gradients) {
        context_->send(
            buffer.output_gradients->data(),
            buffer.output_gradients->size_in_bytes(),
            stage_id - 1);
    }
}

void DistributedExecutor::recv_gradients(size_t stage_id, size_t micro_batch_idx) {
    auto& buffer = pipeline_buffers_[stage_id];
    if (buffer.input_gradients) {
        context_->recv(
            buffer.input_gradients->data(),
            buffer.input_gradients->size_in_bytes(),
            stage_id + 1);
    }
}

} // namespace deeppowers 