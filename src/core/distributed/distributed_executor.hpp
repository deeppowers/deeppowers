#pragma once

#include "distributed_context.hpp"
#include "../execution/models/gpt_model.hpp"
#include <memory>
#include <queue>
#include <thread>
#include <condition_variable>

namespace deeppowers {

// Distributed execution configuration
struct DistributedExecutorConfig {
    size_t num_micro_batches = 1;           // Number of micro batches
    size_t gradient_accumulation_steps = 1;  // Gradient accumulation steps
    bool enable_pipeline = false;            // Whether to enable pipeline parallel
    bool enable_zero = false;                // Whether to enable ZeRO optimization
    size_t prefetch_depth = 2;              // Prefetch depth
    bool overlap_comm = true;               // Whether to overlap communication and computation
    size_t num_worker_threads = 4;          // Number of worker threads
};

// Distributed executor class
class DistributedExecutor {
public:
    DistributedExecutor(
        std::shared_ptr<DistributedContext> context,
        const DistributedExecutorConfig& config);
    ~DistributedExecutor();
    
    // Initialize and clean up
    void initialize();
    void finalize();
    
    // Execution methods
    void execute_forward(GPTModel* model, const std::vector<std::vector<int32_t>>& batch_input_ids);
    void execute_backward(GPTModel* model);
    void execute_pipeline(GPTModel* model, const std::vector<std::vector<int32_t>>& batch_input_ids);
    
    // Synchronization methods
    void sync_gradients();
    void sync_parameters();
    void wait_pipeline();
    
    // Status query
    bool is_initialized() const { return initialized_; }
    const DistributedExecutorConfig& config() const { return config_; }

private:
    // Internal helper methods
    void init_worker_threads();
    void init_pipeline_buffers();
    void schedule_pipeline_tasks();
    void process_micro_batch(size_t micro_batch_idx);
    void overlap_comm_compute();
    
    // Pipeline scheduling related methods
    void schedule_forward_task(size_t stage_id, size_t micro_batch_idx);
    void schedule_backward_task(size_t stage_id, size_t micro_batch_idx);
    void handle_pipeline_bubble();
    
    // Communication related methods
    void send_activations(size_t stage_id, size_t micro_batch_idx);
    void recv_activations(size_t stage_id, size_t micro_batch_idx);
    void send_gradients(size_t stage_id, size_t micro_batch_idx);
    void recv_gradients(size_t stage_id, size_t micro_batch_idx);
    
    // Internal state
    struct PipelineBuffer {
        std::unique_ptr<hal::Tensor> input_activations;
        std::unique_ptr<hal::Tensor> output_activations;
        std::unique_ptr<hal::Tensor> input_gradients;
        std::unique_ptr<hal::Tensor> output_gradients;
    };
    
    struct PipelineTask {
        enum class Type {
            FORWARD,
            BACKWARD,
            SYNC
        };
        
        Type type;
        size_t stage_id;
        size_t micro_batch_idx;
        std::function<void()> task;
    };
    
    // Member variables
    std::shared_ptr<DistributedContext> context_;
    DistributedExecutorConfig config_;
    bool initialized_ = false;
    
    // Pipeline state
    std::vector<PipelineBuffer> pipeline_buffers_;
    std::queue<PipelineTask> task_queue_;
    std::vector<std::thread> worker_threads_;
    std::mutex task_mutex_;
    std::condition_variable task_cv_;
    bool should_stop_ = false;
    
    // Performance statistics
    struct PerfStats {
        double compute_time = 0.0;
        double comm_time = 0.0;
        double idle_time = 0.0;
        size_t num_micro_batches = 0;
    };
    PerfStats perf_stats_;
};

} // namespace deeppowers 