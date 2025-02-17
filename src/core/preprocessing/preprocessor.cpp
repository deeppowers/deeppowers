#include "preprocessor.hpp"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <thread>
#include <future>
#include <queue>
#include <condition_variable>
#include <nlohmann/json.hpp>

namespace deeppowers {

// ThreadPool implementation
struct Preprocessor::ThreadPool {
    ThreadPool(size_t num_threads)
        : stop(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] {
                            return stop || !tasks.empty();
                        });
                        
                        if (stop && tasks.empty()) {
                            return;
                        }
                        
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }
    
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) {
            worker.join();
        }
    }
    
    template<typename F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }
    
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// TensorStatistics implementation
void TensorStatistics::update_minmax(float val) {
    min_val = std::min(min_val, val);
    max_val = std::max(max_val, val);
}

void TensorStatistics::update_running_stats(float val) {
    total_samples++;
    double delta = val - mean;
    mean += delta / total_samples;
    double delta2 = val - mean;
    variance += delta * delta2;
}

void TensorStatistics::update_histogram(float val) {
    if (histogram.empty()) return;
    
    // Find bin index
    size_t bin = 0;
    for (size_t i = 0; i < bin_edges.size() - 1; ++i) {
        if (val >= bin_edges[i] && val < bin_edges[i + 1]) {
            bin = i;
            break;
        }
    }
    
    if (bin < histogram.size()) {
        histogram[bin]++;
    }
}

void TensorStatistics::finalize_statistics() {
    if (total_samples > 1) {
        variance /= (total_samples - 1);
    }
    
    // Compute percentiles from histogram
    if (!histogram.empty()) {
        size_t total = 0;
        for (uint32_t count : histogram) {
            total += count;
        }
        
        size_t count = 0;
        for (size_t i = 0; i < histogram.size(); ++i) {
            count += histogram[i];
            float percentile = static_cast<float>(count) / total;
            
            if (percentile >= 0.999f && p99_9 == 0.0f) {
                p99_9 = bin_edges[i + 1];
            }
            if (percentile >= 0.99f && p99 == 0.0f) {
                p99 = bin_edges[i + 1];
            }
            if (percentile >= 0.95f && p95 == 0.0f) {
                p95 = bin_edges[i + 1];
                break;
            }
        }
    }
}

// Preprocessor implementation
Preprocessor::Preprocessor(const PreprocessingParams& params)
    : params_(params) {
    if (params_.num_worker_threads > 0) {
        thread_pool_ = std::make_unique<ThreadPool>(params_.num_worker_threads);
    }
}

Preprocessor::~Preprocessor() = default;

void Preprocessor::prepare_tensor(hal::Tensor* tensor) {
    if (!tensor) return;
    
    if (params_.enable_async_processing && thread_pool_) {
        process_async(tensor);
    } else {
        if (params_.enable_normalization) {
            normalize_inputs(tensor);
        }
        
        if (params_.enable_padding) {
            apply_padding(tensor);
        }
        
        if (params_.enable_masking) {
            apply_masking(tensor);
        }
    }
}

void Preprocessor::normalize_inputs(hal::Tensor* tensor) {
    if (!tensor) return;
    
    // Create temporary buffer if needed
    if (!temp_buffer_ || temp_buffer_->shape() != tensor->shape()) {
        temp_buffer_ = std::make_unique<hal::Tensor>(
            tensor->shape(), tensor->dtype(), tensor->device());
    }
    
    // Launch normalization kernel
    const float inv_std = 1.0f / params_.std;
    size_t num_elements = tensor->size_in_bytes() / sizeof(float);
    
    auto normalize_kernel = tensor->device()->create_kernel("normalize_kernel");
    normalize_kernel->set_arg(0, sizeof(float*), &tensor->data());
    normalize_kernel->set_arg(1, sizeof(float*), &temp_buffer_->data());
    normalize_kernel->set_arg(2, sizeof(float), &params_.mean);
    normalize_kernel->set_arg(3, sizeof(float), &inv_std);
    normalize_kernel->set_arg(4, sizeof(size_t), &num_elements);
    
    hal::Kernel::LaunchConfig config;
    config.grid_dim = {(num_elements + 255) / 256};
    config.block_dim = {256};
    normalize_kernel->launch(config);
    
    // Copy result back
    tensor->copy_from_device(temp_buffer_->data());
}

void Preprocessor::prepare_for_quantization(hal::Tensor* tensor) {
    if (!tensor) return;
    
    // Collect statistics if enabled
    if (params_.collect_statistics) {
        collect_statistics(tensor);
    }
}

void Preprocessor::collect_statistics(hal::Tensor* tensor) {
    if (!tensor) return;
    
    // Get or create statistics entry
    auto& stats = tensor_stats_[tensor->name()];
    
    // Initialize histogram if needed
    if (stats.histogram.empty()) {
        init_histogram(stats);
    }
    
    // Copy tensor to host for statistics collection
    std::vector<float> host_data(tensor->size_in_bytes() / sizeof(float));
    tensor->copy_to_host(host_data.data());
    
    // Update statistics
    for (float val : host_data) {
        stats.update_minmax(val);
        stats.update_running_stats(val);
        stats.update_histogram(val);
    }
}

void Preprocessor::add_calibration_data(const hal::Tensor* tensor) {
    if (!tensor) return;
    collect_statistics(const_cast<hal::Tensor*>(tensor));
}

void Preprocessor::clear_calibration_data() {
    tensor_stats_.clear();
}

const TensorStatistics& Preprocessor::get_statistics(const std::string& tensor_name) const {
    auto it = tensor_stats_.find(tensor_name);
    if (it == tensor_stats_.end()) {
        throw std::runtime_error("No statistics available for tensor: " + tensor_name);
    }
    return it->second;
}

void Preprocessor::save_statistics(const std::string& path) const {
    nlohmann::json json;
    
    for (const auto& [name, stats] : tensor_stats_) {
        nlohmann::json stat_json;
        stat_json["min_val"] = stats.min_val;
        stat_json["max_val"] = stats.max_val;
        stat_json["mean"] = stats.mean;
        stat_json["variance"] = stats.variance;
        stat_json["total_samples"] = stats.total_samples;
        stat_json["histogram"] = stats.histogram;
        stat_json["bin_edges"] = stats.bin_edges;
        stat_json["p99_9"] = stats.p99_9;
        stat_json["p99"] = stats.p99;
        stat_json["p95"] = stats.p95;
        
        json[name] = stat_json;
    }
    
    std::ofstream file(path);
    file << json.dump(4);
}

void Preprocessor::load_statistics(const std::string& path) {
    std::ifstream file(path);
    nlohmann::json json;
    file >> json;
    
    tensor_stats_.clear();
    
    for (const auto& [name, stat_json] : json.items()) {
        TensorStatistics stats;
        stats.min_val = stat_json["min_val"];
        stats.max_val = stat_json["max_val"];
        stats.mean = stat_json["mean"];
        stats.variance = stat_json["variance"];
        stats.total_samples = stat_json["total_samples"];
        stats.histogram = stat_json["histogram"].get<std::vector<uint32_t>>();
        stats.bin_edges = stat_json["bin_edges"].get<std::vector<float>>();
        stats.p99_9 = stat_json["p99_9"];
        stats.p99 = stat_json["p99"];
        stats.p95 = stat_json["p95"];
        
        tensor_stats_[name] = stats;
    }
}

void Preprocessor::set_preprocessing_params(const PreprocessingParams& params) {
    params_ = params;
    
    // Update thread pool if needed
    if (params_.num_worker_threads > 0) {
        if (!thread_pool_ || 
            thread_pool_->workers.size() != params_.num_worker_threads) {
            thread_pool_ = std::make_unique<ThreadPool>(params_.num_worker_threads);
        }
    } else {
        thread_pool_.reset();
    }
}

const PreprocessingParams& Preprocessor::get_preprocessing_params() const {
    return params_;
}

void Preprocessor::init_histogram(TensorStatistics& stats) {
    stats.histogram.resize(params_.num_histogram_bins, 0);
    stats.bin_edges.resize(params_.num_histogram_bins + 1);
    
    // Initialize bin edges
    float range = stats.max_val - stats.min_val;
    float bin_width = range / params_.num_histogram_bins;
    
    for (size_t i = 0; i <= params_.num_histogram_bins; ++i) {
        stats.bin_edges[i] = stats.min_val + i * bin_width;
    }
}

void Preprocessor::compute_percentiles(TensorStatistics& stats) {
    stats.finalize_statistics();
}

void Preprocessor::apply_padding(hal::Tensor* tensor) {
    if (!tensor) return;
    
    // Get current sequence length
    const auto& shape = tensor->shape();
    size_t seq_length = shape[shape.size() - 2];
    
    if (seq_length >= params_.max_sequence_length) {
        return;  // No padding needed
    }
    
    // Create padded tensor
    std::vector<int64_t> padded_shape = shape;
    padded_shape[shape.size() - 2] = params_.max_sequence_length;
    
    auto padded_tensor = std::make_unique<hal::Tensor>(
        padded_shape, tensor->dtype(), tensor->device());
    
    // Launch padding kernel
    auto padding_kernel = tensor->device()->create_kernel("padding_kernel");
    padding_kernel->set_arg(0, sizeof(float*), &tensor->data());
    padding_kernel->set_arg(1, sizeof(float*), &padded_tensor->data());
    padding_kernel->set_arg(2, sizeof(size_t), &seq_length);
    padding_kernel->set_arg(3, sizeof(size_t), &params_.max_sequence_length);
    
    hal::Kernel::LaunchConfig config;
    config.grid_dim = {(params_.max_sequence_length + 255) / 256};
    config.block_dim = {256};
    padding_kernel->launch(config);
    
    // Update tensor
    tensor->copy_from_device(padded_tensor->data());
}

void Preprocessor::apply_masking(hal::Tensor* tensor) {
    if (!tensor) return;
    
    // Create mask buffer if needed
    const auto& shape = tensor->shape();
    if (!mask_buffer_ || mask_buffer_->shape() != shape) {
        mask_buffer_ = std::make_unique<hal::Tensor>(
            shape, hal::DataType::FLOAT32, tensor->device());
    }
    
    // Launch masking kernel
    auto masking_kernel = tensor->device()->create_kernel("masking_kernel");
    masking_kernel->set_arg(0, sizeof(float*), &tensor->data());
    masking_kernel->set_arg(1, sizeof(float*), &mask_buffer_->data());
    
    hal::Kernel::LaunchConfig config;
    config.grid_dim = {(shape[0] + 255) / 256};
    config.block_dim = {256};
    masking_kernel->launch(config);
    
    // Update tensor
    tensor->copy_from_device(mask_buffer_->data());
}

void Preprocessor::process_async(hal::Tensor* tensor) {
    if (!thread_pool_) return;
    
    std::vector<std::future<void>> futures;
    
    // Queue preprocessing tasks
    if (params_.enable_normalization) {
        futures.push_back(std::async(std::launch::async, [this, tensor]() {
            normalize_inputs(tensor);
        }));
    }
    
    if (params_.enable_padding) {
        futures.push_back(std::async(std::launch::async, [this, tensor]() {
            apply_padding(tensor);
        }));
    }
    
    if (params_.enable_masking) {
        futures.push_back(std::async(std::launch::async, [this, tensor]() {
            apply_masking(tensor);
        }));
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.wait();
    }
}

void Preprocessor::wait_for_async_completion() {
    // No explicit waiting needed as futures handle synchronization
}

} // namespace deeppowers 