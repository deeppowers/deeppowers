#include "postprocessor.hpp"
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
struct Postprocessor::ThreadPool {
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

// ErrorMetrics implementation
void ErrorMetrics::update_error_metrics(float original, float quantized) {
    float abs_error = std::abs(original - quantized);
    float rel_error = (original != 0.0f) ? abs_error / std::abs(original) : 0.0f;
    
    // Update absolute error metrics
    max_absolute_error = std::max(max_absolute_error, abs_error);
    mean_absolute_error += abs_error;
    mean_squared_error += abs_error * abs_error;
    
    // Update relative error metrics
    max_relative_error = std::max(max_relative_error, rel_error);
    mean_relative_error += rel_error;
    
    // Update error distribution
    if (!error_histogram.empty()) {
        size_t bin = static_cast<size_t>(abs_error * error_histogram.size());
        if (bin < error_histogram.size()) {
            error_histogram[bin]++;
        }
    }
}

void ErrorMetrics::update_error_distribution() {
    if (error_histogram.empty()) return;
    
    // Compute cumulative distribution
    std::vector<float> cdf(error_histogram.size());
    float total = 0.0f;
    for (size_t i = 0; i < error_histogram.size(); ++i) {
        total += error_histogram[i];
        cdf[i] = total;
    }
    
    // Normalize CDF
    if (total > 0.0f) {
        for (float& val : cdf) {
            val /= total;
        }
    }
    
    // Compute percentiles
    error_percentiles.resize(9);  // p10, p20, ..., p90
    for (size_t i = 0; i < error_percentiles.size(); ++i) {
        float target = (i + 1) * 0.1f;
        auto it = std::lower_bound(cdf.begin(), cdf.end(), target);
        size_t idx = std::distance(cdf.begin(), it);
        error_percentiles[i] = static_cast<float>(idx) / error_histogram.size();
    }
}

void ErrorMetrics::finalize_metrics() {
    // Finalize mean errors
    if (mean_absolute_error > 0.0f) {
        mean_absolute_error /= num_outliers;
        mean_squared_error /= num_outliers;
        mean_relative_error /= num_outliers;
        root_mean_squared_error = std::sqrt(mean_squared_error);
    }
    
    // Update error distribution
    update_error_distribution();
}

// ValidationReport implementation
std::string ValidationReport::generate_summary() const {
    std::stringstream ss;
    
    ss << "Validation Report Summary:\n";
    ss << "-------------------------\n";
    ss << "Overall Status: " << (validation_passed ? "PASSED" : "FAILED") << "\n\n";
    
    for (const auto& msg : validation_messages) {
        ss << msg << "\n";
    }
    
    ss << "\nError Metrics by Tensor:\n";
    for (const auto& [name, metrics] : tensor_metrics) {
        ss << "Tensor: " << name << "\n";
        ss << "  Max Absolute Error: " << metrics.max_absolute_error << "\n";
        ss << "  Mean Absolute Error: " << metrics.mean_absolute_error << "\n";
        ss << "  RMSE: " << metrics.root_mean_squared_error << "\n";
        ss << "  Max Relative Error: " << metrics.max_relative_error << "\n";
        ss << "  Outliers: " << metrics.num_outliers << "\n\n";
    }
    
    ss << "Performance Metrics:\n";
    ss << "  Processing Time: " << processing_time_ms << " ms\n";
    ss << "  Throughput: " << throughput << " elements/s\n";
    
    return ss.str();
}

void ValidationReport::save_to_file(const std::string& path) const {
    nlohmann::json json;
    
    // Save validation status
    json["validation_passed"] = validation_passed;
    json["validation_messages"] = validation_messages;
    
    // Save tensor metrics
    nlohmann::json tensor_json;
    for (const auto& [name, metrics] : tensor_metrics) {
        nlohmann::json metrics_json;
        metrics_json["max_absolute_error"] = metrics.max_absolute_error;
        metrics_json["mean_absolute_error"] = metrics.mean_absolute_error;
        metrics_json["mean_squared_error"] = metrics.mean_squared_error;
        metrics_json["root_mean_squared_error"] = metrics.root_mean_squared_error;
        metrics_json["max_relative_error"] = metrics.max_relative_error;
        metrics_json["mean_relative_error"] = metrics.mean_relative_error;
        metrics_json["num_outliers"] = metrics.num_outliers;
        metrics_json["error_percentiles"] = metrics.error_percentiles;
        
        tensor_json[name] = metrics_json;
    }
    json["tensor_metrics"] = tensor_json;
    
    // Save performance metrics
    json["processing_time_ms"] = processing_time_ms;
    json["throughput"] = throughput;
    
    // Write to file
    std::ofstream file(path);
    file << json.dump(4);
}

// Postprocessor implementation
Postprocessor::Postprocessor(const PostprocessingParams& params)
    : params_(params) {
    if (params_.num_worker_threads > 0) {
        thread_pool_ = std::make_unique<ThreadPool>(params_.num_worker_threads);
    }
}

Postprocessor::~Postprocessor() = default;

void Postprocessor::process_quantized_output(hal::Tensor* tensor) {
    if (!tensor) return;
    
    if (params_.enable_async_processing && thread_pool_) {
        process_async(tensor);
    } else {
        if (params_.enable_dequantization) {
            apply_dequantization(tensor);
        }
        
        if (params_.validate_outputs) {
            validate_tensor_values(tensor);
        }
    }
}

void Postprocessor::apply_dequantization(hal::Tensor* tensor) {
    if (!tensor) return;
    
    // Create temporary buffer if needed
    if (!temp_buffer_ || temp_buffer_->shape() != tensor->shape()) {
        temp_buffer_ = std::make_unique<hal::Tensor>(
            tensor->shape(), hal::DataType::FLOAT32, tensor->device());
    }
    
    // Launch dequantization kernel
    auto dequant_kernel = tensor->device()->create_kernel("dequantize_kernel");
    dequant_kernel->set_arg(0, sizeof(void*), &tensor->data());
    dequant_kernel->set_arg(1, sizeof(void*), &temp_buffer_->data());
    
    hal::Kernel::LaunchConfig config;
    config.grid_dim = {(tensor->size_in_bytes() + 255) / 256};
    config.block_dim = {256};
    dequant_kernel->launch(config);
    
    // Copy result back
    tensor->copy_from_device(temp_buffer_->data());
}

void Postprocessor::verify_quantization_accuracy(
    const hal::Tensor* original,
    const hal::Tensor* quantized) {
    
    if (!original || !quantized) return;
    
    // Create error buffer if needed
    if (!error_buffer_ || error_buffer_->shape() != original->shape()) {
        error_buffer_ = std::make_unique<hal::Tensor>(
            original->shape(), hal::DataType::FLOAT32, original->device());
    }
    
    // Launch error computation kernel
    auto error_kernel = original->device()->create_kernel("compute_error_kernel");
    error_kernel->set_arg(0, sizeof(void*), &original->data());
    error_kernel->set_arg(1, sizeof(void*), &quantized->data());
    error_kernel->set_arg(2, sizeof(void*), &error_buffer_->data());
    
    hal::Kernel::LaunchConfig config;
    config.grid_dim = {(original->size_in_bytes() + 255) / 256};
    config.block_dim = {256};
    error_kernel->launch(config);
    
    // Copy error data to host and update metrics
    std::vector<float> error_data(error_buffer_->size_in_bytes() / sizeof(float));
    error_buffer_->copy_to_host(error_data.data());
    
    ErrorMetrics metrics;
    for (float error : error_data) {
        metrics.update_error_metrics(error, 0.0f);  // Error is already computed
    }
    
    metrics.finalize_metrics();
    update_error_metrics(metrics);
}

void Postprocessor::collect_output_statistics(const hal::Tensor* tensor) {
    if (!tensor) return;
    
    // Copy tensor to host for statistics collection
    std::vector<float> host_data(tensor->size_in_bytes() / sizeof(float));
    tensor->copy_to_host(host_data.data());
    
    // Compute basic statistics
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    double mean = 0.0;
    double variance = 0.0;
    
    for (float val : host_data) {
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        mean += val;
    }
    
    mean /= host_data.size();
    
    for (float val : host_data) {
        double diff = val - mean;
        variance += diff * diff;
    }
    
    variance /= (host_data.size() - 1);
    
    // Update validation report
    validation_report_.validation_messages.push_back(
        "Output statistics for tensor " + tensor->name() + ":");
    validation_report_.validation_messages.push_back(
        "  Min: " + std::to_string(min_val));
    validation_report_.validation_messages.push_back(
        "  Max: " + std::to_string(max_val));
    validation_report_.validation_messages.push_back(
        "  Mean: " + std::to_string(mean));
    validation_report_.validation_messages.push_back(
        "  Std: " + std::to_string(std::sqrt(variance)));
}

void Postprocessor::update_error_metrics(const ErrorMetrics& metrics) {
    // Update validation report
    validation_report_.tensor_metrics[tensor_name] = metrics;
    
    // Check if validation passed
    bool passed = true;
    if (metrics.max_absolute_error > params_.error_threshold) {
        passed = false;
        validation_report_.validation_messages.push_back(
            "Maximum absolute error exceeds threshold");
    }
    
    if (metrics.max_relative_error > params_.relative_error_threshold) {
        passed = false;
        validation_report_.validation_messages.push_back(
            "Maximum relative error exceeds threshold");
    }
    
    validation_report_.validation_passed &= passed;
}

const ValidationReport& Postprocessor::get_validation_report() const {
    return validation_report_;
}

void Postprocessor::reset_validation_report() {
    validation_report_ = ValidationReport();
}

void Postprocessor::set_postprocessing_params(const PostprocessingParams& params) {
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

const PostprocessingParams& Postprocessor::get_postprocessing_params() const {
    return params_;
}

void Postprocessor::init_error_metrics() {
    // Initialize error histogram
    for (auto& [name, metrics] : validation_report_.tensor_metrics) {
        metrics.error_histogram.resize(100, 0);  // 100 bins for error distribution
    }
}

void Postprocessor::compute_error_statistics() {
    for (auto& [name, metrics] : validation_report_.tensor_metrics) {
        metrics.finalize_metrics();
    }
}

void Postprocessor::validate_tensor_values(const hal::Tensor* tensor) {
    if (!tensor) return;
    
    // Copy tensor to host for validation
    std::vector<float> host_data(tensor->size_in_bytes() / sizeof(float));
    tensor->copy_to_host(host_data.data());
    
    // Check for invalid values
    size_t num_invalid = 0;
    for (float val : host_data) {
        if (std::isnan(val) || std::isinf(val)) {
            num_invalid++;
        }
    }
    
    if (num_invalid > 0) {
        validation_report_.validation_passed = false;
        validation_report_.validation_messages.push_back(
            "Found " + std::to_string(num_invalid) + " invalid values in tensor " +
            tensor->name());
    }
}

void Postprocessor::process_async(hal::Tensor* tensor) {
    if (!thread_pool_) return;
    
    std::vector<std::future<void>> futures;
    
    // Queue postprocessing tasks
    if (params_.enable_dequantization) {
        futures.push_back(std::async(std::launch::async, [this, tensor]() {
            apply_dequantization(tensor);
        }));
    }
    
    if (params_.validate_outputs) {
        futures.push_back(std::async(std::launch::async, [this, tensor]() {
            validate_tensor_values(tensor);
        }));
    }
    
    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.wait();
    }
}

void Postprocessor::wait_for_async_completion() {
    // No explicit waiting needed as futures handle synchronization
}

} // namespace deeppowers 