#pragma once

#include "../hal/hal.hpp"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace deeppowers {

// Postprocessing parameters
struct PostprocessingParams {
    // Dequantization parameters
    bool enable_dequantization = true;
    bool symmetric_dequantization = true;
    
    // Output validation
    bool validate_outputs = true;
    float error_threshold = 0.1f;
    float relative_error_threshold = 0.01f;
    
    // Performance options
    size_t num_worker_threads = 4;
    bool enable_async_processing = true;
};

// Error metrics for quantization validation
struct ErrorMetrics {
    // Basic error metrics
    float max_absolute_error = 0.0f;
    float mean_absolute_error = 0.0f;
    float mean_squared_error = 0.0f;
    float root_mean_squared_error = 0.0f;
    
    // Relative error metrics
    float max_relative_error = 0.0f;
    float mean_relative_error = 0.0f;
    
    // Error distribution
    std::vector<float> error_histogram;
    std::vector<float> error_percentiles;
    size_t num_outliers = 0;
    
    // Update methods
    void update_error_metrics(float original, float quantized);
    void update_error_distribution();
    void finalize_metrics();
};

// Validation report
struct ValidationReport {
    // Overall validation status
    bool validation_passed = true;
    std::vector<std::string> validation_messages;
    
    // Error metrics per tensor
    std::unordered_map<std::string, ErrorMetrics> tensor_metrics;
    
    // Performance metrics
    double processing_time_ms = 0.0;
    double throughput = 0.0;
    
    // Generate report
    std::string generate_summary() const;
    void save_to_file(const std::string& path) const;
};

class Postprocessor {
public:
    // Constructor and destructor
    explicit Postprocessor(const PostprocessingParams& params);
    ~Postprocessor();
    
    // Quantization postprocessing
    void process_quantized_output(hal::Tensor* tensor);
    void apply_dequantization(hal::Tensor* tensor);
    
    // Result verification
    void verify_quantization_accuracy(
        const hal::Tensor* original,
        const hal::Tensor* quantized);
    
    // Statistics collection
    void collect_output_statistics(const hal::Tensor* tensor);
    void update_error_metrics(const ErrorMetrics& metrics);
    
    // Validation reporting
    const ValidationReport& get_validation_report() const;
    void reset_validation_report();
    
    // Configuration
    void set_postprocessing_params(const PostprocessingParams& params);
    const PostprocessingParams& get_postprocessing_params() const;

private:
    // Internal helper methods
    void init_error_metrics();
    void compute_error_statistics();
    void validate_tensor_values(const hal::Tensor* tensor);
    
    // Async processing
    void process_async(hal::Tensor* tensor);
    void wait_for_async_completion();
    
    // Member variables
    PostprocessingParams params_;
    ValidationReport validation_report_;
    
    // Thread pool for async processing
    struct ThreadPool;
    std::unique_ptr<ThreadPool> thread_pool_;
    
    // Temporary buffers
    std::unique_ptr<hal::Tensor> temp_buffer_;
    std::unique_ptr<hal::Tensor> error_buffer_;
};

} // namespace deeppowers 