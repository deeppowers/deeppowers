#pragma once

#include "../hal/hal.hpp"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace deeppowers {

// Preprocessing parameters
struct PreprocessingParams {
    // Normalization parameters
    bool enable_normalization = true;
    float mean = 0.0f;
    float std = 1.0f;
    
    // Quantization preprocessing
    bool collect_statistics = true;
    size_t num_histogram_bins = 2048;
    float outlier_threshold = 3.0f;
    
    // Tensor transformation
    bool enable_padding = true;
    bool enable_masking = true;
    size_t max_sequence_length = 2048;
    
    // Performance options
    size_t num_worker_threads = 4;
    bool enable_async_processing = true;
};

// Statistics for tensor calibration
struct TensorStatistics {
    // Basic statistics
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    double mean = 0.0;
    double variance = 0.0;
    
    // Histogram data
    std::vector<uint32_t> histogram;
    std::vector<float> bin_edges;
    size_t total_samples = 0;
    
    // Percentile values
    float p99_9 = 0.0f;
    float p99 = 0.0f;
    float p95 = 0.0f;
    
    // Update methods
    void update_minmax(float val);
    void update_running_stats(float val);
    void update_histogram(float val);
    void finalize_statistics();
};

class Preprocessor {
public:
    // Constructor and destructor
    explicit Preprocessor(const PreprocessingParams& params);
    ~Preprocessor();
    
    // Tensor preparation
    void prepare_tensor(hal::Tensor* tensor);
    void normalize_inputs(hal::Tensor* tensor);
    
    // Quantization preprocessing
    void prepare_for_quantization(hal::Tensor* tensor);
    void collect_statistics(hal::Tensor* tensor);
    
    // Calibration data management
    void add_calibration_data(const hal::Tensor* tensor);
    void clear_calibration_data();
    
    // Statistics access
    const TensorStatistics& get_statistics(const std::string& tensor_name) const;
    void save_statistics(const std::string& path) const;
    void load_statistics(const std::string& path);
    
    // Configuration
    void set_preprocessing_params(const PreprocessingParams& params);
    const PreprocessingParams& get_preprocessing_params() const;

private:
    // Internal helper methods
    void init_histogram(TensorStatistics& stats);
    void compute_percentiles(TensorStatistics& stats);
    void apply_padding(hal::Tensor* tensor);
    void apply_masking(hal::Tensor* tensor);
    
    // Async processing
    void process_async(hal::Tensor* tensor);
    void wait_for_async_completion();
    
    // Member variables
    PreprocessingParams params_;
    std::unordered_map<std::string, TensorStatistics> tensor_stats_;
    
    // Thread pool for async processing
    struct ThreadPool;
    std::unique_ptr<ThreadPool> thread_pool_;
    
    // Temporary buffers
    std::unique_ptr<hal::Tensor> temp_buffer_;
    std::unique_ptr<hal::Tensor> mask_buffer_;
};

} // namespace deeppowers 