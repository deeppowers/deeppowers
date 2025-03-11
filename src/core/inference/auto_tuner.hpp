/**
 * @file auto_tuner.hpp
 * @brief Automatic performance tuning system for inference optimization.
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <random>
#include <chrono>

#include "../model/model.hpp"
#include "inference_engine.hpp"
#include "inference_optimizer.hpp"

namespace deeppowers {

/**
 * @enum TuningMethod
 * @brief Methods for parameter tuning.
 */
enum class TuningMethod {
    GRID_SEARCH,       ///< Exhaustive search over parameter grid
    RANDOM_SEARCH,     ///< Random sampling of parameter space
    BAYESIAN_OPT,      ///< Bayesian optimization
    GENETIC_ALGORITHM, ///< Genetic algorithm-based optimization
    ANNEALING          ///< Simulated annealing
};

/**
 * @enum TuningObjective
 * @brief Optimization objectives for tuning.
 */
enum class TuningObjective {
    LATENCY,           ///< Minimize inference latency
    THROUGHPUT,        ///< Maximize throughput
    MEMORY,            ///< Minimize memory usage
    ACCURACY,          ///< Maximize accuracy
    BALANCED           ///< Balance between latency, memory, and accuracy
};

/**
 * @enum QuantizationMethod
 * @brief Supported quantization methods
 */
enum class QuantizationMethod {
    NONE,           ///< No quantization (FP32)
    INT8,           ///< 8-bit integer quantization
    FP16,           ///< 16-bit floating point
    INT4,           ///< 4-bit integer quantization
    INT16,          ///< 16-bit integer quantization
    DYNAMIC,        ///< Dynamic quantization
    MIXED          ///< Mixed precision quantization
};

/**
 * @enum CalibrationMethod
 * @brief Methods for quantization calibration
 */
enum class CalibrationMethod {
    MINMAX,         ///< Min-max calibration
    KL_DIVERGENCE,  ///< KL divergence-based calibration
    MSE,           ///< Mean squared error calibration
    ENTROPY,       ///< Entropy-based calibration
    PERCENTILE     ///< Percentile-based calibration
};

/**
 * @struct TuningParameter
 * @brief Definition of a parameter to be tuned.
 */
struct TuningParameter {
    std::string name;                  ///< Parameter name
    std::vector<float> discrete_values; ///< Possible discrete values
    float min_value = 0.0f;            ///< Minimum value for continuous parameters
    float max_value = 0.0f;            ///< Maximum value for continuous parameters
    bool is_discrete = true;           ///< Whether parameter has discrete values
    bool is_log_scale = false;         ///< Whether to sample on log scale
};

/**
 * @struct DynamicDimension
 * @brief 
 */
struct DynamicDimension {
    std::string name;                  ///< 
    int min_size = 1;                 ///< 
    int max_size = 1024;              ///< 
    bool is_dynamic = true;           ///< 
    std::vector<int> fixed_sizes;     ///< （）
};

/**
 * @struct ShapeConfig
 * @brief 
 */
struct ShapeConfig {
    std::string input_name;                    ///< 
    std::vector<DynamicDimension> dimensions;  ///< 
    bool enable_padding = false;               ///< 
    int padding_multiple = 8;                  ///< 
    bool enable_batching = false;              ///< 
};

/**
 * @struct DynamicShapeStrategy
 * @brief 
 */
enum class DynamicShapeStrategy {
    FIXED,              ///< 
    DYNAMIC,            ///< 
    OPTIMIZE_SIZES,     ///< 
    PADDING_OPTIMIZE    ///< 
};

/**
 * @struct QuantizationConfig
 * @brief Configuration for model quantization
 */
struct QuantizationConfig {
    QuantizationMethod method = QuantizationMethod::NONE;  ///< Quantization method
    CalibrationMethod calib_method = CalibrationMethod::MINMAX;  ///< Calibration method
    bool per_channel = false;     ///< Whether to use per-channel quantization
    bool symmetric = true;        ///< Whether to use symmetric quantization
    float calib_ratio = 0.1f;    ///< Ratio of data to use for calibration
    int num_bits = 8;            ///< Number of bits for quantization
    float tolerance = 0.01f;     ///< Accuracy tolerance for quantization
    std::vector<std::string> excluded_ops;  ///< Operations to exclude from quantization
    std::vector<std::string> custom_scales; ///< Custom scaling factors for specific layers
};

/**
 * @struct QuantizationResult
 * @brief Results from quantization process
 */
struct QuantizationResult {
    bool success = false;         ///< Whether quantization was successful
    float accuracy_fp32 = 0.0f;   ///< Accuracy with FP32 precision
    float accuracy_quantized = 0.0f;  ///< Accuracy after quantization
    float memory_reduction = 0.0f;    ///< Memory reduction ratio
    float speed_up = 0.0f;           ///< Speed improvement ratio
    std::string error_message;        ///< Error message if quantization failed
    std::unordered_map<std::string, float> layer_wise_errors;  ///< Per-layer quantization errors
    std::unordered_map<std::string, std::vector<float>> calibration_stats;  ///< Calibration statistics
};

/**
 * @struct TuningConfig
 * @brief Configuration for the auto-tuning process.
 */
struct TuningConfig {
    TuningMethod method = TuningMethod::GRID_SEARCH;  ///< Tuning method
    TuningObjective objective = TuningObjective::BALANCED;  ///< Optimization objective
    int max_trials = 20;               ///< Maximum number of trials
    int warmup_runs = 3;               ///< Number of warmup runs per trial
    int benchmark_runs = 5;            ///< Number of benchmark runs per trial
    float timeout_seconds = 300.0f;    ///< Maximum time for tuning
    bool verbose = false;              ///< Whether to print verbose output
    std::vector<TuningParameter> parameters;  ///< Parameters to tune
    std::string calibration_data_path; ///< Path to calibration data
    std::vector<ShapeConfig> shape_configs;            ///< Shape configurations
    DynamicShapeStrategy shape_strategy =              ///< Shape handling strategy
        DynamicShapeStrategy::DYNAMIC;
    bool optimize_padding = true;                      ///< Whether to optimize padding
    bool optimize_batch_size = true;                   ///< Whether to optimize batch size
    std::vector<int> target_batch_sizes = {1, 4, 8};  ///< Target batch sizes
    QuantizationConfig quantization;                   ///< Quantization configuration
};

/**
 * @struct TuningResult
 * @brief Results from the auto-tuning process.
 */
struct TuningResult {
    bool success = false;              ///< Whether tuning was successful
    std::unordered_map<std::string, float> best_params;  ///< Best parameters found
    float latency_ms = 0.0f;           ///< Achieved latency in milliseconds
    float throughput = 0.0f;           ///< Achieved throughput in tokens/second
    float memory_mb = 0.0f;            ///< Memory usage in MB
    float accuracy = 0.0f;             ///< Achieved accuracy (if applicable)
    std::string error_message;         ///< Error message if tuning failed
    int trials_completed = 0;          ///< Number of trials completed
    float tuning_time_seconds = 0.0f;  ///< Total time spent tuning
    std::vector<std::unordered_map<std::string, float>> all_trials;  ///< Results from all trials
};

/**
 * @class AutoTuner
 * @brief Automatic performance tuning system for inference optimization.
 * 
 * The AutoTuner class provides functionality to automatically tune
 * parameters for optimal inference performance based on the specific
 * model, hardware, and optimization objectives.
 */
class AutoTuner {
public:
    /**
     * @brief Constructor.
     * @param config Configuration for the auto-tuning process.
     */
    explicit AutoTuner(const TuningConfig& config = TuningConfig());
    
    /**
     * @brief Destructor.
     */
    ~AutoTuner();
    
    /**
     * @brief Tune a model for optimal performance.
     * @param model The model to tune.
     * @param sample_inputs Sample inputs for benchmarking.
     * @return Tuning results.
     */
    TuningResult tune(Model* model, const std::vector<std::vector<int>>& sample_inputs);
    
    /**
     * @brief Apply the best parameters found during tuning.
     * @param model The model to apply parameters to.
     * @param params The parameters to apply.
     * @return Whether application was successful.
     */
    bool apply_best_parameters(Model* model, const std::unordered_map<std::string, float>& params);
    
    /**
     * @brief Set the tuning configuration.
     * @param config New configuration.
     */
    void set_config(const TuningConfig& config);
    
    /**
     * @brief Get the current tuning configuration.
     * @return Current configuration.
     */
    TuningConfig get_config() const;
    
    /**
     * @brief Add a parameter to tune.
     * @param param Parameter definition.
     */
    void add_parameter(const TuningParameter& param);
    
    /**
     * @brief Set the optimization objective.
     * @param objective The objective to optimize for.
     */
    void set_objective(TuningObjective objective);
    
    /**
     * @brief Set the tuning method.
     * @param method The method to use for tuning.
     */
    void set_method(TuningMethod method);
    
    /**
     * @brief Register a custom evaluation function.
     * @param eval_func Function that evaluates a parameter set and returns a score.
     */
    void register_custom_evaluator(
        std::function<float(Model*, const std::unordered_map<std::string, float>&)> eval_func);
    
    /**
     * @brief 
     * @param configs 
     */
    void set_shape_configs(const std::vector<ShapeConfig>& configs);
    
    /**
     * @brief 
     * @param strategy 
     */
    void set_shape_strategy(DynamicShapeStrategy strategy);
    
    /**
     * @brief 
     * @return 
     */
    std::vector<std::vector<std::vector<int>>> generate_sample_shapes() const;
    
    /**
     * @brief 
     * @param shape 
     * @return 
     */
    std::vector<int> optimize_padding(const std::vector<int>& shape) const;
    
    /**
     * @brief 
     * @param inputs 
     * @return 
     */
    std::vector<std::vector<std::vector<int>>> optimize_batching(
        const std::vector<std::vector<int>>& inputs) const;

    /**
     * @brief Set quantization configuration
     * @param config Quantization configuration
     */
    void set_quantization_config(const QuantizationConfig& config);

    /**
     * @brief Get current quantization configuration
     * @return Current quantization configuration
     */
    QuantizationConfig get_quantization_config() const;

    /**
     * @brief Quantize the model using current configuration
     * @param model The model to quantize
     * @param calibration_data Data for calibration
     * @return Quantization results
     */
    QuantizationResult quantize_model(
        Model* model,
        const std::vector<std::vector<int>>& calibration_data);

    /**
     * @brief Calibrate quantization parameters
     * @param model The model to calibrate
     * @param calibration_data Calibration dataset
     * @return Whether calibration was successful
     */
    bool calibrate_quantization(
        Model* model,
        const std::vector<std::vector<int>>& calibration_data);

    /**
     * @brief Evaluate quantization accuracy
     * @param model The quantized model
     * @param test_data Test dataset
     * @return Evaluation metrics
     */
    QuantizationResult evaluate_quantization(
        Model* model,
        const std::vector<std::vector<int>>& test_data);

private:
    /**
     * @brief Evaluate a set of parameters.
     * @param model The model to evaluate.
     * @param params The parameters to evaluate.
     * @param sample_inputs Sample inputs for benchmarking.
     * @return Evaluation score (higher is better).
     */
    float evaluate_parameters(
        Model* model, 
        const std::unordered_map<std::string, float>& params,
        const std::vector<std::vector<int>>& sample_inputs);
    
    /**
     * @brief Run grid search optimization.
     * @param model The model to tune.
     * @param sample_inputs Sample inputs for benchmarking.
     * @return Best parameters found.
     */
    std::unordered_map<std::string, float> run_grid_search(
        Model* model, 
        const std::vector<std::vector<int>>& sample_inputs);
    
    /**
     * @brief Run random search optimization.
     * @param model The model to tune.
     * @param sample_inputs Sample inputs for benchmarking.
     * @return Best parameters found.
     */
    std::unordered_map<std::string, float> run_random_search(
        Model* model, 
        const std::vector<std::vector<int>>& sample_inputs);
    
    /**
     * @brief Run Bayesian optimization.
     * @param model The model to tune.
     * @param sample_inputs Sample inputs for benchmarking.
     * @return Best parameters found.
     */
    std::unordered_map<std::string, float> run_bayesian_optimization(
        Model* model, 
        const std::vector<std::vector<int>>& sample_inputs);
    
    /**
     * @brief Run genetic algorithm optimization.
     * @param model The model to tune.
     * @param sample_inputs Sample inputs for benchmarking.
     * @return Best parameters found.
     */
    std::unordered_map<std::string, float> run_genetic_algorithm(
        Model* model, 
        const std::vector<std::vector<int>>& sample_inputs);
    
    /**
     * @brief Run simulated annealing optimization.
     * @param model The model to tune.
     * @param sample_inputs Sample inputs for benchmarking.
     * @return Best parameters found.
     */
    std::unordered_map<std::string, float> run_simulated_annealing(
        Model* model, 
        const std::vector<std::vector<int>>& sample_inputs);
    
    /**
     * @brief Generate a random parameter set.
     * @return Randomly generated parameters.
     */
    std::unordered_map<std::string, float> generate_random_parameters();
    
    /**
     * @brief Calculate score based on the current objective.
     * @param latency Inference latency in milliseconds.
     * @param throughput Throughput in tokens/second.
     * @param memory Memory usage in MB.
     * @param accuracy Accuracy score (0-100).
     * @return Objective score (higher is better).
     */
    float calculate_objective_score(float latency, float throughput, float memory, float accuracy);
    
    /**
     * @brief Benchmark a model with specific parameters.
     * @param model The model to benchmark.
     * @param params The parameters to apply.
     * @param sample_inputs Sample inputs for benchmarking.
     * @return Benchmark metrics.
     */
    std::unordered_map<std::string, float> benchmark_model(
        Model* model, 
        const std::unordered_map<std::string, float>& params,
        const std::vector<std::vector<int>>& sample_inputs);
    
    /**
     * @brief 
     * @param shape 
     * @param config 
     * @return 
     */
    bool validate_shape(const std::vector<int>& shape, const ShapeConfig& config) const;
    
    /**
     * @brief 
     * @param config 
     * @return 
     */
    std::vector<int> generate_random_shape(const ShapeConfig& config) const;
    
    TuningConfig config_;  ///< Tuning configuration
    std::unordered_map<std::string, float> best_params_;  ///< Best parameters found
    float best_score_ = -std::numeric_limits<float>::max();  ///< Best score found
    std::vector<std::unordered_map<std::string, float>> all_trials_;  ///< Results from all trials
    std::function<float(Model*, const std::unordered_map<std::string, float>&)> custom_evaluator_;  ///< Custom evaluation function
    std::mt19937 random_generator_;  ///< Random number generator
};

} // namespace deeppowers 