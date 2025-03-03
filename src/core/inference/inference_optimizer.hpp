/**
 * @file inference_optimizer.hpp
 * @brief Classes for optimizing model inference.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include "../model/model.hpp"
#include "../model/tensor.hpp"
#include "../model/model_types.hpp"

namespace deeppowers {

/**
 * @enum OptimizerType
 * @brief Types of optimization that can be applied to a model.
 */
enum class OptimizerType {
    NONE,           ///< No optimization
    FUSION,         ///< Operator fusion
    PRUNING,        ///< Weight pruning
    DISTILLATION,   ///< Knowledge distillation
    QUANTIZATION,   ///< Weight quantization
    CACHING,        ///< KV-Cache optimization
    AUTO            ///< Automatic selection of optimizations
};

/**
 * @struct OptimizerConfig
 * @brief Configuration for the inference optimizer.
 */
struct OptimizerConfig {
    OptimizerType type = OptimizerType::AUTO;  ///< Type of optimization
    OptimizationLevel level = OptimizationLevel::O1;  ///< Optimization aggressiveness
    bool enable_profiling = false;   ///< Whether to collect performance metrics
    std::vector<std::string> op_blacklist;  ///< Operators to exclude from optimization
    std::unordered_map<std::string, float> parameters;  ///< Additional parameters
};

/**
 * @struct OptimizerResult
 * @brief Results and metrics from optimization.
 */
struct OptimizerResult {
    bool success = false;  ///< Whether optimization was successful
    float speedup = 1.0f;  ///< Speedup factor achieved
    float memory_reduction = 0.0f;  ///< Memory usage reduction percentage
    float accuracy_loss = 0.0f;  ///< Accuracy loss percentage
    std::unordered_map<std::string, float> metrics;  ///< Additional metrics
    std::string error_message;  ///< Error message if optimization failed
};

/**
 * @class InferenceOptimizer
 * @brief Optimizes models for faster inference.
 */
class InferenceOptimizer {
public:
    /**
     * @brief Constructor.
     * @param config Configuration for optimization.
     */
    explicit InferenceOptimizer(const OptimizerConfig& config = OptimizerConfig());
    
    /**
     * @brief Destructor.
     */
    ~InferenceOptimizer();
    
    /**
     * @brief Optimize a model for inference.
     * @param model The model to optimize.
     * @return Optimization results and metrics.
     */
    OptimizerResult optimize(Model* model);
    
    /**
     * @brief Apply operator fusion to a model.
     * @param model The model to optimize.
     * @return Optimization results and metrics.
     */
    OptimizerResult apply_fusion(Model* model);
    
    /**
     * @brief Apply weight pruning to a model.
     * @param model The model to optimize.
     * @param sparsity The target sparsity level (0.0-1.0).
     * @return Optimization results and metrics.
     */
    OptimizerResult apply_pruning(Model* model, float sparsity = 0.7f);
    
    /**
     * @brief Apply weight quantization to a model.
     * @param model The model to optimize.
     * @param precision The target precision for weights.
     * @return Optimization results and metrics.
     */
    OptimizerResult apply_quantization(Model* model, PrecisionMode precision = PrecisionMode::INT8);
    
    /**
     * @brief Apply KV-cache optimization to a model.
     * @param model The model to optimize.
     * @return Optimization results and metrics.
     */
    OptimizerResult apply_kv_cache_optimization(Model* model);
    
    /**
     * @brief Reset optimizer state.
     */
    void reset();
    
    /**
     * @brief Set configuration parameter.
     * @param key Parameter name.
     * @param value Parameter value.
     */
    void set_parameter(const std::string& key, float value);
    
    /**
     * @brief Get configuration parameter.
     * @param key Parameter name.
     * @return Parameter value or default value if not set.
     */
    float get_parameter(const std::string& key, float default_value = 0.0f) const;
    
    /**
     * @brief Enable profiling to collect performance metrics.
     * @param enable Whether to enable profiling.
     */
    void enable_profiling(bool enable = true);
    
private:
    /**
     * @brief Select the best optimization strategy for a model.
     * @param model The model to analyze.
     * @return Selected optimization types.
     */
    std::vector<OptimizerType> select_optimizations(const Model* model);
    
    /**
     * @brief Benchmark a model before and after optimization.
     * @param model The model to benchmark.
     * @param before_optimization Whether this is before or after optimization.
     * @return Benchmarking results.
     */
    std::unordered_map<std::string, float> benchmark(const Model* model, bool before_optimization);
    
    /**
     * @brief Check if operator type is in blacklist.
     * @param op_type Operator type string.
     * @return True if operator should be excluded from optimization.
     */
    bool is_blacklisted(const std::string& op_type) const;
    
    /**
     * @brief Analyze model architecture to identify optimization opportunities.
     * @param model The model to analyze.
     * @return Analysis results.
     */
    std::unordered_map<std::string, float> analyze_model(const Model* model);
    
    OptimizerConfig config_;  ///< Optimizer configuration
    std::unordered_map<std::string, float> parameters_;  ///< Optimization parameters
    std::unordered_map<std::string, float> pre_metrics_;  ///< Metrics before optimization
    bool profiling_enabled_ = false;  ///< Whether profiling is enabled
};

} // namespace deeppowers 