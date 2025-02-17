#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include <nlohmann/json.hpp>

namespace deeppowers {

// Quantization configuration
struct QuantizationConfig {
    // Basic settings
    QuantizationType type = QuantizationType::NONE;
    QuantizationMethod method = QuantizationMethod::NONE;
    bool per_channel = false;
    bool symmetric = true;
    
    // Advanced settings
    float calibration_ratio = 0.01f;
    size_t num_calibration_batches = 100;
    std::vector<std::string> excluded_ops;
    
    // Hardware specific
    bool use_tensor_cores = true;
    size_t workspace_size_mb = 1024;
    
    // Performance tuning
    size_t num_worker_threads = 4;
    size_t cache_size_mb = 512;
    bool enable_kernel_fusion = true;
};

// Hardware configuration
struct HardwareConfig {
    std::string device_type = "CUDA";
    int device_id = 0;
    bool enable_mixed_precision = true;
    size_t max_memory_mb = 0;  // 0 means use all available
    bool enable_peer_access = true;
    bool enable_unified_memory = false;
};

// Runtime configuration
struct RuntimeConfig {
    size_t batch_size = 32;
    size_t max_sequence_length = 2048;
    float timeout_ms = 100.0f;
    bool enable_async_execution = true;
    bool enable_profiling = false;
    std::string log_level = "INFO";
};

class ConfigManager {
public:
    // Constructor and destructor
    ConfigManager();
    ~ConfigManager() = default;
    
    // Load/save configurations
    void load_from_file(const std::string& path);
    void save_to_file(const std::string& path) const;
    
    // Access configurations
    const QuantizationConfig& get_quant_config() const { return quant_config_; }
    const HardwareConfig& get_hardware_config() const { return hardware_config_; }
    const RuntimeConfig& get_runtime_config() const { return runtime_config_; }
    
    // Update configurations
    void update_quant_config(const QuantizationConfig& config);
    void update_hardware_config(const HardwareConfig& config);
    void update_runtime_config(const RuntimeConfig& config);
    
    // Runtime parameter updates
    void update_runtime_param(const std::string& param, const std::string& value);
    
    // Reset configurations
    void reset_to_defaults();
    
    // Validation
    bool validate_configurations() const;
    std::string get_validation_errors() const;

private:
    // Configuration storage
    QuantizationConfig quant_config_;
    HardwareConfig hardware_config_;
    RuntimeConfig runtime_config_;
    
    // Validation state
    mutable std::vector<std::string> validation_errors_;
    
    // Internal helper methods
    void load_json_config(const nlohmann::json& json);
    nlohmann::json save_json_config() const;
    bool validate_quant_config() const;
    bool validate_hardware_config() const;
    bool validate_runtime_config() const;
    
    // Parameter conversion helpers
    template<typename T>
    T convert_param_value(const std::string& value) const;
    
    // Default configuration initialization
    void init_default_configs();
};

} // namespace deeppowers 