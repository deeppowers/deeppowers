#include "config_manager.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace deeppowers {

ConfigManager::ConfigManager() {
    init_default_configs();
}

void ConfigManager::load_from_file(const std::string& path) {
    try {
        // Read JSON file
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open config file: " + path);
        }
        
        nlohmann::json json;
        file >> json;
        
        // Load configurations
        load_json_config(json);
        
        // Validate loaded configurations
        if (!validate_configurations()) {
            throw std::runtime_error("Invalid configuration: " + get_validation_errors());
        }
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("JSON parsing error: " + std::string(e.what()));
    }
}

void ConfigManager::save_to_file(const std::string& path) const {
    try {
        // Create JSON configuration
        nlohmann::json json = save_json_config();
        
        // Write to file
        std::ofstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to create config file: " + path);
        }
        
        file << json.dump(4);  // Pretty print with 4-space indent
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("JSON serialization error: " + std::string(e.what()));
    }
}

const QuantizationConfig& ConfigManager::get_quant_config() const {
    return quant_config_;
}

void ConfigManager::update_quant_config(const QuantizationConfig& config) {
    quant_config_ = config;
    if (!validate_quant_config()) {
        throw std::runtime_error("Invalid quantization configuration: " + get_validation_errors());
    }
}

void ConfigManager::update_runtime_params(const std::string& param, const std::string& value) {
    try {
        if (param == "batch_size") {
            runtime_config_.batch_size = convert_param_value<size_t>(value);
        } else if (param == "max_sequence_length") {
            runtime_config_.max_sequence_length = convert_param_value<size_t>(value);
        } else if (param == "timeout_ms") {
            runtime_config_.timeout_ms = convert_param_value<float>(value);
        } else if (param == "enable_async_execution") {
            runtime_config_.enable_async_execution = convert_param_value<bool>(value);
        } else if (param == "enable_profiling") {
            runtime_config_.enable_profiling = convert_param_value<bool>(value);
        } else if (param == "log_level") {
            runtime_config_.log_level = value;
        } else {
            throw std::runtime_error("Unknown parameter: " + param);
        }
        
        if (!validate_runtime_config()) {
            throw std::runtime_error("Invalid runtime configuration: " + get_validation_errors());
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to update parameter " + param + ": " + e.what());
    }
}

void ConfigManager::reset_to_defaults() {
    init_default_configs();
}

bool ConfigManager::validate_configurations() const {
    validation_errors_.clear();
    return validate_quant_config() && 
           validate_hardware_config() && 
           validate_runtime_config();
}

std::string ConfigManager::get_validation_errors() const {
    std::stringstream ss;
    for (const auto& error : validation_errors_) {
        ss << error << "\n";
    }
    return ss.str();
}

void ConfigManager::load_json_config(const nlohmann::json& json) {
    // Load quantization config
    if (json.contains("quantization")) {
        const auto& quant = json["quantization"];
        quant_config_.type = static_cast<QuantizationType>(quant.value("type", 0));
        quant_config_.method = static_cast<QuantizationMethod>(quant.value("method", 0));
        quant_config_.per_channel = quant.value("per_channel", false);
        quant_config_.symmetric = quant.value("symmetric", true);
        quant_config_.calibration_ratio = quant.value("calibration_ratio", 0.01f);
        quant_config_.num_calibration_batches = quant.value("num_calibration_batches", 100);
        quant_config_.excluded_ops = quant.value("excluded_ops", std::vector<std::string>());
        quant_config_.use_tensor_cores = quant.value("use_tensor_cores", true);
        quant_config_.workspace_size_mb = quant.value("workspace_size_mb", 1024);
        quant_config_.num_worker_threads = quant.value("num_worker_threads", 4);
        quant_config_.cache_size_mb = quant.value("cache_size_mb", 512);
        quant_config_.enable_kernel_fusion = quant.value("enable_kernel_fusion", true);
    }
    
    // Load hardware config
    if (json.contains("hardware")) {
        const auto& hw = json["hardware"];
        hardware_config_.device_type = hw.value("device_type", "CUDA");
        hardware_config_.device_id = hw.value("device_id", 0);
        hardware_config_.enable_mixed_precision = hw.value("enable_mixed_precision", true);
        hardware_config_.max_memory_mb = hw.value("max_memory_mb", 0);
        hardware_config_.enable_peer_access = hw.value("enable_peer_access", true);
        hardware_config_.enable_unified_memory = hw.value("enable_unified_memory", false);
    }
    
    // Load runtime config
    if (json.contains("runtime")) {
        const auto& rt = json["runtime"];
        runtime_config_.batch_size = rt.value("batch_size", 32);
        runtime_config_.max_sequence_length = rt.value("max_sequence_length", 2048);
        runtime_config_.timeout_ms = rt.value("timeout_ms", 100.0f);
        runtime_config_.enable_async_execution = rt.value("enable_async_execution", true);
        runtime_config_.enable_profiling = rt.value("enable_profiling", false);
        runtime_config_.log_level = rt.value("log_level", "INFO");
    }
}

nlohmann::json ConfigManager::save_json_config() const {
    nlohmann::json json;
    
    // Save quantization config
    json["quantization"] = {
        {"type", static_cast<int>(quant_config_.type)},
        {"method", static_cast<int>(quant_config_.method)},
        {"per_channel", quant_config_.per_channel},
        {"symmetric", quant_config_.symmetric},
        {"calibration_ratio", quant_config_.calibration_ratio},
        {"num_calibration_batches", quant_config_.num_calibration_batches},
        {"excluded_ops", quant_config_.excluded_ops},
        {"use_tensor_cores", quant_config_.use_tensor_cores},
        {"workspace_size_mb", quant_config_.workspace_size_mb},
        {"num_worker_threads", quant_config_.num_worker_threads},
        {"cache_size_mb", quant_config_.cache_size_mb},
        {"enable_kernel_fusion", quant_config_.enable_kernel_fusion}
    };
    
    // Save hardware config
    json["hardware"] = {
        {"device_type", hardware_config_.device_type},
        {"device_id", hardware_config_.device_id},
        {"enable_mixed_precision", hardware_config_.enable_mixed_precision},
        {"max_memory_mb", hardware_config_.max_memory_mb},
        {"enable_peer_access", hardware_config_.enable_peer_access},
        {"enable_unified_memory", hardware_config_.enable_unified_memory}
    };
    
    // Save runtime config
    json["runtime"] = {
        {"batch_size", runtime_config_.batch_size},
        {"max_sequence_length", runtime_config_.max_sequence_length},
        {"timeout_ms", runtime_config_.timeout_ms},
        {"enable_async_execution", runtime_config_.enable_async_execution},
        {"enable_profiling", runtime_config_.enable_profiling},
        {"log_level", runtime_config_.log_level}
    };
    
    return json;
}

bool ConfigManager::validate_quant_config() const {
    bool valid = true;
    
    // Validate quantization type and method
    if (quant_config_.type != QuantizationType::NONE &&
        quant_config_.method == QuantizationMethod::NONE) {
        validation_errors_.push_back("Quantization method must be specified when type is not NONE");
        valid = false;
    }
    
    // Validate calibration parameters
    if (quant_config_.calibration_ratio <= 0.0f || quant_config_.calibration_ratio > 1.0f) {
        validation_errors_.push_back("Calibration ratio must be between 0 and 1");
        valid = false;
    }
    
    if (quant_config_.num_calibration_batches == 0) {
        validation_errors_.push_back("Number of calibration batches must be greater than 0");
        valid = false;
    }
    
    // Validate hardware-specific parameters
    if (quant_config_.workspace_size_mb == 0) {
        validation_errors_.push_back("Workspace size must be greater than 0");
        valid = false;
    }
    
    if (quant_config_.num_worker_threads == 0) {
        validation_errors_.push_back("Number of worker threads must be greater than 0");
        valid = false;
    }
    
    return valid;
}

bool ConfigManager::validate_hardware_config() const {
    bool valid = true;
    
    // Validate device type
    if (hardware_config_.device_type != "CUDA" &&
        hardware_config_.device_type != "ROCm" &&
        hardware_config_.device_type != "CPU") {
        validation_errors_.push_back("Unsupported device type: " + hardware_config_.device_type);
        valid = false;
    }
    
    // Validate device ID
    if (hardware_config_.device_id < 0) {
        validation_errors_.push_back("Device ID must be non-negative");
        valid = false;
    }
    
    return valid;
}

bool ConfigManager::validate_runtime_config() const {
    bool valid = true;
    
    // Validate batch size
    if (runtime_config_.batch_size == 0) {
        validation_errors_.push_back("Batch size must be greater than 0");
        valid = false;
    }
    
    // Validate sequence length
    if (runtime_config_.max_sequence_length == 0) {
        validation_errors_.push_back("Maximum sequence length must be greater than 0");
        valid = false;
    }
    
    // Validate timeout
    if (runtime_config_.timeout_ms <= 0.0f) {
        validation_errors_.push_back("Timeout must be greater than 0");
        valid = false;
    }
    
    // Validate log level
    if (runtime_config_.log_level != "DEBUG" &&
        runtime_config_.log_level != "INFO" &&
        runtime_config_.log_level != "WARNING" &&
        runtime_config_.log_level != "ERROR") {
        validation_errors_.push_back("Invalid log level: " + runtime_config_.log_level);
        valid = false;
    }
    
    return valid;
}

template<typename T>
T ConfigManager::convert_param_value(const std::string& value) const {
    std::istringstream iss(value);
    T result;
    
    if (!(iss >> result)) {
        throw std::runtime_error("Failed to convert value: " + value);
    }
    
    return result;
}

void ConfigManager::init_default_configs() {
    // Quantization config already has default values from struct definition
    // Hardware config already has default values from struct definition
    // Runtime config already has default values from struct definition
    
    // Validate default configurations
    if (!validate_configurations()) {
        throw std::runtime_error("Default configuration is invalid: " + get_validation_errors());
    }
}

// Explicit template instantiations
template size_t ConfigManager::convert_param_value<size_t>(const std::string&) const;
template int ConfigManager::convert_param_value<int>(const std::string&) const;
template float ConfigManager::convert_param_value<float>(const std::string&) const;
template bool ConfigManager::convert_param_value<bool>(const std::string&) const;

} // namespace deeppowers 