#include "monitor.hpp"
#include <algorithm>
#include <numeric>
#include <nvml.h>
#include <sstream>
#include <fstream>

namespace deeppowers {

Monitor::Monitor(hal::Device* device, const MonitorConfig& config)
    : device_(device)
    , config_(config) {
}

Monitor::~Monitor() {
    if (initialized_) {
        finalize();
    }
}

void Monitor::initialize() {
    if (initialized_) return;
    
    // Initialize GPU monitoring if enabled
    if (config_.enable_gpu_monitoring) {
        init_gpu_monitoring();
    }
    
    // Initialize memory monitoring if enabled
    if (config_.enable_memory_monitoring) {
        init_memory_monitoring();
    }
    
    // Start monitoring thread
    should_stop_ = false;
    monitor_thread_ = std::thread(&Monitor::monitor_thread_func, this);
    
    initialized_ = true;
}

void Monitor::finalize() {
    if (!initialized_) return;
    
    // Stop monitoring thread
    should_stop_ = true;
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
    
    // Cleanup metrics
    cleanup_old_metrics();
    
    initialized_ = false;
}

void Monitor::record_latency(const std::string& operation, double latency_ms) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    // Add latency sample
    latency_samples_.push_back(latency_ms);
    
    // Update component metrics
    auto& metrics = component_metrics_[operation];
    metrics.compute_time_ms += latency_ms;
    metrics.num_operations++;
    
    // Calculate new average
    metrics.throughput = metrics.num_operations / (metrics.compute_time_ms / 1000.0);
    
    // Cleanup old samples if needed
    if (latency_samples_.size() > config_.metrics_history_size) {
        latency_samples_.erase(latency_samples_.begin());
    }
    
    // Recalculate percentiles
    calculate_latency_percentiles();
}

void Monitor::record_throughput(const std::string& operation, double operations) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    // Add throughput sample
    throughput_samples_.push_back(operations);
    
    // Update component metrics
    auto& metrics = component_metrics_[operation];
    metrics.num_operations += static_cast<size_t>(operations);
    
    // Cleanup old samples if needed
    if (throughput_samples_.size() > config_.metrics_history_size) {
        throughput_samples_.erase(throughput_samples_.begin());
    }
    
    // Recalculate statistics
    calculate_throughput_statistics();
}

void Monitor::record_error(const std::string& operation, const std::string& error_type) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    // Add error sample
    error_samples_.push_back(error_type);
    
    // Update component metrics
    auto& metrics = component_metrics_[operation];
    metrics.num_errors++;
    
    // Cleanup old samples if needed
    if (error_samples_.size() > config_.metrics_history_size) {
        error_samples_.erase(error_samples_.begin());
    }
    
    // Analyze error patterns
    analyze_error_patterns();
}

void Monitor::update_component_metrics(
    const std::string& component,
    const ComponentMetrics& metrics) {
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    component_metrics_[component] = metrics;
}

void Monitor::update_hardware_metrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    if (auto* cuda_device = dynamic_cast<hal::CUDADevice*>(device_)) {
        nvmlDevice_t device;
        if (nvmlDeviceGetHandleByIndex(cuda_device->device_id(), &device) == NVML_SUCCESS) {
            // Get GPU utilization
            nvmlUtilization_t utilization;
            if (nvmlDeviceGetUtilizationRates(device, &utilization) == NVML_SUCCESS) {
                hardware_metrics_.gpu_compute_util = utilization.gpu / 100.0;
                hardware_metrics_.gpu_memory_util = utilization.memory / 100.0;
            }
            
            // Get memory usage
            nvmlMemory_t memory;
            if (nvmlDeviceGetMemoryInfo(device, &memory) == NVML_SUCCESS) {
                hardware_metrics_.gpu_memory_used_mb = memory.used / (1024.0 * 1024.0);
            }
            
            // Get temperature
            unsigned int temperature;
            if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature) == NVML_SUCCESS) {
                hardware_metrics_.gpu_temperature = temperature;
            }
            
            // Get power usage
            unsigned int power;
            if (nvmlDeviceGetPowerUsage(device, &power) == NVML_SUCCESS) {
                hardware_metrics_.gpu_power_watts = power / 1000.0;  // Convert from milliwatts to watts
            }
            
            // Get PCIe throughput
            unsigned int rx_throughput, tx_throughput;
            if (nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_RX_BYTES, &rx_throughput) == NVML_SUCCESS &&
                nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_TX_BYTES, &tx_throughput) == NVML_SUCCESS) {
                hardware_metrics_.pcie_throughput = rx_throughput + tx_throughput;
                hardware_metrics_.pcie_bandwidth = (rx_throughput + tx_throughput) / 
                    static_cast<double>(device_->total_memory());
            }
        }
    }
}

void Monitor::track_memory_usage(size_t bytes_allocated, size_t bytes_freed) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    // Update host memory usage
    hardware_metrics_.host_memory_used_mb += 
        static_cast<double>(bytes_allocated - bytes_freed) / (1024.0 * 1024.0);
}

void Monitor::track_gpu_utilization() {
    update_hardware_metrics();
}

ComponentMetrics Monitor::get_component_metrics(const std::string& component) const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    auto it = component_metrics_.find(component);
    return (it != component_metrics_.end()) ? it->second : ComponentMetrics{};
}

HardwareMetrics Monitor::get_hardware_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return hardware_metrics_;
}

LatencyMetrics Monitor::get_latency_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    LatencyMetrics metrics;
    if (!latency_samples_.empty()) {
        // Calculate basic statistics
        metrics.min_latency_ms = *std::min_element(latency_samples_.begin(), latency_samples_.end());
        metrics.max_latency_ms = *std::max_element(latency_samples_.begin(), latency_samples_.end());
        metrics.avg_latency_ms = std::accumulate(latency_samples_.begin(), latency_samples_.end(), 0.0) /
                                latency_samples_.size();
        
        // Copy histogram
        metrics.latency_histogram = latency_samples_;
    }
    return metrics;
}

ThroughputMetrics Monitor::get_throughput_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    ThroughputMetrics metrics;
    if (!throughput_samples_.empty()) {
        // Calculate average throughput
        metrics.operations_per_second = std::accumulate(throughput_samples_.begin(),
                                                      throughput_samples_.end(), 0.0) /
                                      throughput_samples_.size();
        
        // Copy history
        metrics.throughput_history = throughput_samples_;
    }
    return metrics;
}

ErrorMetrics Monitor::get_error_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    ErrorMetrics metrics;
    for (const auto& error : error_samples_) {
        metrics.error_counts[error]++;
        if (error.find("timeout") != std::string::npos) {
            metrics.num_timeouts++;
        } else if (error.find("out of memory") != std::string::npos) {
            metrics.num_oom_errors++;
        }
    }
    
    metrics.num_errors = error_samples_.size();
    metrics.error_types.assign(error_samples_.begin(), error_samples_.end());
    
    return metrics;
}

void Monitor::set_alert_threshold(const std::string& metric, double threshold) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    alert_thresholds_[metric] = threshold;
}

void Monitor::check_alerts() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    // Check hardware metrics
    if (hardware_metrics_.gpu_temperature > alert_thresholds_["gpu_temperature"]) {
        handle_threshold_violation("gpu_temperature",
                                hardware_metrics_.gpu_temperature,
                                alert_thresholds_["gpu_temperature"]);
    }
    
    if (hardware_metrics_.gpu_memory_util > alert_thresholds_["gpu_memory_util"]) {
        handle_threshold_violation("gpu_memory_util",
                                hardware_metrics_.gpu_memory_util,
                                alert_thresholds_["gpu_memory_util"]);
    }
    
    // Check latency metrics
    if (!latency_samples_.empty()) {
        double p99_latency = latency_samples_[static_cast<size_t>(latency_samples_.size() * 0.99)];
        if (p99_latency > alert_thresholds_["p99_latency"]) {
            handle_threshold_violation("p99_latency",
                                    p99_latency,
                                    alert_thresholds_["p99_latency"]);
        }
    }
}

void Monitor::update_config(const MonitorConfig& config) {
    if (initialized_) {
        throw std::runtime_error("Cannot update config while monitor is running");
    }
    config_ = config;
}

void Monitor::init_gpu_monitoring() {
    // Initialize NVML
    if (nvmlInit() != NVML_SUCCESS) {
        throw std::runtime_error("Failed to initialize NVML");
    }
}

void Monitor::init_memory_monitoring() {
    // TODO: Initialize memory monitoring
}

void Monitor::monitor_thread_func() {
    while (!should_stop_) {
        // Update hardware metrics
        update_hardware_metrics();
        
        // Check for alerts
        check_alerts();
        
        // Cleanup old metrics
        cleanup_old_metrics();
        
        // Sleep for sampling interval
        std::this_thread::sleep_for(config_.sampling_interval);
    }
}

void Monitor::cleanup_old_metrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    // Cleanup latency samples
    while (latency_samples_.size() > config_.metrics_history_size) {
        latency_samples_.erase(latency_samples_.begin());
    }
    
    // Cleanup throughput samples
    while (throughput_samples_.size() > config_.metrics_history_size) {
        throughput_samples_.erase(throughput_samples_.begin());
    }
    
    // Cleanup error samples
    while (error_samples_.size() > config_.metrics_history_size) {
        error_samples_.erase(error_samples_.begin());
    }
}

void Monitor::calculate_latency_percentiles() {
    if (latency_samples_.empty()) return;
    
    // Sort samples for percentile calculation
    std::vector<double> sorted_samples = latency_samples_;
    std::sort(sorted_samples.begin(), sorted_samples.end());
    
    size_t n = sorted_samples.size();
    auto get_percentile = [&](double p) {
        size_t idx = static_cast<size_t>(p * n);
        return sorted_samples[idx];
    };
    
    // Calculate percentiles
    auto metrics = get_latency_metrics();
    metrics.p50_latency_ms = get_percentile(0.50);
    metrics.p90_latency_ms = get_percentile(0.90);
    metrics.p95_latency_ms = get_percentile(0.95);
    metrics.p99_latency_ms = get_percentile(0.99);
}

void Monitor::calculate_throughput_statistics() {
    if (throughput_samples_.empty()) return;
    
    // Calculate moving average
    size_t window_size = std::min(throughput_samples_.size(), size_t(10));
    auto end = throughput_samples_.end();
    auto start = end - window_size;
    
    double sum = std::accumulate(start, end, 0.0);
    double avg = sum / window_size;
    
    auto metrics = get_throughput_metrics();
    metrics.operations_per_second = avg;
}

void Monitor::analyze_error_patterns() {
    if (error_samples_.empty()) return;
    
    // Count error frequencies
    std::unordered_map<std::string, size_t> error_counts;
    for (const auto& error : error_samples_) {
        error_counts[error]++;
    }
    
    // Detect error patterns
    for (const auto& [error_type, count] : error_counts) {
        if (count > error_samples_.size() / 2) {
            // More than 50% of recent errors are of this type
            std::stringstream ss;
            ss << "High frequency of error type: " << error_type
               << " (" << count << " occurrences)";
            log_alert(ss.str());
        }
    }
}

void Monitor::handle_threshold_violation(
    const std::string& metric,
    double value,
    double threshold) {
    
    std::stringstream ss;
    ss << "Alert: " << metric << " threshold violated. "
       << "Current value: " << value
       << ", Threshold: " << threshold;
    log_alert(ss.str());
}

void Monitor::log_alert(const std::string& message) {
    // TODO: Implement proper logging system
    std::cerr << "[ALERT] " << message << std::endl;
    
    // Write to log file
    std::ofstream log_file("monitor.log", std::ios::app);
    if (log_file) {
        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        log_file << std::ctime(&now_c) << message << std::endl;
    }
}

} // namespace deeppowers 