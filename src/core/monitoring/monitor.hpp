#pragma once

#include "../hal/hal.hpp"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <mutex>

namespace deeppowers {

// Performance metrics for different components
struct ComponentMetrics {
    double compute_time_ms = 0.0;      // Computation time in milliseconds
    double memory_usage_mb = 0.0;      // Memory usage in MB
    double throughput = 0.0;           // Operations per second
    double utilization = 0.0;          // Resource utilization (0-1)
    size_t num_operations = 0;         // Number of operations processed
    size_t num_errors = 0;             // Number of errors encountered
};

// Hardware utilization metrics
struct HardwareMetrics {
    // GPU metrics
    double gpu_compute_util = 0.0;     // GPU compute utilization
    double gpu_memory_util = 0.0;      // GPU memory utilization
    double gpu_memory_used_mb = 0.0;   // GPU memory used in MB
    double gpu_temperature = 0.0;      // GPU temperature in Celsius
    double gpu_power_watts = 0.0;      // GPU power consumption in watts
    
    // Memory metrics
    double host_memory_used_mb = 0.0;  // Host memory used in MB
    double memory_bandwidth = 0.0;     // Memory bandwidth utilization
    
    // PCIe metrics
    double pcie_bandwidth = 0.0;       // PCIe bandwidth utilization
    size_t pcie_throughput = 0;        // PCIe throughput in bytes/sec
};

// Latency analysis metrics
struct LatencyMetrics {
    double avg_latency_ms = 0.0;       // Average latency in milliseconds
    double p50_latency_ms = 0.0;       // 50th percentile latency
    double p90_latency_ms = 0.0;       // 90th percentile latency
    double p95_latency_ms = 0.0;       // 95th percentile latency
    double p99_latency_ms = 0.0;       // 99th percentile latency
    double max_latency_ms = 0.0;       // Maximum latency
    double min_latency_ms = 0.0;       // Minimum latency
    std::vector<double> latency_histogram;  // Latency distribution histogram
};

// Throughput analysis metrics
struct ThroughputMetrics {
    double requests_per_second = 0.0;   // Request throughput
    double tokens_per_second = 0.0;     // Token throughput
    double bytes_per_second = 0.0;      // Data throughput
    double operations_per_second = 0.0;  // Operation throughput
    std::vector<double> throughput_history;  // Historical throughput data
};

// Error and warning metrics
struct ErrorMetrics {
    size_t num_errors = 0;             // Total number of errors
    size_t num_warnings = 0;           // Total number of warnings
    size_t num_timeouts = 0;           // Number of timeout errors
    size_t num_oom_errors = 0;         // Number of out-of-memory errors
    std::vector<std::string> error_types;  // Types of errors encountered
    std::unordered_map<std::string, size_t> error_counts;  // Error frequency
};

// Monitor configuration
struct MonitorConfig {
    bool enable_gpu_monitoring = true;      // Enable GPU monitoring
    bool enable_memory_monitoring = true;   // Enable memory monitoring
    bool enable_latency_tracking = true;    // Enable latency tracking
    bool enable_error_tracking = true;      // Enable error tracking
    size_t metrics_history_size = 1000;     // Number of historical metrics to keep
    std::chrono::milliseconds sampling_interval{100};  // Metrics sampling interval
};

// Performance monitoring and analysis system
class Monitor {
public:
    explicit Monitor(hal::Device* device, const MonitorConfig& config);
    ~Monitor();

    // Initialization and cleanup
    void initialize();
    void finalize();
    
    // Metrics collection
    void record_latency(const std::string& operation, double latency_ms);
    void record_throughput(const std::string& operation, double operations);
    void record_error(const std::string& operation, const std::string& error_type);
    void update_component_metrics(const std::string& component, const ComponentMetrics& metrics);
    
    // Hardware monitoring
    void update_hardware_metrics();
    void track_memory_usage(size_t bytes_allocated, size_t bytes_freed);
    void track_gpu_utilization();
    
    // Analysis and reporting
    ComponentMetrics get_component_metrics(const std::string& component) const;
    HardwareMetrics get_hardware_metrics() const;
    LatencyMetrics get_latency_metrics() const;
    ThroughputMetrics get_throughput_metrics() const;
    ErrorMetrics get_error_metrics() const;
    
    // Alerting and warnings
    void set_alert_threshold(const std::string& metric, double threshold);
    void check_alerts();
    
    // Configuration
    const MonitorConfig& config() const { return config_; }
    void update_config(const MonitorConfig& config);

private:
    // Internal helper methods
    void init_gpu_monitoring();
    void init_memory_monitoring();
    void monitor_thread_func();
    void cleanup_old_metrics();
    
    // Metrics calculation
    void calculate_latency_percentiles();
    void calculate_throughput_statistics();
    void analyze_error_patterns();
    
    // Alert handling
    void handle_threshold_violation(const std::string& metric, double value, double threshold);
    void log_alert(const std::string& message);
    
    // Member variables
    hal::Device* device_;
    MonitorConfig config_;
    bool initialized_ = false;
    bool should_stop_ = false;
    
    // Metrics storage
    std::unordered_map<std::string, ComponentMetrics> component_metrics_;
    HardwareMetrics hardware_metrics_;
    std::vector<double> latency_samples_;
    std::vector<double> throughput_samples_;
    std::vector<std::string> error_samples_;
    
    // Alert configuration
    std::unordered_map<std::string, double> alert_thresholds_;
    
    // Synchronization
    mutable std::mutex metrics_mutex_;
    std::thread monitor_thread_;
};

} // namespace deeppowers 