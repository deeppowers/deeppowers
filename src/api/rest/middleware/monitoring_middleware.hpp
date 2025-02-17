#pragma once

#include <httplib.h>
#include <string>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <atomic>
#include <vector>

namespace deeppowers {
namespace api {
namespace rest {

// Monitoring configuration
struct MonitoringConfig {
    bool enable_metrics = true;             // Whether to enable metrics collection
    bool enable_tracing = true;            // Whether to enable request tracing
    bool enable_alerts = true;             // Whether to enable alerting
    size_t metrics_window_size = 60;       // Metrics window size in seconds
    size_t max_traces_per_minute = 1000;   // Maximum traces to collect per minute
    double error_threshold = 0.05;         // Error rate threshold for alerts
    double latency_threshold_ms = 1000.0;  // Latency threshold for alerts
};

// Monitoring metrics
struct Metrics {
    // Request metrics
    std::atomic<size_t> total_requests{0};
    std::atomic<size_t> active_requests{0};
    std::atomic<size_t> successful_requests{0};
    std::atomic<size_t> failed_requests{0};
    
    // Latency metrics
    std::atomic<double> total_latency_ms{0.0};
    std::atomic<double> min_latency_ms{std::numeric_limits<double>::max()};
    std::atomic<double> max_latency_ms{0.0};
    
    // Status code metrics
    std::unordered_map<int, std::atomic<size_t>> status_codes;
    
    // Endpoint metrics
    struct EndpointMetrics {
        std::atomic<size_t> request_count{0};
        std::atomic<size_t> error_count{0};
        std::atomic<double> total_latency_ms{0.0};
    };
    std::unordered_map<std::string, EndpointMetrics> endpoints;
};

// Trace information
struct Trace {
    std::string request_id;
    std::string path;
    std::string method;
    std::chrono::system_clock::time_point start_time;
    std::chrono::microseconds duration;
    int status_code;
    std::string error;
    std::vector<std::pair<std::string, std::chrono::microseconds>> spans;
};

// Alert information
struct Alert {
    enum class Type {
        ERROR_RATE,
        HIGH_LATENCY,
        HIGH_MEMORY,
        HIGH_CPU
    };
    
    Type type;
    std::string message;
    double threshold;
    double current_value;
    std::chrono::system_clock::time_point timestamp;
};

// Monitoring middleware class
class MonitoringMiddleware {
public:
    explicit MonitoringMiddleware(const MonitoringConfig& config = MonitoringConfig());
    
    // Request monitoring
    void start_request(const httplib::Request& req);
    void end_request(const httplib::Request& req,
                    const httplib::Response& res,
                    const std::chrono::microseconds& duration);
    
    // Trace management
    void start_span(const std::string& request_id, const std::string& name);
    void end_span(const std::string& request_id, const std::string& name);
    
    // Metrics access
    Metrics get_current_metrics() const;
    std::vector<Trace> get_recent_traces() const;
    std::vector<Alert> get_recent_alerts() const;
    
    // Configuration access
    const MonitoringConfig& config() const { return config_; }
    void update_config(const MonitoringConfig& config);

private:
    // Internal helper methods
    void update_metrics(const httplib::Request& req,
                       const httplib::Response& res,
                       const std::chrono::microseconds& duration);
    void check_alerts();
    void cleanup_old_data();
    
    // Metrics management
    struct TimeWindow {
        std::chrono::system_clock::time_point start_time;
        Metrics metrics;
    };
    
    // Member variables
    MonitoringConfig config_;
    std::vector<TimeWindow> metrics_windows_;
    std::vector<Trace> traces_;
    std::vector<Alert> alerts_;
    std::mutex metrics_mutex_;
    std::mutex traces_mutex_;
    std::mutex alerts_mutex_;
    
    // Active spans tracking
    struct SpanInfo {
        std::string name;
        std::chrono::system_clock::time_point start_time;
    };
    std::unordered_map<std::string, std::vector<SpanInfo>> active_spans_;
    std::mutex spans_mutex_;
};

// Global middleware instance
extern std::unique_ptr<MonitoringMiddleware> monitoring_middleware_instance;

// Middleware functions
void start_request_monitoring(const httplib::Request& req);
void end_request_monitoring(const httplib::Request& req,
                          const httplib::Response& res,
                          const std::chrono::microseconds& duration);

} // namespace rest
} // namespace api
} // namespace deeppowers 