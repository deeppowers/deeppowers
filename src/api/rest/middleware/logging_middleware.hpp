#pragma once

#include <httplib.h>
#include <string>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <fstream>

namespace deeppowers {
namespace api {
namespace rest {

// Logging configuration
struct LoggingConfig {
    bool enable_request_logging = true;     // Whether to enable request logging
    bool enable_response_logging = true;    // Whether to enable response logging
    bool enable_performance_logging = true; // Whether to enable performance logging
    std::string log_format = "json";       // Log format (json/text)
    std::string log_path = "logs/api.log"; // Log file path
    size_t max_body_size = 1024;           // Maximum body size to log
    bool mask_sensitive_data = true;       // Whether to mask sensitive data
};

// Logging middleware class
class LoggingMiddleware {
public:
    explicit LoggingMiddleware(const LoggingConfig& config = LoggingConfig());
    ~LoggingMiddleware();

    // Request/Response logging
    void log_request(const httplib::Request& req);
    void log_response(const httplib::Request& req, 
                     const httplib::Response& res,
                     const std::chrono::microseconds& duration);
    
    // Performance logging
    void log_performance(const std::string& endpoint,
                        const std::chrono::microseconds& duration,
                        size_t request_size,
                        size_t response_size);
    
    // Configuration access
    const LoggingConfig& config() const { return config_; }
    void update_config(const LoggingConfig& config);

private:
    // Internal helper methods
    void open_log_file();
    void rotate_log_file();
    void write_log_entry(const std::string& entry);
    std::string format_log_entry(const std::string& type,
                                const std::unordered_map<std::string, std::string>& fields);
    std::string mask_sensitive_data(const std::string& data);
    
    // Log file management
    struct LogFile {
        std::ofstream file;
        size_t current_size = 0;
        std::chrono::system_clock::time_point last_rotation;
    };
    
    // Member variables
    LoggingConfig config_;
    LogFile log_file_;
    std::mutex log_mutex_;
    
    // Performance tracking
    struct EndpointStats {
        size_t request_count = 0;
        size_t total_duration_us = 0;
        size_t total_request_size = 0;
        size_t total_response_size = 0;
    };
    std::unordered_map<std::string, EndpointStats> endpoint_stats_;
};

// Global middleware instance
extern std::unique_ptr<LoggingMiddleware> logging_middleware_instance;

// Middleware functions
void log_request_middleware(const httplib::Request& req);
void log_response_middleware(const httplib::Request& req,
                           const httplib::Response& res,
                           const std::chrono::microseconds& duration);

} // namespace rest
} // namespace api
} // namespace deeppowers 