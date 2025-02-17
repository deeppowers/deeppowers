#pragma once

#include "../execution/models/gpt_model.hpp"
#include "../monitoring/monitor.hpp"
#include "../scheduling/scheduler.hpp"
#include <httplib.h>
#include <memory>
#include <string>
#include <thread>
#include <mutex>

namespace deeppowers {

// HTTP server configuration
struct HTTPServerConfig {
    std::string host = "0.0.0.0";     // Server host
    int port = 8080;                  // Server port
    size_t num_threads = 4;           // Number of worker threads
    bool enable_ssl = false;          // Enable HTTPS
    std::string ssl_cert;             // SSL certificate path
    std::string ssl_key;              // SSL private key path
    size_t max_request_size = 1024*1024;  // Maximum request size in bytes
    size_t request_timeout_ms = 30000;    // Request timeout in milliseconds
};

// HTTP server class
class HTTPServer {
public:
    explicit HTTPServer(
        std::shared_ptr<GPTModel> model,
        std::shared_ptr<Monitor> monitor,
        std::shared_ptr<Scheduler> scheduler,
        const HTTPServerConfig& config);
    ~HTTPServer();

    // Server lifecycle
    void start();
    void stop();
    bool is_running() const { return running_; }
    
    // Configuration
    const HTTPServerConfig& config() const { return config_; }
    void update_config(const HTTPServerConfig& config);

private:
    // Request handlers
    void handle_generate(const httplib::Request& req, httplib::Response& res);
    void handle_metrics(const httplib::Request& req, httplib::Response& res);
    void handle_health(const httplib::Request& req, httplib::Response& res);
    
    // Middleware
    bool validate_request(const httplib::Request& req);
    bool check_rate_limit(const std::string& client_ip);
    void update_metrics(const std::string& endpoint, size_t latency_ms);
    
    // Error handling
    void send_error(httplib::Response& res, int status, const std::string& message);
    void log_error(const std::string& error);
    
    // Rate limiting
    struct RateLimitInfo {
        size_t request_count = 0;
        std::chrono::steady_clock::time_point last_reset;
    };
    
    // Member variables
    std::shared_ptr<GPTModel> model_;
    std::shared_ptr<Monitor> monitor_;
    std::shared_ptr<Scheduler> scheduler_;
    HTTPServerConfig config_;
    bool running_ = false;
    
    // HTTP server
    std::unique_ptr<httplib::Server> server_;
    
    // Rate limiting state
    std::unordered_map<std::string, RateLimitInfo> rate_limits_;
    std::mutex rate_limit_mutex_;
    
    // Metrics
    struct APIMetrics {
        size_t total_requests = 0;
        size_t failed_requests = 0;
        size_t total_tokens = 0;
        double avg_latency_ms = 0.0;
    };
    APIMetrics metrics_;
    std::mutex metrics_mutex_;
};

} // namespace deeppowers 