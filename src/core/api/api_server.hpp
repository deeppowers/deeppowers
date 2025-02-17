#pragma once

#include "../execution/models/gpt_model.hpp"
#include "../monitoring/monitor.hpp"
#include "../scheduling/scheduler.hpp"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>

namespace deeppowers {

// API endpoint configuration
struct EndpointConfig {
    std::string path;                  // Endpoint path
    std::string method;                // HTTP method (GET, POST, etc.)
    bool require_auth = false;         // Whether authentication is required
    size_t rate_limit = 0;            // Rate limit (requests per second, 0 for unlimited)
    size_t timeout_ms = 30000;        // Request timeout in milliseconds
};

// API server configuration
struct APIServerConfig {
    std::string host = "0.0.0.0";     // Server host
    int port = 8080;                  // Server port
    size_t num_threads = 4;           // Number of worker threads
    bool enable_ssl = false;          // Enable HTTPS
    std::string ssl_cert;             // SSL certificate path
    std::string ssl_key;              // SSL private key path
    size_t max_request_size = 1024*1024;  // Maximum request size in bytes
    std::vector<EndpointConfig> endpoints;  // API endpoints
};

// API request context
struct APIRequest {
    std::string method;               // HTTP method
    std::string path;                 // Request path
    std::string body;                 // Request body
    std::unordered_map<std::string, std::string> headers;  // Request headers
    std::unordered_map<std::string, std::string> params;   // Query parameters
};

// API response
struct APIResponse {
    int status_code = 200;            // HTTP status code
    std::string body;                 // Response body
    std::unordered_map<std::string, std::string> headers;  // Response headers
};

// API server class
class APIServer {
public:
    explicit APIServer(
        std::shared_ptr<GPTModel> model,
        std::shared_ptr<Monitor> monitor,
        std::shared_ptr<Scheduler> scheduler,
        const APIServerConfig& config);
    ~APIServer();

    // Server lifecycle
    void start();
    void stop();
    bool is_running() const { return running_; }
    
    // Request handling
    APIResponse handle_request(const APIRequest& request);
    
    // Endpoint management
    void register_endpoint(const EndpointConfig& config);
    void remove_endpoint(const std::string& path);
    
    // Authentication and authorization
    void set_auth_handler(std::function<bool(const APIRequest&)> handler);
    void set_rate_limiter(std::function<bool(const std::string&)> limiter);
    
    // Monitoring and metrics
    Monitor::ComponentMetrics get_api_metrics() const;
    void update_metrics(const Monitor::ComponentMetrics& metrics);
    
    // Configuration
    const APIServerConfig& config() const { return config_; }
    void update_config(const APIServerConfig& config);

private:
    // Internal helper methods
    void init_server();
    void init_ssl();
    void worker_thread_func();
    
    // Request processing
    bool validate_request(const APIRequest& request);
    bool check_auth(const APIRequest& request);
    bool check_rate_limit(const APIRequest& request);
    APIResponse process_model_request(const APIRequest& request);
    APIResponse process_monitor_request(const APIRequest& request);
    APIResponse process_scheduler_request(const APIRequest& request);
    
    // Error handling
    APIResponse create_error_response(int status_code, const std::string& message);
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
    APIServerConfig config_;
    bool running_ = false;
    
    // Thread management
    std::vector<std::thread> worker_threads_;
    std::mutex server_mutex_;
    
    // Request handling
    std::unordered_map<std::string, EndpointConfig> endpoints_;
    std::function<bool(const APIRequest&)> auth_handler_;
    std::function<bool(const std::string&)> rate_limiter_;
    
    // Rate limiting state
    std::unordered_map<std::string, RateLimitInfo> rate_limits_;
    std::mutex rate_limit_mutex_;
    
    // Metrics
    Monitor::ComponentMetrics api_metrics_;
    std::mutex metrics_mutex_;
};

} // namespace deeppowers 