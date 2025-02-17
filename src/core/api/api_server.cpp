#include "api_server.hpp"
#include <nlohmann/json.hpp>
#include <sstream>
#include <chrono>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <cpprest/http_listener.h>
#include <cpprest/json.h>

using json = nlohmann::json;
using namespace web;
using namespace web::http;
using namespace web::http::experimental::listener;

namespace deeppowers {

APIServer::APIServer(
    std::shared_ptr<GPTModel> model,
    std::shared_ptr<Monitor> monitor,
    std::shared_ptr<Scheduler> scheduler,
    const APIServerConfig& config)
    : model_(model)
    , monitor_(monitor)
    , scheduler_(scheduler)
    , config_(config) {
}

APIServer::~APIServer() {
    if (running_) {
        stop();
    }
}

void APIServer::start() {
    if (running_) return;
    
    // Initialize server
    init_server();
    
    // Initialize SSL if enabled
    if (config_.enable_ssl) {
        init_ssl();
    }
    
    // Start worker threads
    worker_threads_.resize(config_.num_threads);
    for (size_t i = 0; i < config_.num_threads; ++i) {
        worker_threads_[i] = std::thread(&APIServer::worker_thread_func, this);
    }
    
    running_ = true;
}

void APIServer::stop() {
    if (!running_) return;
    
    running_ = false;
    
    // Stop worker threads
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
}

APIResponse APIServer::handle_request(const APIRequest& request) {
    // Validate request
    if (!validate_request(request)) {
        return create_error_response(400, "Invalid request");
    }
    
    // Check authentication if required
    if (!check_auth(request)) {
        return create_error_response(401, "Unauthorized");
    }
    
    // Check rate limit
    if (!check_rate_limit(request)) {
        return create_error_response(429, "Too many requests");
    }
    
    try {
        // Route request based on path
        if (request.path.find("/model") == 0) {
            return process_model_request(request);
        } else if (request.path.find("/monitor") == 0) {
            return process_monitor_request(request);
        } else if (request.path.find("/scheduler") == 0) {
            return process_scheduler_request(request);
        } else {
            return create_error_response(404, "Endpoint not found");
        }
    } catch (const std::exception& e) {
        log_error(e.what());
        return create_error_response(500, "Internal server error");
    }
}

void APIServer::register_endpoint(const EndpointConfig& config) {
    std::lock_guard<std::mutex> lock(server_mutex_);
    endpoints_[config.path] = config;
}

void APIServer::remove_endpoint(const std::string& path) {
    std::lock_guard<std::mutex> lock(server_mutex_);
    endpoints_.erase(path);
}

void APIServer::set_auth_handler(std::function<bool(const APIRequest&)> handler) {
    std::lock_guard<std::mutex> lock(server_mutex_);
    auth_handler_ = handler;
}

void APIServer::set_rate_limiter(std::function<bool(const std::string&)> limiter) {
    std::lock_guard<std::mutex> lock(server_mutex_);
    rate_limiter_ = limiter;
}

Monitor::ComponentMetrics APIServer::get_api_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return api_metrics_;
}

void APIServer::update_metrics(const Monitor::ComponentMetrics& metrics) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    api_metrics_ = metrics;
}

void APIServer::update_config(const APIServerConfig& config) {
    if (running_) {
        throw std::runtime_error("Cannot update config while server is running");
    }
    config_ = config;
}

void APIServer::init_server() {
    // Create HTTP listener
    std::string address = "http://" + config_.host + ":" + std::to_string(config_.port);
    http_listener_config listener_config;
    listener_config.set_timeout(std::chrono::seconds(30));
    
    // Register default endpoints
    register_endpoint({"POST", "/model/generate", true, 10});
    register_endpoint({"GET", "/monitor/metrics", false, 100});
    register_endpoint({"GET", "/scheduler/status", false, 100});
}

void APIServer::init_ssl() {
    // Initialize OpenSSL
    SSL_load_error_strings();
    OpenSSL_add_ssl_algorithms();
    
    // Create SSL context
    SSL_CTX* ctx = SSL_CTX_new(TLS_server_method());
    if (!ctx) {
        throw std::runtime_error("Failed to create SSL context");
    }
    
    // Load certificate and private key
    if (SSL_CTX_use_certificate_file(ctx, config_.ssl_cert.c_str(), SSL_FILETYPE_PEM) <= 0) {
        throw std::runtime_error("Failed to load SSL certificate");
    }
    if (SSL_CTX_use_PrivateKey_file(ctx, config_.ssl_key.c_str(), SSL_FILETYPE_PEM) <= 0) {
        throw std::runtime_error("Failed to load SSL private key");
    }
}

void APIServer::worker_thread_func() {
    while (running_) {
        // TODO: Implement request processing loop
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

bool APIServer::validate_request(const APIRequest& request) {
    // Check request size
    if (request.body.size() > config_.max_request_size) {
        return false;
    }
    
    // Check endpoint exists
    auto it = endpoints_.find(request.path);
    if (it == endpoints_.end()) {
        return false;
    }
    
    // Check HTTP method
    if (it->second.method != request.method) {
        return false;
    }
    
    return true;
}

bool APIServer::check_auth(const APIRequest& request) {
    auto it = endpoints_.find(request.path);
    if (it == endpoints_.end()) {
        return false;
    }
    
    if (!it->second.require_auth) {
        return true;
    }
    
    if (!auth_handler_) {
        return false;
    }
    
    return auth_handler_(request);
}

bool APIServer::check_rate_limit(const APIRequest& request) {
    auto it = endpoints_.find(request.path);
    if (it == endpoints_.end() || it->second.rate_limit == 0) {
        return true;
    }
    
    std::lock_guard<std::mutex> lock(rate_limit_mutex_);
    
    auto& limit_info = rate_limits_[request.path];
    auto now = std::chrono::steady_clock::now();
    
    // Reset counter if window has passed
    if (now - limit_info.last_reset >= std::chrono::seconds(1)) {
        limit_info.request_count = 0;
        limit_info.last_reset = now;
    }
    
    // Check limit
    if (limit_info.request_count >= it->second.rate_limit) {
        return false;
    }
    
    limit_info.request_count++;
    return true;
}

APIResponse APIServer::process_model_request(const APIRequest& request) {
    try {
        // Parse request body
        json request_json = json::parse(request.body);
        
        // Extract parameters
        std::string prompt = request_json["prompt"];
        size_t max_tokens = request_json.value("max_tokens", 100);
        float temperature = request_json.value("temperature", 1.0f);
        float top_p = request_json.value("top_p", 1.0f);
        
        // Create model request
        Request model_request(
            request_json.value("request_id", ""),
            prompt,
            RequestPriority::NORMAL);
        model_request.mutable_config().max_tokens = max_tokens;
        
        // Submit request to scheduler
        scheduler_->submit_request(std::make_shared<Request>(model_request));
        
        // Create response
        json response_json = {
            {"status", "success"},
            {"request_id", model_request.id()}
        };
        
        APIResponse response;
        response.status_code = 200;
        response.body = response_json.dump();
        response.headers["Content-Type"] = "application/json";
        
        return response;
        
    } catch (const json::exception& e) {
        return create_error_response(400, "Invalid JSON: " + std::string(e.what()));
    } catch (const std::exception& e) {
        return create_error_response(500, "Internal error: " + std::string(e.what()));
    }
}

APIResponse APIServer::process_monitor_request(const APIRequest& request) {
    try {
        // Get metrics
        auto hardware_metrics = monitor_->get_hardware_metrics();
        auto latency_metrics = monitor_->get_latency_metrics();
        auto throughput_metrics = monitor_->get_throughput_metrics();
        auto error_metrics = monitor_->get_error_metrics();
        
        // Create response
        json response_json = {
            {"hardware", {
                {"gpu_util", hardware_metrics.gpu_compute_util},
                {"gpu_memory", hardware_metrics.gpu_memory_used_mb},
                {"gpu_temperature", hardware_metrics.gpu_temperature}
            }},
            {"latency", {
                {"avg_ms", latency_metrics.avg_latency_ms},
                {"p99_ms", latency_metrics.p99_latency_ms}
            }},
            {"throughput", {
                {"requests_per_second", throughput_metrics.requests_per_second},
                {"tokens_per_second", throughput_metrics.tokens_per_second}
            }},
            {"errors", {
                {"total", error_metrics.num_errors},
                {"timeouts", error_metrics.num_timeouts},
                {"oom", error_metrics.num_oom_errors}
            }}
        };
        
        APIResponse response;
        response.status_code = 200;
        response.body = response_json.dump();
        response.headers["Content-Type"] = "application/json";
        
        return response;
        
    } catch (const std::exception& e) {
        return create_error_response(500, "Internal error: " + std::string(e.what()));
    }
}

APIResponse APIServer::process_scheduler_request(const APIRequest& request) {
    try {
        // Get scheduler stats
        const auto& stats = scheduler_->get_stats();
        
        // Create response
        json response_json = {
            {"active_requests", stats.active_requests},
            {"total_requests", stats.total_requests},
            {"dropped_requests", stats.dropped_requests},
            {"avg_latency_ms", stats.avg_latency_ms},
            {"avg_throughput", stats.avg_throughput}
        };
        
        APIResponse response;
        response.status_code = 200;
        response.body = response_json.dump();
        response.headers["Content-Type"] = "application/json";
        
        return response;
        
    } catch (const std::exception& e) {
        return create_error_response(500, "Internal error: " + std::string(e.what()));
    }
}

APIResponse APIServer::create_error_response(int status_code, const std::string& message) {
    json error_json = {
        {"status", "error"},
        {"code", status_code},
        {"message", message}
    };
    
    APIResponse response;
    response.status_code = status_code;
    response.body = error_json.dump();
    response.headers["Content-Type"] = "application/json";
    
    return response;
}

void APIServer::log_error(const std::string& error) {
    // Update error metrics
    monitor_->record_error("api_server", error);
    
    // Log error message
    std::cerr << "[ERROR] " << error << std::endl;
}

} // namespace deeppowers