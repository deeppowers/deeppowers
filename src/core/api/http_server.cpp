#include "http_server.hpp"
#include "../common/logging.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <sstream>

namespace deeppowers {

using json = nlohmann::json;

HTTPServer::HTTPServer(
    std::shared_ptr<GPTModel> model,
    std::shared_ptr<Monitor> monitor,
    std::shared_ptr<Scheduler> scheduler,
    const HTTPServerConfig& config)
    : model_(model)
    , monitor_(monitor)
    , scheduler_(scheduler)
    , config_(config)
    , server_(std::make_unique<httplib::Server>()) {
    
    if (!model_) {
        throw std::runtime_error("Model cannot be null");
    }
    if (!monitor_) {
        throw std::runtime_error("Monitor cannot be null");
    }
    if (!scheduler_) {
        throw std::runtime_error("Scheduler cannot be null");
    }
    
    // Set up request handlers
    server_->Post("/v1/generate", [this](const auto& req, auto& res) {
        handle_generate(req, res);
    });
    
    server_->Get("/v1/metrics", [this](const auto& req, auto& res) {
        handle_metrics(req, res);
    });
    
    server_->Get("/health", [this](const auto& req, auto& res) {
        handle_health(req, res);
    });
    
    // Set up middleware
    server_->set_pre_routing_handler([this](const auto& req, auto& res) {
        return validate_request(req);
    });
    
    // Set up error handler
    server_->set_error_handler([this](const auto& req, auto& res) {
        send_error(res, 500, "Internal server error");
    });
    
    // Set up exception handler
    server_->set_exception_handler([this](const auto& req, auto& res, std::exception_ptr ep) {
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception& e) {
            log_error(e.what());
            send_error(res, 500, e.what());
        }
    });
}

HTTPServer::~HTTPServer() {
    stop();
}

void HTTPServer::start() {
    if (running_) {
        return;
    }
    
    // Initialize SSL if enabled
    if (config_.enable_ssl) {
        if (!server_->set_ssl_cert(config_.ssl_cert.c_str(), config_.ssl_key.c_str())) {
            throw std::runtime_error("Failed to set SSL certificate");
        }
    }
    
    // Start server
    running_ = true;
    if (!server_->listen(config_.host.c_str(), config_.port)) {
        running_ = false;
        throw std::runtime_error("Failed to start server");
    }
}

void HTTPServer::stop() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    server_->stop();
}

void HTTPServer::update_config(const HTTPServerConfig& config) {
    bool was_running = running_;
    if (was_running) {
        stop();
    }
    
    config_ = config;
    
    if (was_running) {
        start();
    }
}

void HTTPServer::handle_generate(const httplib::Request& req, httplib::Response& res) {
    auto start_time = std::chrono::steady_clock::now();
    
    // Check rate limit
    if (!check_rate_limit(req.remote_addr)) {
        send_error(res, 429, "Rate limit exceeded");
        return;
    }
    
    try {
        // Parse request
        json request = json::parse(req.body);
        
        // Validate request parameters
        if (!request.contains("prompt")) {
            send_error(res, 400, "Missing prompt parameter");
            return;
        }
        
        // Extract parameters
        std::string prompt = request["prompt"];
        size_t max_tokens = request.value("max_tokens", 100);
        float temperature = request.value("temperature", 0.7f);
        float top_p = request.value("top_p", 1.0f);
        float top_k = request.value("top_k", 0.0f);
        
        // Create generation request
        auto gen_request = std::make_shared<Request>(
            "req_" + std::to_string(metrics_.total_requests),
            prompt,
            RequestPriority::NORMAL);
        gen_request->mutable_config().max_tokens = max_tokens;
        gen_request->mutable_config().temperature = temperature;
        gen_request->mutable_config().top_p = top_p;
        gen_request->mutable_config().top_k = top_k;
        
        // Submit request to scheduler
        scheduler_->submit_request(gen_request);
        
        // Wait for completion
        while (gen_request->status() != RequestStatus::COMPLETED &&
               gen_request->status() != RequestStatus::FAILED) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // Prepare response
        json response;
        if (gen_request->status() == RequestStatus::COMPLETED) {
            response["text"] = gen_request->result().generated_texts[0];
            response["logprobs"] = gen_request->result().logprobs;
            response["tokens"] = gen_request->result().top_tokens[0];
        } else {
            send_error(res, 500, gen_request->result().error_message);
            return;
        }
        
        // Send response
        res.set_content(response.dump(), "application/json");
        
        // Update metrics
        auto end_time = std::chrono::steady_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        update_metrics("/v1/generate", latency);
        
    } catch (const json::exception& e) {
        send_error(res, 400, "Invalid JSON: " + std::string(e.what()));
    } catch (const std::exception& e) {
        send_error(res, 500, e.what());
    }
}

void HTTPServer::handle_metrics(const httplib::Request& req, httplib::Response& res) {
    json metrics;
    
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics["total_requests"] = metrics_.total_requests;
        metrics["failed_requests"] = metrics_.failed_requests;
        metrics["total_tokens"] = metrics_.total_tokens;
        metrics["avg_latency_ms"] = metrics_.avg_latency_ms;
    }
    
    // Add model metrics
    metrics["model"] = {
        {"gpu_utilization", monitor_->get_hardware_metrics().gpu_utilization},
        {"memory_usage_mb", monitor_->get_hardware_metrics().gpu_memory_used_mb},
        {"tokens_per_second", monitor_->get_throughput_metrics().tokens_per_second}
    };
    
    // Add scheduler metrics
    metrics["scheduler"] = {
        {"active_requests", scheduler_->get_stats().active_requests},
        {"total_processed", scheduler_->get_stats().total_requests},
        {"dropped_requests", scheduler_->get_stats().dropped_requests}
    };
    
    res.set_content(metrics.dump(), "application/json");
}

void HTTPServer::handle_health(const httplib::Request& req, httplib::Response& res) {
    json health;
    health["status"] = "healthy";
    health["version"] = "1.0.0";
    health["uptime_seconds"] = 0;  // TODO: Add uptime tracking
    
    res.set_content(health.dump(), "application/json");
}

bool HTTPServer::validate_request(const httplib::Request& req) {
    // Check request size
    if (req.body.size() > config_.max_request_size) {
        return false;
    }
    
    // Check content type for POST requests
    if (req.method == "POST" && 
        req.get_header_value("Content-Type") != "application/json") {
        return false;
    }
    
    return true;
}

bool HTTPServer::check_rate_limit(const std::string& client_ip) {
    std::lock_guard<std::mutex> lock(rate_limit_mutex_);
    
    auto now = std::chrono::steady_clock::now();
    auto& limit_info = rate_limits_[client_ip];
    
    // Reset counter if time window has passed
    if (now - limit_info.last_reset > std::chrono::seconds(60)) {
        limit_info.request_count = 0;
        limit_info.last_reset = now;
    }
    
    // Check rate limit
    if (limit_info.request_count >= 100) {  // 100 requests per minute
        return false;
    }
    
    limit_info.request_count++;
    return true;
}

void HTTPServer::update_metrics(const std::string& endpoint, size_t latency_ms) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    metrics_.total_requests++;
    metrics_.avg_latency_ms = 
        (metrics_.avg_latency_ms * (metrics_.total_requests - 1) + latency_ms) /
        metrics_.total_requests;
}

void HTTPServer::send_error(httplib::Response& res, int status, const std::string& message) {
    json error;
    error["error"] = {
        {"code", status},
        {"message", message}
    };
    
    res.status = status;
    res.set_content(error.dump(), "application/json");
    
    // Update metrics
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_.failed_requests++;
}

void HTTPServer::log_error(const std::string& error) {
    LOG_ERROR(error);
}

} // namespace deeppowers 