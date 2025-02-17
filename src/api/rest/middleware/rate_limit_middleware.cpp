#include "rate_limit_middleware.hpp"
#include "../../../common/logging.hpp"
#include <nlohmann/json.hpp>

namespace deeppowers {
namespace api {
namespace rest {

using json = nlohmann::json;

// Global instance
std::unique_ptr<RateLimitMiddleware> rate_limit_middleware_instance;

RateLimitMiddleware::RateLimitMiddleware(const RateLimitConfig& config)
    : config_(config) {
}

bool RateLimitMiddleware::check_rate_limit(const httplib::Request& req, httplib::Response& res) {
    if (!config_.enable_rate_limit) {
        return true;
    }
    
    const std::string& ip = req.remote_addr;
    
    // Check whitelist
    if (config_.enable_ip_whitelist && is_whitelisted(ip)) {
        return true;
    }
    
    std::lock_guard<std::mutex> lock(rate_mutex_);
    
    // Clean up expired records
    cleanup_expired_records();
    
    auto now = std::chrono::steady_clock::now();
    auto& record = rate_records_[ip];
    
    // Initialize new record
    if (record.request_count == 0) {
        record.window_start = now;
        record.tokens = config_.burst_size;
    }
    
    // Check if time window is reset
    auto window_duration = std::chrono::duration_cast<std::chrono::seconds>(
        now - record.window_start).count();
    if (window_duration >= 60) {
        record.request_count = 0;
        record.window_start = now;
        record.tokens = config_.burst_size;
    }
    
    // Add tokens
    if (record.last_request.time_since_epoch().count() > 0) {
        auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - record.last_request).count();
        size_t new_tokens = (time_since_last * config_.requests_per_minute) / (60 * 1000);
        record.tokens = std::min(record.tokens + new_tokens, config_.burst_size);
    }
    
    // Check if limit is exceeded
    if (record.request_count >= config_.requests_per_minute || record.tokens == 0) {
        json error = {
            {"error", {
                {"code", 429},
                {"message", "Rate limit exceeded"},
                {"retry_after", 60 - window_duration}
            }}
        };
        res.status = 429;
        res.set_content(error.dump(), "application/json");
        return false;
    }
    
    // Update record
    record.request_count++;
    record.last_request = now;
    record.tokens--;
    
    return true;
}

void RateLimitMiddleware::add_to_whitelist(const std::string& ip) {
    std::lock_guard<std::mutex> lock(rate_mutex_);
    
    if (std::find(config_.ip_whitelist.begin(), 
                  config_.ip_whitelist.end(), ip) == config_.ip_whitelist.end()) {
        config_.ip_whitelist.push_back(ip);
    }
}

void RateLimitMiddleware::remove_from_whitelist(const std::string& ip) {
    std::lock_guard<std::mutex> lock(rate_mutex_);
    
    auto it = std::find(config_.ip_whitelist.begin(), 
                       config_.ip_whitelist.end(), ip);
    if (it != config_.ip_whitelist.end()) {
        config_.ip_whitelist.erase(it);
    }
}

void RateLimitMiddleware::reset_limits(const std::string& ip) {
    std::lock_guard<std::mutex> lock(rate_mutex_);
    rate_records_.erase(ip);
}

void RateLimitMiddleware::update_config(const RateLimitConfig& config) {
    std::lock_guard<std::mutex> lock(rate_mutex_);
    config_ = config;
}

bool RateLimitMiddleware::is_whitelisted(const std::string& ip) const {
    return std::find(config_.ip_whitelist.begin(), 
                    config_.ip_whitelist.end(), ip) != config_.ip_whitelist.end();
}

void RateLimitMiddleware::cleanup_expired_records() {
    auto now = std::chrono::steady_clock::now();
    
    for (auto it = rate_records_.begin(); it != rate_records_.end();) {
        auto window_duration = std::chrono::duration_cast<std::chrono::seconds>(
            now - it->second.window_start).count();
        if (window_duration >= 120) {  // 2 minutes later cleanup
            it = rate_records_.erase(it);
        } else {
            ++it;
        }
    }
}

bool rate_limit_middleware(const httplib::Request& req, httplib::Response& res) {
    if (!rate_limit_middleware_instance) {
        rate_limit_middleware_instance = std::make_unique<RateLimitMiddleware>();
    }
    return rate_limit_middleware_instance->check_rate_limit(req, res);
}

} // namespace rest
} // namespace api
} // namespace deeppowers 