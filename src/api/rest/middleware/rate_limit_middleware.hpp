#pragma once

#include <httplib.h>
#include <string>
#include <unordered_map>
#include <mutex>
#include <chrono>

namespace deeppowers {
namespace api {
namespace rest {

// Rate limit configuration
struct RateLimitConfig {
    bool enable_rate_limit = true;          // Whether to enable rate limiting
    size_t requests_per_minute = 60;        // Requests per minute limit
    size_t burst_size = 10;                // Burst request limit
    bool enable_ip_whitelist = false;       // Whether to enable IP whitelist
    std::vector<std::string> ip_whitelist;  // IP whitelist
};

// Rate limit middleware class
class RateLimitMiddleware {
public:
    explicit RateLimitMiddleware(const RateLimitConfig& config = RateLimitConfig());
    
    // Rate check
    bool check_rate_limit(const httplib::Request& req, httplib::Response& res);
    
    // Limit management
    void add_to_whitelist(const std::string& ip);
    void remove_from_whitelist(const std::string& ip);
    void reset_limits(const std::string& ip);
    
    // Configuration access
    const RateLimitConfig& config() const { return config_; }
    void update_config(const RateLimitConfig& config);

private:
    // Internal helper methods
    bool is_whitelisted(const std::string& ip) const;
    void cleanup_expired_records();
    
    // Rate limit records
    struct RateRecord {
        size_t request_count = 0;
        std::chrono::steady_clock::time_point last_request;
        std::chrono::steady_clock::time_point window_start;
        size_t tokens = 0;  // Token count for token bucket algorithm
    };
    
    // Member variables
    RateLimitConfig config_;
    std::unordered_map<std::string, RateRecord> rate_records_;
    std::mutex rate_mutex_;
};

// Global middleware instance
extern std::unique_ptr<RateLimitMiddleware> rate_limit_middleware_instance;

// Middleware function
bool rate_limit_middleware(const httplib::Request& req, httplib::Response& res);

} // namespace rest
} // namespace api
} // namespace deeppowers 