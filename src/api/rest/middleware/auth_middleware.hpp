#pragma once

#include <httplib.h>
#include <string>
#include <unordered_map>
#include <mutex>

namespace deeppowers {
namespace api {
namespace rest {

// Authentication configuration
struct AuthConfig {
    bool enable_auth = true;                // Whether to enable authentication
    std::string auth_type = "bearer";       // Authentication type (bearer/basic)
    std::string auth_header = "Authorization"; // Authentication header name
    size_t token_expire_hours = 24;         // Token expiration time (hours)
    size_t max_tokens_per_user = 5;         // Maximum number of tokens per user
};

// Authentication middleware class
class AuthMiddleware {
public:
    explicit AuthMiddleware(const AuthConfig& config = AuthConfig());
    
    // Authentication check
    bool authenticate(const httplib::Request& req, httplib::Response& res);
    
    // Token management
    std::string generate_token(const std::string& user_id);
    bool validate_token(const std::string& token);
    void revoke_token(const std::string& token);
    
    // User management
    bool add_user(const std::string& user_id, const std::string& api_key);
    void remove_user(const std::string& user_id);
    
    // Configuration access
    const AuthConfig& config() const { return config_; }
    void update_config(const AuthConfig& config);

private:
    // Internal helper methods
    bool verify_bearer_token(const std::string& token);
    bool verify_basic_auth(const std::string& credentials);
    void cleanup_expired_tokens();
    
    // Token information
    struct TokenInfo {
        std::string user_id;
        std::chrono::system_clock::time_point created_at;
        std::chrono::system_clock::time_point expires_at;
        bool is_revoked = false;
    };
    
    // Member variables
    AuthConfig config_;
    std::unordered_map<std::string, std::string> users_;  // user_id -> api_key
    std::unordered_map<std::string, TokenInfo> tokens_;   // token -> info
    std::mutex auth_mutex_;
};

// Global middleware instance
extern std::unique_ptr<AuthMiddleware> auth_middleware_instance;

// Middleware function
bool auth_middleware(const httplib::Request& req, httplib::Response& res);

} // namespace rest
} // namespace api
} // namespace deeppowers