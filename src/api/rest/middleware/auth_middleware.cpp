#include "auth_middleware.hpp"
#include "../../../common/logging.hpp"
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <openssl/hmac.h>
#include <random>
#include <iomanip>
#include <sstream>

namespace deeppowers {
namespace api {
namespace rest {

using json = nlohmann::json;

// Global instance
std::unique_ptr<AuthMiddleware> auth_middleware_instance;

// Helper function: Generate random string
std::string generate_random_string(size_t length) {
    static const char charset[] = "0123456789"
                                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                "abcdefghijklmnopqrstuvwxyz";
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, sizeof(charset) - 2);
    
    std::string str;
    str.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        str += charset[dis(gen)];
    }
    return str;
}

// Helper function: Compute HMAC-SHA256
std::string compute_hmac(const std::string& key, const std::string& message) {
    unsigned char hash[32];
    unsigned int hash_len;
    
    HMAC(EVP_sha256(), key.c_str(), key.length(),
         reinterpret_cast<const unsigned char*>(message.c_str()),
         message.length(), hash, &hash_len);
    
    std::stringstream ss;
    for (unsigned int i = 0; i < hash_len; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') 
           << static_cast<int>(hash[i]);
    }
    return ss.str();
}

AuthMiddleware::AuthMiddleware(const AuthConfig& config)
    : config_(config) {
}

bool AuthMiddleware::authenticate(const httplib::Request& req, httplib::Response& res) {
    if (!config_.enable_auth) {
        return true;
    }
    
    // Get authentication header
    auto auth_header = req.get_header_value(config_.auth_header);
    if (auth_header.empty()) {
        json error = {
            {"error", {
                {"code", 401},
                {"message", "Missing authentication"}
            }}
        };
        res.status = 401;
        res.set_content(error.dump(), "application/json");
        return false;
    }
    
    // Validate based on authentication type
    bool authenticated = false;
    if (config_.auth_type == "bearer") {
        if (auth_header.substr(0, 7) != "Bearer ") {
            json error = {
                {"error", {
                    {"code", 401},
                    {"message", "Invalid authentication format"}
                }}
            };
            res.status = 401;
            res.set_content(error.dump(), "application/json");
            return false;
        }
        std::string token = auth_header.substr(7);
        authenticated = verify_bearer_token(token);
    } else if (config_.auth_type == "basic") {
        if (auth_header.substr(0, 6) != "Basic ") {
            json error = {
                {"error", {
                    {"code", 401},
                    {"message", "Invalid authentication format"}
                }}
            };
            res.status = 401;
            res.set_content(error.dump(), "application/json");
            return false;
        }
        std::string credentials = auth_header.substr(6);
        authenticated = verify_basic_auth(credentials);
    }
    
    if (!authenticated) {
        json error = {
            {"error", {
                {"code", 401},
                {"message", "Invalid authentication"}
            }}
        };
        res.status = 401;
        res.set_content(error.dump(), "application/json");
        return false;
    }
    
    return true;
}

std::string AuthMiddleware::generate_token(const std::string& user_id) {
    std::lock_guard<std::mutex> lock(auth_mutex_);
    
    // Clean up expired tokens
    cleanup_expired_tokens();
    
    // Check user token count limit
    size_t user_token_count = 0;
    for (const auto& [token, info] : tokens_) {
        if (info.user_id == user_id && !info.is_revoked) {
            user_token_count++;
        }
    }
    if (user_token_count >= config_.max_tokens_per_user) {
        return "";
    }
    
    // Generate new token
    std::string token = generate_random_string(32);
    auto now = std::chrono::system_clock::now();
    
    TokenInfo info;
    info.user_id = user_id;
    info.created_at = now;
    info.expires_at = now + std::chrono::hours(config_.token_expire_hours);
    info.is_revoked = false;
    
    tokens_[token] = info;
    return token;
}

bool AuthMiddleware::validate_token(const std::string& token) {
    std::lock_guard<std::mutex> lock(auth_mutex_);
    
    auto it = tokens_.find(token);
    if (it == tokens_.end()) {
        return false;
    }
    
    const auto& info = it->second;
    auto now = std::chrono::system_clock::now();
    
    return !info.is_revoked && now < info.expires_at;
}

void AuthMiddleware::revoke_token(const std::string& token) {
    std::lock_guard<std::mutex> lock(auth_mutex_);
    
    auto it = tokens_.find(token);
    if (it != tokens_.end()) {
        it->second.is_revoked = true;
    }
}

bool AuthMiddleware::add_user(const std::string& user_id, const std::string& api_key) {
    std::lock_guard<std::mutex> lock(auth_mutex_);
    
    if (users_.find(user_id) != users_.end()) {
        return false;
    }
    
    users_[user_id] = api_key;
    return true;
}

void AuthMiddleware::remove_user(const std::string& user_id) {
    std::lock_guard<std::mutex> lock(auth_mutex_);
    
    users_.erase(user_id);
    
    // Revoke all tokens for this user
    for (auto& [token, info] : tokens_) {
        if (info.user_id == user_id) {
            info.is_revoked = true;
        }
    }
}

void AuthMiddleware::update_config(const AuthConfig& config) {
    std::lock_guard<std::mutex> lock(auth_mutex_);
    config_ = config;
}

bool AuthMiddleware::verify_bearer_token(const std::string& token) {
    return validate_token(token);
}

bool AuthMiddleware::verify_basic_auth(const std::string& credentials) {
    // Base64 decode
    std::string decoded;
    try {
        decoded = httplib::detail::base64_decode(credentials);
    } catch (...) {
        return false;
    }
    
    // Split username and password
    auto pos = decoded.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    
    std::string user_id = decoded.substr(0, pos);
    std::string api_key = decoded.substr(pos + 1);
    
    std::lock_guard<std::mutex> lock(auth_mutex_);
    auto it = users_.find(user_id);
    return it != users_.end() && it->second == api_key;
}

void AuthMiddleware::cleanup_expired_tokens() {
    auto now = std::chrono::system_clock::now();
    
    for (auto it = tokens_.begin(); it != tokens_.end();) {
        if (it->second.expires_at < now || it->second.is_revoked) {
            it = tokens_.erase(it);
        } else {
            ++it;
        }
    }
}

bool auth_middleware(const httplib::Request& req, httplib::Response& res) {
    if (!auth_middleware_instance) {
        auth_middleware_instance = std::make_unique<AuthMiddleware>();
    }
    return auth_middleware_instance->authenticate(req, res);
}

} // namespace rest
} // namespace api
} // namespace deeppowers