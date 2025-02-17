#include "logging_middleware.hpp"
#include "../../../common/logging.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <regex>
#include <iomanip>

namespace deeppowers {
namespace api {
namespace rest {

using json = nlohmann::json;
namespace fs = std::filesystem;

// Global instance
std::unique_ptr<LoggingMiddleware> logging_middleware_instance;

// Constants
constexpr size_t MAX_LOG_FILE_SIZE = 100 * 1024 * 1024;  // 100MB
constexpr auto LOG_ROTATION_INTERVAL = std::chrono::hours(24);

LoggingMiddleware::LoggingMiddleware(const LoggingConfig& config)
    : config_(config) {
    open_log_file();
}

LoggingMiddleware::~LoggingMiddleware() {
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (log_file_.file.is_open()) {
        log_file_.file.close();
    }
}

void LoggingMiddleware::log_request(const httplib::Request& req) {
    if (!config_.enable_request_logging) {
        return;
    }

    // Prepare log fields
    std::unordered_map<std::string, std::string> fields;
    fields["type"] = "request";
    fields["timestamp"] = std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count());
    fields["method"] = req.method;
    fields["path"] = req.path;
    fields["remote_addr"] = req.remote_addr;
    fields["user_agent"] = req.get_header_value("User-Agent");
    fields["request_id"] = req.get_header_value("X-Request-ID");

    // Log request body if present
    if (!req.body.empty() && req.body.size() <= config_.max_body_size) {
        fields["body"] = config_.mask_sensitive_data ? 
            mask_sensitive_data(req.body) : req.body;
    }

    // Write log entry
    write_log_entry(format_log_entry("request", fields));
}

void LoggingMiddleware::log_response(const httplib::Request& req,
                                   const httplib::Response& res,
                                   const std::chrono::microseconds& duration) {
    if (!config_.enable_response_logging) {
        return;
    }

    // Prepare log fields
    std::unordered_map<std::string, std::string> fields;
    fields["type"] = "response";
    fields["timestamp"] = std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count());
    fields["method"] = req.method;
    fields["path"] = req.path;
    fields["status"] = std::to_string(res.status);
    fields["duration_us"] = std::to_string(duration.count());
    fields["request_id"] = req.get_header_value("X-Request-ID");

    // Log response body if present
    if (!res.body.empty() && res.body.size() <= config_.max_body_size) {
        fields["body"] = config_.mask_sensitive_data ? 
            mask_sensitive_data(res.body) : res.body;
    }

    // Write log entry
    write_log_entry(format_log_entry("response", fields));

    // Update performance stats
    if (config_.enable_performance_logging) {
        log_performance(req.path, duration, req.body.size(), res.body.size());
    }
}

void LoggingMiddleware::log_performance(const std::string& endpoint,
                                      const std::chrono::microseconds& duration,
                                      size_t request_size,
                                      size_t response_size) {
    std::lock_guard<std::mutex> lock(log_mutex_);

    auto& stats = endpoint_stats_[endpoint];
    stats.request_count++;
    stats.total_duration_us += duration.count();
    stats.total_request_size += request_size;
    stats.total_response_size += response_size;

    // Log performance stats periodically
    if (stats.request_count % 100 == 0) {  // Every 100 requests
        std::unordered_map<std::string, std::string> fields;
        fields["type"] = "performance";
        fields["endpoint"] = endpoint;
        fields["request_count"] = std::to_string(stats.request_count);
        fields["avg_duration_us"] = std::to_string(
            stats.total_duration_us / stats.request_count);
        fields["avg_request_size"] = std::to_string(
            stats.total_request_size / stats.request_count);
        fields["avg_response_size"] = std::to_string(
            stats.total_response_size / stats.request_count);

        write_log_entry(format_log_entry("performance", fields));
    }
}

void LoggingMiddleware::update_config(const LoggingConfig& config) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    bool path_changed = config_.log_path != config.log_path;
    config_ = config;

    if (path_changed) {
        if (log_file_.file.is_open()) {
            log_file_.file.close();
        }
        open_log_file();
    }
}

void LoggingMiddleware::open_log_file() {
    // Create log directory if not exists
    fs::path log_path(config_.log_path);
    fs::create_directories(log_path.parent_path());

    // Open log file in append mode
    log_file_.file.open(config_.log_path, std::ios::app);
    if (!log_file_.file) {
        LOG_ERROR("Failed to open log file: " + config_.log_path);
        return;
    }

    // Get current file size
    log_file_.file.seekp(0, std::ios::end);
    log_file_.current_size = log_file_.file.tellp();
    log_file_.last_rotation = std::chrono::system_clock::now();
}

void LoggingMiddleware::rotate_log_file() {
    auto now = std::chrono::system_clock::now();
    bool should_rotate = false;

    // Check file size
    if (log_file_.current_size >= MAX_LOG_FILE_SIZE) {
        should_rotate = true;
    }

    // Check time interval
    if (now - log_file_.last_rotation >= LOG_ROTATION_INTERVAL) {
        should_rotate = true;
    }

    if (should_rotate) {
        // Close current file
        log_file_.file.close();

        // Generate new filename with timestamp
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << config_.log_path << "." 
           << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
        
        // Rename current file
        fs::rename(config_.log_path, ss.str());

        // Open new file
        open_log_file();
    }
}

void LoggingMiddleware::write_log_entry(const std::string& entry) {
    std::lock_guard<std::mutex> lock(log_mutex_);

    // Check if file needs rotation
    rotate_log_file();

    // Write log entry
    if (log_file_.file.is_open()) {
        log_file_.file << entry << std::endl;
        log_file_.current_size += entry.size() + 1;
        log_file_.file.flush();
    }
}

std::string LoggingMiddleware::format_log_entry(
    const std::string& type,
    const std::unordered_map<std::string, std::string>& fields) {
    
    if (config_.log_format == "json") {
        json log_entry;
        for (const auto& [key, value] : fields) {
            log_entry[key] = value;
        }
        return log_entry.dump();
    } else {
        // Text format
        std::stringstream ss;
        ss << "[" << type << "] ";
        for (const auto& [key, value] : fields) {
            if (key != "type") {
                ss << key << "=" << value << " ";
            }
        }
        return ss.str();
    }
}

std::string LoggingMiddleware::mask_sensitive_data(const std::string& data) {
    // Define patterns for sensitive data
    static const std::vector<std::pair<std::regex, std::string>> patterns = {
        {std::regex("\"password\"\\s*:\\s*\"[^\"]+\""), "\"password\":\"***\""},
        {std::regex("\"api_key\"\\s*:\\s*\"[^\"]+\""), "\"api_key\":\"***\""},
        {std::regex("\"token\"\\s*:\\s*\"[^\"]+\""), "\"token\":\"***\""},
        {std::regex("\\b\\d{16}\\b"), "****-****-****-****"},  // Credit card numbers
        {std::regex("\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\\b"), "***@***.***"}  // Email addresses
    };

    std::string masked = data;
    for (const auto& [pattern, replacement] : patterns) {
        masked = std::regex_replace(masked, pattern, replacement);
    }
    return masked;
}

void log_request_middleware(const httplib::Request& req) {
    if (!logging_middleware_instance) {
        logging_middleware_instance = std::make_unique<LoggingMiddleware>();
    }
    logging_middleware_instance->log_request(req);
}

void log_response_middleware(const httplib::Request& req,
                           const httplib::Response& res,
                           const std::chrono::microseconds& duration) {
    if (!logging_middleware_instance) {
        logging_middleware_instance = std::make_unique<LoggingMiddleware>();
    }
    logging_middleware_instance->log_response(req, res, duration);
}

} // namespace rest
} // namespace api
} // namespace deeppowers 