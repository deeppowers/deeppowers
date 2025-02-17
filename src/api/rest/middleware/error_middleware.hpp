#pragma once

#include <httplib.h>
#include <string>
#include <functional>
#include <unordered_map>
#include <mutex>

namespace deeppowers {
namespace api {
namespace rest {

// Error handling configuration
struct ErrorHandlerConfig {
    bool enable_detailed_errors = false;    // Whether to return detailed error information
    bool enable_error_logging = true;       // Whether to log errors
    std::string error_log_path = "logs/error.log";  // Error log path
    size_t max_stack_trace_depth = 20;     // Maximum stack trace depth
};

// Error handling middleware class
class ErrorMiddleware {
public:
    explicit ErrorMiddleware(const ErrorHandlerConfig& config = ErrorHandlerConfig());
    
    // Error handling
    void handle_error(const httplib::Request& req, 
                     httplib::Response& res,
                     int status_code,
                     const std::string& message,
                     const std::exception* e = nullptr);
    
    // Exception handling
    void handle_exception(const httplib::Request& req,
                         httplib::Response& res,
                         const std::exception& e);
    
    // Error handler registration
    using ErrorHandler = std::function<void(const httplib::Request&,
                                          httplib::Response&,
                                          const std::string&)>;
    void register_error_handler(int status_code, ErrorHandler handler);
    
    // Configuration access
    const ErrorHandlerConfig& config() const { return config_; }
    void update_config(const ErrorHandlerConfig& config);

private:
    // Internal helper methods
    void log_error(const std::string& message,
                  const httplib::Request& req,
                  int status_code);
    std::string get_stack_trace(const std::exception* e);
    std::string format_error_response(int status_code,
                                    const std::string& message,
                                    const std::string& stack_trace = "");
    
    // Member variables
    ErrorHandlerConfig config_;
    std::unordered_map<int, ErrorHandler> error_handlers_;
    std::mutex error_mutex_;
};

// Global middleware instance
extern std::unique_ptr<ErrorMiddleware> error_middleware_instance;

// Middleware function
void error_middleware(const httplib::Request& req,
                     httplib::Response& res,
                     int status_code,
                     const std::string& message);

} // namespace rest
} // namespace api
} // namespace deeppowers 