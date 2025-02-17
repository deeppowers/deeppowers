#include "error_middleware.hpp"
#include "../../../common/logging.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <ctime>
#include <execinfo.h>
#include <cxxabi.h>
#include <dlfcn.h>

namespace deeppowers {
namespace api {
namespace rest {

using json = nlohmann::json;

// Global instance
std::unique_ptr<ErrorMiddleware> error_middleware_instance;

ErrorMiddleware::ErrorMiddleware(const ErrorHandlerConfig& config)
    : config_(config) {
    
    // Register default error handler
    register_error_handler(400, [](const auto& req, auto& res, const auto& msg) {
        json error = {
            {"error", {
                {"code", 400},
                {"message", msg},
                {"type", "BadRequest"}
            }}
        };
        res.set_content(error.dump(), "application/json");
    });
    
    register_error_handler(401, [](const auto& req, auto& res, const auto& msg) {
        json error = {
            {"error", {
                {"code", 401},
                {"message", msg},
                {"type", "Unauthorized"}
            }}
        };
        res.set_content(error.dump(), "application/json");
    });
    
    register_error_handler(403, [](const auto& req, auto& res, const auto& msg) {
        json error = {
            {"error", {
                {"code", 403},
                {"message", msg},
                {"type", "Forbidden"}
            }}
        };
        res.set_content(error.dump(), "application/json");
    });
    
    register_error_handler(404, [](const auto& req, auto& res, const auto& msg) {
        json error = {
            {"error", {
                {"code", 404},
                {"message", msg},
                {"type", "NotFound"}
            }}
        };
        res.set_content(error.dump(), "application/json");
    });
    
    register_error_handler(429, [](const auto& req, auto& res, const auto& msg) {
        json error = {
            {"error", {
                {"code", 429},
                {"message", msg},
                {"type", "TooManyRequests"}
            }}
        };
        res.set_content(error.dump(), "application/json");
    });
    
    register_error_handler(500, [](const auto& req, auto& res, const auto& msg) {
        json error = {
            {"error", {
                {"code", 500},
                {"message", msg},
                {"type", "InternalServerError"}
            }}
        };
        res.set_content(error.dump(), "application/json");
    });
}

void ErrorMiddleware::handle_error(const httplib::Request& req,
                                 httplib::Response& res,
                                 int status_code,
                                 const std::string& message,
                                 const std::exception* e) {
    std::lock_guard<std::mutex> lock(error_mutex_);
    
    // Record error log
    if (config_.enable_error_logging) {
        log_error(message, req, status_code);
    }
    
    // Get stack trace
    std::string stack_trace;
    if (config_.enable_detailed_errors && e != nullptr) {
        stack_trace = get_stack_trace(e);
    }
    
    // Set response status code
    res.status = status_code;
    
    // Find and execute error handler
    auto it = error_handlers_.find(status_code);
    if (it != error_handlers_.end()) {
        it->second(req, res, format_error_response(status_code, message, stack_trace));
    } else {
        // Use default error handler
        json error = {
            {"error", {
                {"code", status_code},
                {"message", message},
                {"type", "Error"}
            }}
        };
        if (!stack_trace.empty()) {
            error["error"]["stack_trace"] = stack_trace;
        }
        res.set_content(error.dump(), "application/json");
    }
}

void ErrorMiddleware::handle_exception(const httplib::Request& req,
                                     httplib::Response& res,
                                     const std::exception& e) {
    handle_error(req, res, 500, e.what(), &e);
}

void ErrorMiddleware::register_error_handler(int status_code, ErrorHandler handler) {
    std::lock_guard<std::mutex> lock(error_mutex_);
    error_handlers_[status_code] = std::move(handler);
}

void ErrorMiddleware::update_config(const ErrorHandlerConfig& config) {
    std::lock_guard<std::mutex> lock(error_mutex_);
    config_ = config;
}

void ErrorMiddleware::log_error(const std::string& message,
                              const httplib::Request& req,
                              int status_code) {
    std::ofstream log_file(config_.error_log_path, std::ios::app);
    if (!log_file) {
        return;
    }
    
    // Get current time
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    
    // Format log message
    log_file << "["  << std::ctime(&time) << "] "
             << status_code << " " << req.method << " " << req.path << "\n"
             << "Error: " << message << "\n"
             << "Client IP: " << req.remote_addr << "\n"
             << "User Agent: " << req.get_header_value("User-Agent") << "\n"
             << "Request ID: " << req.get_header_value("X-Request-ID") << "\n"
             << "----------------------------------------\n";
}

std::string ErrorMiddleware::get_stack_trace(const std::exception* e) {
    std::stringstream ss;
    
    // Get stack trace
    void* array[50];
    int size = backtrace(array, 50);
    char** messages = backtrace_symbols(array, size);
    
    if (messages == nullptr) {
        return "";
    }
    
    // Parse stack trace
    for (int i = 0; i < size && i < config_.max_stack_trace_depth; ++i) {
        char* mangled_name = nullptr;
        char* offset_begin = nullptr;
        char* offset_end = nullptr;
        
        // Find function name
        for (char* p = messages[i]; *p; ++p) {
            if (*p == '(') {
                mangled_name = p;
            } else if (*p == '+') {
                offset_begin = p;
            } else if (*p == ')') {
                offset_end = p;
                break;
            }
        }
        
        if (mangled_name && offset_begin && offset_end) {
            *mangled_name++ = '\0';
            *offset_begin++ = '\0';
            *offset_end = '\0';
            
            int status;
            char* demangled = abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);
            
            ss << "#" << i << " " << messages[i] << ": ";
            if (status == 0) {
                ss << demangled;
                free(demangled);
            } else {
                ss << mangled_name;
            }
            ss << "+" << offset_begin << "\n";
        } else {
            ss << "#" << i << " " << messages[i] << "\n";
        }
    }
    
    free(messages);
    return ss.str();
}

std::string ErrorMiddleware::format_error_response(int status_code,
                                                 const std::string& message,
                                                 const std::string& stack_trace) {
    json error = {
        {"error", {
            {"code", status_code},
            {"message", message}
        }}
    };
    
    if (!stack_trace.empty()) {
        error["error"]["stack_trace"] = stack_trace;
    }
    
    return error.dump();
}

void error_middleware(const httplib::Request& req,
                     httplib::Response& res,
                     int status_code,
                     const std::string& message) {
    if (!error_middleware_instance) {
        error_middleware_instance = std::make_unique<ErrorMiddleware>();
    }
    error_middleware_instance->handle_error(req, res, status_code, message);
}

} // namespace rest
} // namespace api
} // namespace deeppowers 