#pragma once

#include <string>
#include <sstream>
#include <mutex>
#include <chrono>

namespace deeppowers {
namespace common {

// Log levels
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    FATAL
};

// Log configuration
struct LogConfig {
    LogLevel level = LogLevel::INFO;          // Log level
    std::string log_dir = "./logs";          // Log directory
    std::string log_file = "deeppowers.log"; // Log file name
    size_t max_file_size = 100 * 1024 * 1024;// Maximum file size (100MB)
    size_t max_files = 10;                   // Maximum number of files
    bool console_output = true;              // Whether to output to console
    bool file_output = true;                 // Whether to output to file
    bool async_logging = true;               // Whether to log asynchronously
};

// Logger class
class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }
    
    // Initialize and clean up
    void initialize(const LogConfig& config);
    void shutdown();
    
    // Log
    void log(LogLevel level, const char* file, int line, const char* func, const std::string& msg);
    
    // Log level control
    void set_level(LogLevel level) { config_.level = level; }
    LogLevel get_level() const { return config_.level; }
    
    // Configuration access
    const LogConfig& config() const { return config_; }
    void update_config(const LogConfig& config);

private:
    Logger() = default;
    ~Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    // Internal helper methods
    void write_to_console(LogLevel level, const std::string& msg);
    void write_to_file(const std::string& msg);
    void rotate_log_files();
    std::string format_message(LogLevel level, const char* file, int line, 
                             const char* func, const std::string& msg);
    
    // Member variables
    LogConfig config_;
    std::mutex mutex_;
    std::ofstream log_file_;
    size_t current_file_size_ = 0;
};

// Log macros
#define LOG_DEBUG(msg) \
    if (deeppowers::common::Logger::instance().get_level() <= deeppowers::common::LogLevel::DEBUG) \
        deeppowers::common::Logger::instance().log(deeppowers::common::LogLevel::DEBUG, __FILE__, __LINE__, __func__, msg)

#define LOG_INFO(msg) \
    if (deeppowers::common::Logger::instance().get_level() <= deeppowers::common::LogLevel::INFO) \
        deeppowers::common::Logger::instance().log(deeppowers::common::LogLevel::INFO, __FILE__, __LINE__, __func__, msg)

#define LOG_WARNING(msg) \
    if (deeppowers::common::Logger::instance().get_level() <= deeppowers::common::LogLevel::WARNING) \
        deeppowers::common::Logger::instance().log(deeppowers::common::LogLevel::WARNING, __FILE__, __LINE__, __func__, msg)

#define LOG_ERROR(msg) \
    if (deeppowers::common::Logger::instance().get_level() <= deeppowers::common::LogLevel::ERROR) \
        deeppowers::common::Logger::instance().log(deeppowers::common::LogLevel::ERROR, __FILE__, __LINE__, __func__, msg)

#define LOG_FATAL(msg) \
    deeppowers::common::Logger::instance().log(deeppowers::common::LogLevel::FATAL, __FILE__, __LINE__, __func__, msg)

} // namespace common
} // namespace deeppowers 