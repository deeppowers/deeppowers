#include "logging.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>

namespace deeppowers {
namespace common {

void Logger::initialize(const LogConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    
    if (config_.file_output) {
        // Create log directory
        std::filesystem::create_directories(config_.log_dir);
        
        // Open log file
        std::string log_path = config_.log_dir + "/" + config_.log_file;
        log_file_.open(log_path, std::ios::app);
        if (!log_file_) {
            std::cerr << "Failed to open log file: " << log_path << std::endl;
        }
        
        // Get current file size
        current_file_size_ = std::filesystem::file_size(log_path);
    }
}

void Logger::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (log_file_.is_open()) {
        log_file_.close();
    }
}

void Logger::log(LogLevel level, const char* file, int line, 
                const char* func, const std::string& msg) {
    if (level < config_.level) return;
    
    std::string formatted_msg = format_message(level, file, line, func, msg);
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Output to console
    if (config_.console_output) {
        write_to_console(level, formatted_msg);
    }
    
    // Output to file
    if (config_.file_output) {
        write_to_file(formatted_msg);
    }
    
    // Check if log file needs rotation
    if (current_file_size_ >= config_.max_file_size) {
        rotate_log_files();
    }
}

void Logger::update_config(const LogConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // If file output status changes
    if (config_.file_output != config.file_output) {
        if (config.file_output) {
            // Open log file
            std::string log_path = config.log_dir + "/" + config.log_file;
            log_file_.open(log_path, std::ios::app);
        } else {
            // Close log file
            if (log_file_.is_open()) {
                log_file_.close();
            }
        }
    }
    
    config_ = config;
}

void Logger::write_to_console(LogLevel level, const std::string& msg) {
    // Set different levels of colors
    const char* color = "\033[0m";  // Default color
    switch (level) {
        case LogLevel::DEBUG:
            color = "\033[36m";  // Cyan
            break;
        case LogLevel::INFO:
            color = "\033[32m";  // Green
            break;
        case LogLevel::WARNING:
            color = "\033[33m";  // Yellow
            break;
        case LogLevel::ERROR:
            color = "\033[31m";  // Red
            break;
        case LogLevel::FATAL:
            color = "\033[35m";  // Purple
            break;
    }
    
    std::cout << color << msg << "\033[0m" << std::endl;
}

void Logger::write_to_file(const std::string& msg) {
    if (!log_file_.is_open()) return;
    
    log_file_ << msg << std::endl;
    log_file_.flush();
    
    current_file_size_ += msg.size() + 1;  // +1 for newline
}

void Logger::rotate_log_files() {
    if (!log_file_.is_open()) return;
    
    // Close current log file
    log_file_.close();
    
    // Build file path
    std::string base_path = config_.log_dir + "/" + config_.log_file;
    
    // Delete oldest log file
    std::string oldest_log = base_path + "." + std::to_string(config_.max_files - 1);
    std::filesystem::remove(oldest_log);
    
    // Rename existing log files
    for (int i = config_.max_files - 2; i >= 0; --i) {
        std::string old_name = base_path + (i > 0 ? "." + std::to_string(i) : "");
        std::string new_name = base_path + "." + std::to_string(i + 1);
        if (std::filesystem::exists(old_name)) {
            std::filesystem::rename(old_name, new_name);
        }
    }
    
    // Open new log file
    log_file_.open(base_path, std::ios::out);
    current_file_size_ = 0;
}

std::string Logger::format_message(LogLevel level, const char* file, int line,
                                 const char* func, const std::string& msg) {
    std::ostringstream oss;
    
    // Get current time
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    // Format time
    oss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S")
        << '.' << std::setfill('0') << std::setw(3) << now_ms.count() << " ";
    
    // Add log level
    switch (level) {
        case LogLevel::DEBUG:   oss << "[DEBUG]   "; break;
        case LogLevel::INFO:    oss << "[INFO]    "; break;
        case LogLevel::WARNING: oss << "[WARNING] "; break;
        case LogLevel::ERROR:   oss << "[ERROR]   "; break;
        case LogLevel::FATAL:   oss << "[FATAL]   "; break;
    }
    
    // Add file information
    oss << "[" << file << ":" << line << "] ";
    
    // Add function name
    oss << "[" << func << "] ";
    
    // Add message
    oss << msg;
    
    return oss.str();
}

} // namespace common
} // namespace deeppowers