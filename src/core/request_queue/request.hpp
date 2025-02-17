#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace deeppowers {

// Request status
enum class RequestStatus {
    PENDING,    // Waiting to be processed
    RUNNING,    // Currently being processed
    COMPLETED,  // Completed
    FAILED      // Failed
};

// Request priority
enum class RequestPriority {
    LOW,
    NORMAL,
    HIGH,
    CRITICAL
};

// Request configuration options
struct RequestConfig {
    size_t max_tokens = 2048;           // Maximum number of generated tokens
    float temperature = 1.0f;           // Sampling temperature
    float top_p = 1.0f;                 // Top-p sampling probability threshold
    float top_k = 0.0f;                 // Top-k sampling k value
    float presence_penalty = 0.0f;      // Presence penalty
    float frequency_penalty = 0.0f;     // Frequency penalty
    std::vector<std::string> stop_tokens;  // Stop token list
};

// Request result
struct RequestResult {
    std::vector<std::string> generated_texts;  // Generated texts
    std::vector<float> logprobs;              // Token log probabilities
    std::vector<std::vector<std::string>> top_tokens;  // Top tokens for each position
    std::chrono::microseconds processing_time;  // Processing time
    std::string error_message;                  // Error message (if any)
};

// Inference request
class Request {
public:
    Request(const std::string& request_id,
            const std::string& prompt,
            RequestPriority priority = RequestPriority::NORMAL);

    // Basic attribute access
    const std::string& id() const { return id_; }
    const std::string& prompt() const { return prompt_; }
    RequestStatus status() const { return status_; }
    RequestPriority priority() const { return priority_; }
    const RequestConfig& config() const { return config_; }
    RequestConfig& mutable_config() { return config_; }
    
    // Time-related
    std::chrono::system_clock::time_point created_time() const { return created_time_; }
    std::chrono::system_clock::time_point start_time() const { return start_time_; }
    std::chrono::system_clock::time_point end_time() const { return end_time_; }
    
    // Result access
    const RequestResult& result() const { return result_; }
    RequestResult& mutable_result() { return result_; }
    
    // Status update
    void set_status(RequestStatus status);
    void mark_started();
    void mark_completed();
    void mark_failed(const std::string& error_message);
    
    // Calculate request wait time and processing time
    std::chrono::microseconds wait_time() const;
    std::chrono::microseconds processing_time() const;

private:
    std::string id_;                    // Request ID
    std::string prompt_;                // Input prompt
    RequestStatus status_;              // Current status
    RequestPriority priority_;          // Priority
    RequestConfig config_;              // Configuration options
    RequestResult result_;              // Request result
    
    // Timestamp
    std::chrono::system_clock::time_point created_time_;  // Creation time
    std::chrono::system_clock::time_point start_time_;    // Start processing time
    std::chrono::system_clock::time_point end_time_;      // End time
};

using RequestPtr = std::shared_ptr<Request>;

} // namespace deeppowers 