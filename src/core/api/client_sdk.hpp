#pragma once

#include <string>
#include <vector>
#include <memory>
#include <future>
#include <chrono>
#include <mutex>
#include <grpcpp/grpcpp.h>

namespace deeppowers {

// Forward declarations
class DeepPowers;
class AsyncRequestHandler;

// Client configuration
struct ClientConfig {
    std::string server_address;
    bool enable_ssl = false;
    std::string ssl_cert;
    int timeout_ms = 30000;
    int max_retries = 3;
    bool enable_compression = true;
};

// Retry configuration
struct RetryConfig {
    size_t max_attempts = 3;
    std::chrono::milliseconds initial_backoff{100};
    std::chrono::milliseconds max_backoff{5000};
    double backoff_multiplier = 2.0;
};

// Generation parameters
struct GenerationParams {
    std::string prompt;
    int max_tokens = 100;
    float temperature = 0.7f;
    float top_p = 1.0f;
    std::vector<std::string> stop;
};

// Generation result
struct GenerationResult {
    std::string text;
    std::vector<float> logprobs;
    std::vector<std::string> tokens;
    std::chrono::microseconds latency;
};

// Client metrics
struct ClientMetrics {
    size_t request_count = 0;
    size_t error_count = 0;
    size_t retry_count = 0;
    double avg_latency_ms = 0.0;
};

// Main client class
class Client {
public:
    explicit Client(const ClientConfig& config);
    
    // Generation method
    GenerationResult generate(const GenerationParams& params);
    std::future<GenerationResult> generate_async(const GenerationParams& params);
    
    // Metrics methods
    HardwareMetrics get_hardware_metrics();
    LatencyMetrics get_latency_metrics();
    ThroughputMetrics get_throughput_metrics();
    ErrorMetrics get_error_metrics();
    
    // Scheduler status
    SchedulerStatusResponse get_scheduler_status();
    
    // Configuration update
    void update_config(const ClientConfig& config);
    
private:
    friend class AsyncRequestHandler;
    
    // Initialize method
    void init_channel();
    void init_stub();
    
    // Request processing
    GenerateRequest create_generate_request(const GenerationParams& params);
    GenerationResult process_generate_response(
        const GenerateResponse& response,
        const std::chrono::microseconds& latency);
    
    // Error handling
    void handle_error(const grpc::Status& status);
    
    // Retry logic
    template<typename F>
    auto retry_with_backoff(F&& func) -> decltype(func());
    
    // Member variables
    ClientConfig config_;
    RetryConfig retry_config_;
    ClientMetrics metrics_;
    std::mutex metrics_mutex_;
    
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<DeepPowers::Stub> stub_;
};

// Async request handler
class AsyncRequestHandler {
public:
    explicit AsyncRequestHandler(Client* client);
    ~AsyncRequestHandler();
    
    void submit_request(const GenerationParams& params);
    void wait_for_completion();
    
    bool is_completed() const;
    bool has_error() const;
    const std::string& error_message() const;
    GenerationResult get_result();
    
private:
    friend class Client;
    
    enum class State {
        INITIAL,
        PROCESSING,
        COMPLETED,
        ERROR
    };
    
    Client* client_;
    State state_ = State::INITIAL;
    GenerationResult result_;
    std::string error_message_;
    std::promise<GenerationResult> result_promise_;
    mutable std::mutex state_mutex_;
};

} // namespace deeppowers 