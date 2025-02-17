#include "client_sdk.hpp"
#include "deeppowers.grpc.pb.h"
#include <sstream>
#include <thread>
#include <random>

namespace deeppowers {

Client::Client(const ClientConfig& config)
    : config_(config) {
    
    // Initialize retry configuration
    retry_config_.max_attempts = config.max_retries;
    
    // Initialize channel and stub
    init_channel();
    init_stub();
}

void Client::init_channel() {
    grpc::ChannelArguments args;
    
    // Set channel arguments
    if (config_.enable_compression) {
        args.SetCompressionAlgorithm(GRPC_COMPRESS_GZIP);
    }
    args.SetInt(GRPC_ARG_ENABLE_RETRIES, 1);
    args.SetInt(GRPC_ARG_MAX_RECONNECT_BACKOFF_MS, 
                retry_config_.max_backoff.count());
    
    // Create channel credentials
    std::shared_ptr<grpc::ChannelCredentials> creds;
    if (config_.enable_ssl) {
        grpc::SslCredentialsOptions ssl_opts;
        ssl_opts.pem_root_certs = config_.ssl_cert;
        creds = grpc::SslCredentials(ssl_opts);
    } else {
        creds = grpc::InsecureChannelCredentials();
    }
    
    // Create channel
    channel_ = grpc::CreateCustomChannel(
        config_.server_address, creds, args);
}

void Client::init_stub() {
    stub_ = DeepPowers::NewStub(channel_);
}

GenerationResult Client::generate(const GenerationParams& params) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Create request
    auto request = create_generate_request(params);
    
    // Set timeout
    grpc::ClientContext context;
    context.set_deadline(
        std::chrono::system_clock::now() + 
        std::chrono::milliseconds(config_.timeout_ms));
    
    // Execute request with retry
    GenerateResponse response;
    auto status = retry_with_backoff([&]() {
        return stub_->Generate(&context, request, &response);
    });
    
    // Calculate latency
    auto end_time = std::chrono::high_resolution_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    
    // Process response
    return process_generate_response(response, latency);
}

std::future<GenerationResult> Client::generate_async(const GenerationParams& params) {
    auto handler = std::make_shared<AsyncRequestHandler>(this);
    handler->submit_request(params);
    return handler->result_promise_.get_future();
}

HardwareMetrics Client::get_hardware_metrics() {
    grpc::ClientContext context;
    MetricsRequest request;
    MetricsResponse response;
    
    auto status = retry_with_backoff([&]() {
        return stub_->GetMetrics(&context, request, &response);
    });
    
    if (!status.ok()) {
        handle_error(status);
        return HardwareMetrics{};
    }
    
    return response.hardware();
}

LatencyMetrics Client::get_latency_metrics() {
    grpc::ClientContext context;
    MetricsRequest request;
    MetricsResponse response;
    
    auto status = retry_with_backoff([&]() {
        return stub_->GetMetrics(&context, request, &response);
    });
    
    if (!status.ok()) {
        handle_error(status);
        return LatencyMetrics{};
    }
    
    return response.latency();
}

ThroughputMetrics Client::get_throughput_metrics() {
    grpc::ClientContext context;
    MetricsRequest request;
    MetricsResponse response;
    
    auto status = retry_with_backoff([&]() {
        return stub_->GetMetrics(&context, request, &response);
    });
    
    if (!status.ok()) {
        handle_error(status);
        return ThroughputMetrics{};
    }
    
    return response.throughput();
}

ErrorMetrics Client::get_error_metrics() {
    grpc::ClientContext context;
    MetricsRequest request;
    MetricsResponse response;
    
    auto status = retry_with_backoff([&]() {
        return stub_->GetMetrics(&context, request, &response);
    });
    
    if (!status.ok()) {
        handle_error(status);
        return ErrorMetrics{};
    }
    
    return response.errors();
}

SchedulerStatusResponse Client::get_scheduler_status() {
    grpc::ClientContext context;
    SchedulerStatusRequest request;
    SchedulerStatusResponse response;
    
    auto status = retry_with_backoff([&]() {
        return stub_->GetSchedulerStatus(&context, request, &response);
    });
    
    if (!status.ok()) {
        handle_error(status);
        return SchedulerStatusResponse{};
    }
    
    return response;
}

void Client::update_config(const ClientConfig& config) {
    config_ = config;
    retry_config_.max_attempts = config.max_retries;
    
    // Reinitialize channel and stub
    init_channel();
    init_stub();
}

template<typename F>
auto Client::retry_with_backoff(F&& func) -> decltype(func()) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.5, 1.5);
    
    auto backoff = retry_config_.initial_backoff;
    
    for (size_t attempt = 1; attempt <= retry_config_.max_attempts; ++attempt) {
        auto status = func();
        
        if (status.ok()) {
            return status;
        }
        
        // Update metrics
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            metrics_.error_count++;
            metrics_.retry_count++;
        }
        
        // Check if we should retry
        if (attempt == retry_config_.max_attempts ||
            status.error_code() == grpc::StatusCode::INVALID_ARGUMENT ||
            status.error_code() == grpc::StatusCode::UNAUTHENTICATED) {
            return status;
        }
        
        // Calculate jittered backoff
        auto jittered_backoff = 
            std::chrono::milliseconds(
                static_cast<int64_t>(backoff.count() * dis(gen)));
        
        std::this_thread::sleep_for(jittered_backoff);
        
        // Increase backoff
        backoff = std::min(
            backoff * retry_config_.backoff_multiplier,
            retry_config_.max_backoff);
    }
    
    return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "Max retries exceeded");
}

GenerateRequest Client::create_generate_request(const GenerationParams& params) {
    GenerateRequest request;
    
    // Set request fields
    request.set_prompt(params.prompt);
    request.set_max_tokens(params.max_tokens);
    request.set_temperature(params.temperature);
    request.set_top_p(params.top_p);
    
    for (const auto& stop : params.stop) {
        request.add_stop(stop);
    }
    
    // Generate request ID if not provided
    if (request.request_id().empty()) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 0xFFFFFFFF);
        
        std::stringstream ss;
        ss << std::hex << dis(gen);
        request.set_request_id(ss.str());
    }
    
    return request;
}

GenerationResult Client::process_generate_response(
    const GenerateResponse& response,
    const std::chrono::microseconds& latency) {
    
    GenerationResult result;
    
    // Copy response fields
    result.text = response.text();
    result.logprobs.assign(
        response.logprobs().begin(),
        response.logprobs().end());
    result.tokens.assign(
        response.tokens().begin(),
        response.tokens().end());
    result.latency = latency;
    
    // Update metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.request_count++;
        metrics_.avg_latency_ms = 
            (metrics_.avg_latency_ms * (metrics_.request_count - 1) +
             latency.count() / 1000.0) / metrics_.request_count;
    }
    
    return result;
}

void Client::handle_error(const grpc::Status& status) {
    std::stringstream ss;
    ss << "gRPC error: " << status.error_code() << ": " 
       << status.error_message();
    
    if (!status.error_details().empty()) {
        ss << " (" << status.error_details() << ")";
    }
    
    throw std::runtime_error(ss.str());
}

AsyncRequestHandler::AsyncRequestHandler(Client* client)
    : client_(client) {
}

AsyncRequestHandler::~AsyncRequestHandler() {
    wait_for_completion();
}

void AsyncRequestHandler::submit_request(const GenerationParams& params) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (state_ != State::INITIAL) {
        throw std::runtime_error("Request already submitted");
    }
    
    state_ = State::PROCESSING;
    
    // Submit request asynchronously
    std::thread([this, params]() {
        try {
            auto result = client_->generate(params);
            
            std::lock_guard<std::mutex> lock(state_mutex_);
            result_ = result;
            state_ = State::COMPLETED;
            result_promise_.set_value(result);
            
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            error_message_ = e.what();
            state_ = State::ERROR;
            result_promise_.set_exception(
                std::current_exception());
        }
    }).detach();
}

void AsyncRequestHandler::wait_for_completion() {
    std::unique_lock<std::mutex> lock(state_mutex_);
    while (state_ == State::PROCESSING) {
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        lock.lock();
    }
}

bool AsyncRequestHandler::is_completed() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return state_ == State::COMPLETED;
}

bool AsyncRequestHandler::has_error() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return state_ == State::ERROR;
}

const std::string& AsyncRequestHandler::error_message() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return error_message_;
}

GenerationResult AsyncRequestHandler::get_result() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (state_ == State::ERROR) {
        throw std::runtime_error(error_message_);
    }
    
    if (state_ != State::COMPLETED) {
        throw std::runtime_error("Result not available");
    }
    
    return result_;
}

} // namespace deeppowers 